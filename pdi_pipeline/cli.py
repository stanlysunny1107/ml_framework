import argparse
import csv
import random
from pathlib import Path
import numpy as np
import torch
from .data import *
from .model import BENet
from .inference import Predictor
from .report import confusion, metrics, export_report, curves


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def evaluate(predictor, m, split):
    results = {}
    total = np.zeros((3, 3), np.int64)
    for r in m["patients"]:
        if r["id"] not in m["splits"][split]:
            continue
        if (
            signature(r["cube"], m["signature"]["preprocessing"])
            != predictor.ck["input_signature"]
        ):
            raise ValueError("Input signature changed")
        c = load_cube(r["cube"])
        a = mask(r["gt"], c.shape[1:])
        cm = confusion(a, predictor.predict_array(c))
        total += cm
        results[r["id"]] = metrics(cm)

    def avg(key):
        values = [v[key] for v in results.values() if v[key] is not None]
        return float(np.mean(values)) if values else None

    return {
        "split": split,
        "checkpoint_epoch": predictor.ck.get("epoch"),
        "selection_metric": "pooled_pixels.mean_iou on validation only",
        "patients": results,
        "pooled_pixels": metrics(total),
        "patient_macro_mean_iou": avg("mean_iou"),
        "patient_macro_pixel_accuracy": avg("pixel_accuracy"),
    }


def loss_eval(model, loader, device):
    model.eval()
    total = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            count = int((y != -100).sum())
            total += float(
                torch.nn.functional.cross_entropy(
                    model(x), y, ignore_index=-100, reduction="sum"
                )
            )
            n += count
    return total / n if n else None


def context(c):
    out = Path(c["output"])
    m = read_json(out / "split_manifest.json")
    validate_manifest(m)
    norm = read_json(out / "normalization.json")
    if (
        norm["manifest_hash"] != digest(m)
        or norm["training_ids"] != m["splits"]["train"]
    ):
        raise ValueError("Normalization provenance mismatch")
    for r in m["patients"]:
        if (
            file_hash(r["cube"]) != r["cube_sha256"]
            or file_hash(r["gt"]) != r["mask_sha256"]
            or file_hash(Path(r["cube"]).with_suffix(".hdr")) != r["header_sha256"]
        ):
            raise ValueError("Dataset content changed since inspection")
    return out, m, norm


def train(c, resume=None, max_steps=None):
    if c["patch_size"] < 32 or c["patch_size"] % 16:
        raise ValueError("Use patch size >=32 divisible by 16")
    seed_all(c["seed"])
    out, m, norm = context(c)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = [
        Tiles(
            [r for r in m["patients"] if r["id"] in m["splits"][s]],
            c["patch_size"],
            norm,
            s == "train",
        )
        for s in ("train", "validation")
    ]
    generator = torch.Generator().manual_seed(c["seed"])
    loaders = [
        torch.utils.data.DataLoader(
            ds,
            batch_size=c["batch_size"],
            shuffle=i == 0,
            num_workers=0,
            generator=generator,
        )
        for i, ds in enumerate(datasets)
    ]
    model = BENet(c["bands"], 3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=c["scheduler_step"], gamma=0.5
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=c["mixed_precision"] and device.type == "cuda"
    )
    start = 0
    best = -1.0
    history = []
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            ck["architecture"] != {"name": "BENet", "in_ch": c["bands"], "out_ch": 3}
            or ck["class_mapping"] != LABELS
        ):
            raise ValueError("Resume architecture/class mismatch")
        if ck["manifest_hash"] != digest(m) or ck["normalization"] != norm:
            raise ValueError("Resume dataset provenance mismatch")
        for key in c:
            if (
                key not in ("epochs", "checkpoint_interval", "output", "dataset_root")
                and c[key] != ck["training_config"][key]
            ):
                raise ValueError(f"Resume configuration changed: {key}")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start = ck["epoch"]
        best = ck["best_mean_iou"]
        history = ck["history"]
        random.setstate(ck["rng_python"])
        ns = ck["rng_numpy"]
        np.random.set_state(
            (ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4])
        )
        torch.set_rng_state(ck["rng_torch"])
        generator.set_state(ck["rng_loader"])
        if device.type == "cuda" and ck["rng_cuda"]:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
    for epoch in range(start, c["epochs"]):
        model.train()
        total = n = 0
        for step, (x, y) in enumerate(loaders[0]):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = torch.nn.functional.cross_entropy(model(x), y, ignore_index=-100)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            count = int((y != -100).sum())
            total += float(loss.detach()) * count
            n += count
            if max_steps and step + 1 >= max_steps:
                break
        val_loss = loss_eval(model, loaders[1], device)
        # Same tiling policy for validation, test and inference.
        temporary = object.__new__(Predictor)
        temporary.model = model
        temporary.device = device
        temporary.ck = {
            "training_config": c,
            "normalization": norm,
            "architecture": {"in_ch": c["bands"]},
            "input_signature": m["signature"],
        }
        report = evaluate(temporary, m, "validation")
        score = report["pooled_pixels"]["mean_iou"]
        if score is None:
            raise ValueError("Validation has no evaluable labels")
        improved = score > best
        best = max(best, score)
        scheduler.step()
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": total / n,
                "validation_loss": val_loss,
                "validation_mean_iou": score,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        ns = np.random.get_state()
        ck = {
            "format_version": 1,
            "architecture": {"name": "BENet", "in_ch": c["bands"], "out_ch": 3},
            "class_mapping": LABELS,
            "normalization": norm,
            "preprocessing_version": "band_zscore_v1",
            "input_signature": m["signature"],
            "epoch": epoch + 1,
            "validation_metrics": report,
            "split_manifest_reference": str(out / "split_manifest.json"),
            "manifest_hash": digest(m),
            "split_manifest": m,
            "training_config": c,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_mean_iou": best,
            "history": history,
            "rng_python": random.getstate(),
            "rng_numpy": (ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]),
            "rng_torch": torch.get_rng_state(),
            "rng_loader": generator.get_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            "software": {"torch": str(torch.__version__), "numpy": str(np.__version__)},
            "short_training_check": bool(max_steps),
        }
        torch.save(ck, out / "last_model.pt")
        if improved:
            torch.save(ck, out / "best_model.pt")
        if (epoch + 1) % c["checkpoint_interval"] == 0:
            torch.save(ck, out / f"epoch_{epoch+1:04d}.pt")
        write_json(out / "history.json", history)
        with (out / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        export_report(out / f"validation_epoch_{epoch+1:04d}", report)
        curves(history, out / "curves.png")
        print(history[-1], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=[
            "inspect",
            "prepare",
            "train",
            "validate",
            "compare",
            "finalize",
            "test",
            "predict",
        ],
    )
    p.add_argument("--config", default="pdi_pipeline/config.example.json")
    p.add_argument("--checkpoint")
    p.add_argument("--resume")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--cube")
    p.add_argument("--gt")
    p.add_argument("--prediction-output", default="prediction")
    p.add_argument("--registry")
    p.add_argument("--name")
    a = p.parse_args()
    c = read_json(a.config)
    if a.command == "inspect":
        print("Patients inspected:", inspect(c)["count"])
        return
    if a.command == "prepare":
        prepare(c)
        return
    if a.command == "train":
        train(c, a.resume, a.max_steps)
        return
    if a.command == "predict":
        pr = (
            Predictor.from_registry(a.registry, a.name)
            if a.registry
            else Predictor(a.checkpoint)
        )
        result = pr.predict(a.cube, a.prediction_output, c["preprocessing"], a.gt)
        print(result["metadata"])
        return
    out, m, norm = context(c)
    if a.command == "compare":
        rows = []
        for path in sorted(out.glob("epoch_*.pt")):
            pr = Predictor(path)
            if pr.ck["manifest_hash"] != digest(m):
                raise ValueError("Manifest mismatch")
            report = evaluate(pr, m, "validation")
            rows.append(
                {
                    "checkpoint": path.name,
                    "epoch": pr.ck["epoch"],
                    "validation_mean_iou": report["pooled_pixels"]["mean_iou"],
                }
            )
        write_json(out / "checkpoint_comparison.json", rows)
        with (out / "checkpoint_comparison.csv").open("w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["checkpoint", "epoch", "validation_mean_iou"]
            )
            w.writeheader()
            w.writerows(rows)
        print(rows)
        return
    path = a.checkpoint or str(out / "best_model.pt")
    pr = Predictor(path)
    if pr.ck["manifest_hash"] != digest(m) or pr.ck["normalization"] != norm:
        raise ValueError("Checkpoint provenance mismatch")
    if a.command == "finalize":
        if (
            Path(path).resolve() != (out / "best_model.pt").resolve()
            or pr.ck["short_training_check"]
        ):
            raise ValueError("Finalize full-run validation-selected best_model.pt only")
        write_json(
            out / "finalized.json",
            {
                "checkpoint_sha256": hashlib.sha256(
                    Path(path).read_bytes()
                ).hexdigest(),
                "config_hash": digest(c),
                "manifest_hash": digest(m),
                "notice": "Configuration finalized by operator before held-out testing",
            },
        )
        print("Finalization recorded. Do not tune using test results.")
        return
    split = "test" if a.command == "test" else "validation"
    if split == "test":
        lock = read_json(out / "finalized.json")
        if lock["checkpoint_sha256"] != hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest() or lock["config_hash"] != digest(c):
            raise ValueError("Finalized checkpoint/config changed")
        if (out / "test.json").exists():
            raise ValueError(
                "Final test already recorded; preserve it and do not tune against it"
            )
    report = evaluate(pr, m, split)
    export_report(out / split, report)
    print(report["pooled_pixels"])


if __name__ == "__main__":
    main()
