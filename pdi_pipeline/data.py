import hashlib
import json
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .original_be_net import load_cube

LABELS = [103, 104, 105]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def signature(path, preprocessing):
    text = Path(path).with_suffix(".hdr").read_text()
    m = re.search(r"wavelength\s*=\s*\{([^}]+)\}", text, re.I)
    if not m:
        raise ValueError(
            "Ordered wavelengths required in header; do not infer spectral ordering from band count."
        )
    waves = [float(x) for x in re.split(r"[,\s]+", m[1].strip()) if x]
    units = re.search(r"wavelength units\s*=\s*([^\r\n]+)", text, re.I)
    if not units:
        raise ValueError("Header requires wavelength units")
    return {
        "wavelengths": waves,
        "units": units[1].strip().lower(),
        "preprocessing": preprocessing,
    }


def mask(path, shape):
    a = np.asarray(loadmat(path)["annotation_map"])
    if (
        a.shape != tuple(shape)
        or not np.isfinite(a).all()
        or not np.equal(a, np.floor(a)).all()
    ):
        raise ValueError(
            f"{path}: annotation_map must be aligned integer H x W; no automatic transpose"
        )
    if set(np.unique(a)) - {0, *LABELS}:
        raise ValueError(f"{path}: unexpected labels {np.unique(a)}")
    return a.astype(np.int64)


def encode(a):
    y = np.full(a.shape, -100, dtype=np.int64)
    for i, label in enumerate(LABELS):
        y[a == label] = i
    return y


def discover(root):
    records = []
    for dat in sorted(Path(root).resolve().rglob("*.dat")):
        mats = list(dat.parent.glob("*.mat"))
        if len(mats) != 1:
            raise ValueError(
                f"{dat.parent}: expected one .mat; use a clean input-only folder"
            )
        records.append({"id": dat.parent.name, "cube": str(dat), "gt": str(mats[0])})
    if not records:
        raise ValueError("No .dat cubes found")
    if len({r["id"] for r in records}) != len(records):
        raise ValueError(
            "Duplicate patient ID (or multiple cubes per patient); consolidate explicitly"
        )
    return records


def inspect(config):
    records, reference = discover(config["dataset_root"]), None
    errors = []
    for r in records:
        try:
            cube = load_cube(r["cube"])
            sig = signature(r["cube"], config["preprocessing"])
            if (
                len(sig["wavelengths"]) != cube.shape[0]
                or cube.shape[0] != config["bands"]
            ):
                raise ValueError(f"{r['id']}: incompatible band count")
            if reference is not None and sig != reference:
                raise ValueError(f"{r['id']}: spectral order/units differ")
            reference = sig
            a = mask(r["gt"], cube.shape[1:])
            r.update(
                shape=list(cube.shape),
                labels={
                    str(int(v)): int(n)
                    for v, n in zip(*np.unique(a, return_counts=True))
                },
            )
            # Scan all voxels; reject rather than silently changing invalid input.
            for y in range(0, cube.shape[1], 32):
                if not np.isfinite(cube[:, y : y + 32]).all():
                    raise ValueError(f"{r['id']}: nonfinite cube values")
            r["cube_sha256"] = file_hash(r["cube"])
            r["mask_sha256"] = file_hash(r["gt"])
            r["header_sha256"] = file_hash(Path(r["cube"]).with_suffix(".hdr"))
            r["alignment"] = "shape checked; overlay requires human spatial review"
            from .report import overlay

            overlay(cube, a, Path(config["output"]) / "inspection" / f"{r['id']}.png")
        except Exception as e:
            r["error"] = str(e)
            errors.append({"id": r["id"], "error": str(e)})
    successful = [r for r in records if "cube_sha256" in r]
    if len({r["cube_sha256"] for r in successful}) != len(successful):
        raise ValueError("Identical cubes under different patient IDs")
    report = {
        "patients": records,
        "signature": reference,
        "count": len(records),
        "errors": errors,
    }
    write_json(Path(config["output"]) / "inspection.json", report)
    if errors:
        raise ValueError(
            f"Inspection found {len(errors)} invalid patients; see inspection.json"
        )
    return report


def validate_manifest(m):
    ids = [x for s in ("train", "validation", "test") for x in m["splits"][s]]
    if len(ids) != len(set(ids)) or set(ids) != {r["id"] for r in m["patients"]}:
        raise ValueError("Patient leakage, duplicate, or incomplete split")
    if [len(m["splits"][s]) for s in ("train", "validation", "test")] != [53, 12, 12]:
        raise ValueError("Expected 53/12/12 patient split")
    if len({str(Path(r["cube"]).resolve()) for r in m["patients"]}) != len(ids):
        raise ValueError("Aliased cube paths across patients")


def prepare(config):
    report = inspect(config)
    if report["count"] != 77:
        raise ValueError(
            f"Expected 77 patients; inspected {report['count']}. Inspection report saved; no split created."
        )
    ids = sorted(r["id"] for r in report["patients"])
    np.random.default_rng(config["seed"]).shuffle(ids)
    m = {
        **report,
        "seed": config["seed"],
        "splits": {"train": ids[:53], "validation": ids[53:65], "test": ids[65:]},
    }
    validate_manifest(m)
    m["missing_classes"] = {}
    for s, members in m["splits"].items():
        present = {
            int(k)
            for r in m["patients"]
            if r["id"] in members
            for k in r["labels"]
            if r["labels"][k]
        }
        m["missing_classes"][s] = sorted(set(LABELS) - present)
    out = Path(config["output"])
    write_json(out / "split_manifest.json", m)
    print("Missing classes:", m["missing_classes"])
    if m["missing_classes"]["train"]:
        raise ValueError(
            "Training missing classes; do not train. Review cohort/split design without test outcome tuning."
        )
    # Pixel-weighted population statistics, training cubes only, float64 accumulation.
    sums = np.zeros(config["bands"])
    squares = sums.copy()
    n = 0
    for r in m["patients"]:
        if r["id"] not in m["splits"]["train"]:
            continue
        c = load_cube(r["cube"])
        for y in range(0, c.shape[1], 32):
            a = np.asarray(c[:, y : y + 32], np.float64).reshape(config["bands"], -1)
            sums += a.sum(1)
            squares += (a * a).sum(1)
            n += a.shape[1]
    mean = sums / n
    std = np.maximum(np.sqrt(np.maximum(squares / n - mean**2, 0)), 1e-6)
    write_json(
        out / "normalization.json",
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "training_ids": m["splits"]["train"],
            "manifest_hash": digest(m),
        },
    )


class Tiles:
    def __init__(self, records, size, norm, augment=False):
        self.records, self.size, self.norm, self.augment = records, size, norm, augment
        self.coords = []
        self.cubes = []
        self.masks = []
        for i, r in enumerate(records):
            c = load_cube(r["cube"])
            a = mask(r["gt"], c.shape[1:])
            self.cubes.append(c)
            self.masks.append(a)
            for y in range(0, c.shape[1], size):
                for x in range(0, c.shape[2], size):
                    if np.any(a[y : y + size, x : x + size] != 0):
                        self.coords.append((i, y, x))
        if not self.coords:
            raise ValueError("No annotated patches")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        import torch

        i, y, x = self.coords[index]
        r = self.records[i]
        p = self.size
        c = self.cubes[i]
        a = self.masks[i]
        crop = np.asarray(c[:, y : y + p, x : x + p], np.float32)
        h, w = crop.shape[1:]
        image = np.zeros((c.shape[0], p, p), np.float32)
        target = np.full((p, p), -100, np.int64)
        image[:, :h, :w] = (
            crop - np.asarray(self.norm["mean"], np.float32)[:, None, None]
        ) / np.asarray(self.norm["std"], np.float32)[:, None, None]
        target[:h, :w] = encode(a[y : y + p, x : x + p])
        if self.augment:
            k = np.random.randint(4)
            image = np.rot90(image, k, axes=(1, 2))
            target = np.rot90(target, k)
            if np.random.rand() < 0.5:
                image = image[:, :, ::-1]
                target = target[:, ::-1]
        return torch.from_numpy(image.copy()), torch.from_numpy(target.copy())
