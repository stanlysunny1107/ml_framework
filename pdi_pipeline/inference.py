"""Public PDI API: Predictor.from_registry(...).predict(cube, output, preprocessing, gt=None)."""

from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from .model import BENet
from .data import LABELS, load_cube, signature, mask, read_json, write_json
from .report import confusion, metrics, overlay


class Predictor:
    def __init__(self, checkpoint, device="auto"):
        self.path = Path(checkpoint)
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        # Only load locally trusted checkpoints. weights_only blocks arbitrary pickle globals.
        self.ck = torch.load(self.path, map_location="cpu", weights_only=True)
        if (
            self.ck["architecture"]
            != {
                "name": "BENet",
                "in_ch": len(self.ck["normalization"]["mean"]),
                "out_ch": 3,
            }
            or self.ck["class_mapping"] != LABELS
        ):
            raise ValueError("Incompatible architecture or class mapping")
        if self.ck["preprocessing_version"] != "band_zscore_v1":
            raise ValueError("Unsupported preprocessing")
        norm = self.ck["normalization"]
        if (
            len(norm["std"]) != len(norm["mean"])
            or not np.isfinite(norm["mean"]).all()
            or not np.isfinite(norm["std"]).all()
            or np.any(np.asarray(norm["std"]) <= 0)
        ):
            raise ValueError("Invalid checkpoint normalization")
        self.model = BENet(self.ck["architecture"]["in_ch"], 3).to(self.device)
        self.model.load_state_dict(self.ck["model"], strict=True)
        self.model.eval()

    @classmethod
    def from_registry(cls, registry, name, device="auto"):
        r = read_json(registry)
        entry = next(e for e in r["checkpoints"] if e["name"] == name)
        p = cls(Path(registry).resolve().parent / entry["path"], device)
        if (
            entry["architecture"] != p.ck["architecture"]
            or entry["class_mapping"] != p.ck["class_mapping"]
        ):
            raise ValueError("Registry metadata mismatch")
        p.registry_name = name
        return p

    def predict_array(self, cube):
        p = self.ck["training_config"]["patch_size"]
        b, h, w = cube.shape
        if b != self.ck["architecture"]["in_ch"]:
            raise ValueError("Band count mismatch")
        norm = self.ck["normalization"]
        mean = np.asarray(norm["mean"], np.float32)[:, None, None]
        std = np.asarray(norm["std"], np.float32)[:, None, None]
        result = np.empty((h, w), np.int64)
        with torch.inference_mode():
            for y in range(0, h, p):
                for x in range(0, w, p):
                    a = np.asarray(cube[:, y : y + p, x : x + p], np.float32)
                    if not np.isfinite(a).all():
                        raise ValueError("Nonfinite input")
                    hh, ww = a.shape[1:]
                    tile = np.zeros((b, p, p), np.float32)
                    tile[:, :hh, :ww] = (a - mean) / std
                    out = (
                        self.model(torch.from_numpy(tile[None]).to(self.device))
                        .argmax(1)[0, :hh, :ww]
                        .cpu()
                        .numpy()
                    )
                    result[y : y + hh, x : x + ww] = np.asarray(LABELS)[out]
        return result

    def predict(self, cube, output, preprocessing, gt=None):
        if signature(cube, preprocessing) != self.ck["input_signature"]:
            raise ValueError(
                "Spectral order, wavelength units, or preprocessing mismatch"
            )
        c = load_cube(cube)
        prediction = self.predict_array(c)
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "prediction.npy", prediction)
        savemat(out / "prediction.mat", {"prediction": prediction})
        overlay(c, prediction, out / "prediction.png")
        meta = {
            "checkpoint_name": getattr(self, "registry_name", self.path.name),
            "checkpoint_path": str(self.path.resolve()),
            "epoch": self.ck["epoch"],
            "class_mapping": LABELS,
            "architecture": self.ck["architecture"],
            "input_signature": self.ck["input_signature"],
            "split_manifest_hash": self.ck["manifest_hash"],
            "shape": list(prediction.shape),
            "notice": "Research predictions, not validated clinical diagnoses",
        }
        if gt is not None:
            meta["ground_truth_comparison"] = metrics(
                confusion(mask(gt, prediction.shape), prediction)
            )
        write_json(out / "prediction_metadata.json", meta)
        return {
            "mask": prediction,
            "metadata": meta,
            "prediction_path": str((out / "prediction.npy").resolve()),
        }
