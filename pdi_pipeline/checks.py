"""Synthetic checks only; these do not measure patient performance."""

import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat
from .data import LABELS, encode, mask, validate_manifest, write_json
from .report import metrics, confusion
from .inference import Predictor
from .model import BENet


class Checks(unittest.TestCase):
    def test_architecture_preserved(self):
        import ast

        root = Path(__file__).parent

        def classes(path):
            tree = ast.parse(path.read_text())
            return {
                n.name: ast.dump(n) for n in tree.body if isinstance(n, ast.ClassDef) and n.name != "CubeTiles"
            }

        self.assertEqual(
            classes(root / "model.py"), classes(root / "original_be_net.py")
        )

    def test_ignored_and_absent(self):
        cm = confusion(np.array([[0, 103, 104]]), np.array([[105, 103, 103]]))
        self.assertEqual(int(cm.sum()), 2)
        m = metrics(cm)
        self.assertIsNone(m["per_class"]["105"]["iou"])
        self.assertEqual(m["per_class"]["104"]["iou"], 0)
        self.assertEqual(encode(np.array([0, 103, 104, 105])).tolist(), [-100, 0, 1, 2])
        self.assertIsNone(metrics(np.zeros((3, 3), int))["pixel_accuracy"])

    def test_leakage(self):
        m = {
            "patients": [{"id": str(i), "cube": str(i) + ".dat"} for i in range(77)],
            "splits": {
                "train": list(map(str, range(53))),
                "validation": list(map(str, range(53, 65))),
                "test": list(map(str, range(65, 77))),
            },
        }
        validate_manifest(m)
        m["splits"]["test"][0] = "0"
        with self.assertRaises(ValueError):
            validate_manifest(m)

    def test_checkpoint_and_native_shape(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            c = np.zeros((3, 35, 49), np.float32)
            c.tofile(d / "cube.dat")
            (d / "cube.hdr").write_text(
                "ENVI\nbands = 3\nlines = 35\nsamples = 49\ndata type = 4\ninterleave = bsq\nwavelength units = Nanometers\nwavelength = {1,2,3}\n"
            )
            sig = {
                "wavelengths": [1.0, 2.0, 3.0],
                "units": "nanometers",
                "preprocessing": "raw",
            }
            ck = {
                "architecture": {"name": "BENet", "in_ch": 3, "out_ch": 3},
                "class_mapping": LABELS,
                "normalization": {"mean": [0] * 3, "std": [1] * 3},
                "preprocessing_version": "band_zscore_v1",
                "training_config": {"patch_size": 32},
                "model": BENet(3, 3).state_dict(),
                "input_signature": sig,
                "epoch": 1,
                "manifest_hash": "synthetic",
            }
            torch.save(ck, d / "synthetic.pt")
            pr = Predictor(d / "synthetic.pt", "cpu")
            r = pr.predict(d / "cube.dat", d / "out", "raw")
            self.assertEqual(r["mask"].shape, (35, 49))
            self.assertLessEqual(set(np.unique(r["mask"])), set(LABELS))
            self.assertNotIn("ground_truth_comparison", r["metadata"])
            self.assertTrue((d / "out" / "prediction.mat").exists())
            with self.assertRaises(ValueError):
                pr.predict(d / "cube.dat", d / "bad", "different")
            bad = copy.deepcopy(ck)
            bad["class_mapping"] = [1, 2, 3]
            torch.save(bad, d / "bad.pt")
            with self.assertRaises(ValueError):
                Predictor(d / "bad.pt")
            savemat(d / "bad.mat", {"annotation_map": np.zeros((49, 35))})
            with self.assertRaises(ValueError):
                mask(d / "bad.mat", (35, 49))
            savemat(d / "bad.mat", {"annotation_map": np.full((35, 49), 100)})
            with self.assertRaises(ValueError):
                mask(d / "bad.mat", (35, 49))

    def test_training_resume_equivalence(self):
        from unittest.mock import patch
        from .data import prepare, read_json
        from .cli import train
        import shutil

        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "data"
            rng = np.random.default_rng(8)
            for i in range(77):
                folder = data / f"Patient_{i+1}"
                folder.mkdir(parents=True)
                rng.normal(size=(3, 32, 32)).astype(np.float32).tofile(
                    folder / "cube.dat"
                )
                (folder / "cube.hdr").write_text(
                    "ENVI\nbands = 3\nlines = 32\nsamples = 32\ndata type = 4\ninterleave = bsq\nwavelength units = Nanometers\nwavelength = {1,2,3}\n"
                )
                savemat(
                    folder / "mask.mat",
                    {
                        "annotation_map": np.resize(
                            np.array([0, 103, 104, 105]), (32, 32)
                        )
                    },
                )
            c = {
                "dataset_root": str(data),
                "output": str(root / "resumed"),
                "bands": 3,
                "preprocessing": "synthetic",
                "seed": 17,
                "epochs": 1,
                "learning_rate": 0.0001,
                "batch_size": 2,
                "patch_size": 32,
                "mixed_precision": False,
                "checkpoint_interval": 1,
                "scheduler_step": 2,
            }
            with patch("pdi_pipeline.report.overlay"):
                prepare(c)
            train(c, max_steps=1)
            c["epochs"] = 2
            train(c, resume=str(root / "resumed" / "last_model.pt"), max_steps=1)
            resumed = torch.load(root / "resumed" / "last_model.pt", weights_only=True)
            uninterrupted = root / "uninterrupted"
            uninterrupted.mkdir()
            for filename in ("split_manifest.json", "normalization.json"):
                shutil.copy(root / "resumed" / filename, uninterrupted / filename)
            train(dict(c, output=str(uninterrupted)), max_steps=1)
            full = torch.load(uninterrupted / "last_model.pt", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["history"], full["history"])
            for key in full["model"]:
                self.assertTrue(
                    torch.equal(full["model"][key], resumed["model"][key]), key
                )
            self.assertIn("optimizer", resumed)
            self.assertIn("scheduler", resumed)
            self.assertIn("scaler", resumed)
            Predictor(root / "resumed" / "last_model.pt", "cpu")


if __name__ == "__main__":
    unittest.main()
