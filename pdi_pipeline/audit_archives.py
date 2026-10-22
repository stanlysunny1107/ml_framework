"""Audit ZIP metadata and annotations locally without extracting patient cubes."""

import argparse
import io
import re
import zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .data import write_json, LABELS


def audit(root, output):
    rows = []
    errors = []
    for path in sorted(Path(root).glob("Patient_*.zip")):
        try:
            with zipfile.ZipFile(path) as z:
                pid = path.stem
                header = z.read(f"{pid}/cube.hdr").decode()

                def field(k):
                    return re.search(
                        r"^\s*" + k + r"\s*=\s*([^\r\n]+)", header, re.M | re.I
                    )[1].strip()

                b, h, w = [int(field(k)) for k in ("bands", "lines", "samples")]
                dtype = {1: 1, 2: 2, 3: 4, 4: 4, 5: 8, 12: 2, 13: 4}[
                    int(field("data type"))
                ]
                if z.getinfo(f"{pid}/cube.dat").file_size != b * h * w * dtype + int(
                    field("header offset")
                ):
                    raise ValueError("Cube size/header mismatch")
                a = loadmat(io.BytesIO(z.read(f"{pid}/{pid}_GT.mat")))["annotation_map"]
                if (
                    a.shape != (h, w)
                    or not np.isfinite(a).all()
                    or not np.equal(a, np.floor(a)).all()
                ):
                    raise ValueError("Invalid mask dimensions or labels")
                labels = {
                    str(int(v)): int(n)
                    for v, n in zip(*np.unique(a, return_counts=True))
                }
                unexpected = sorted(set(map(int, labels)) - {0, *LABELS})
                rows.append(
                    {
                        "id": pid,
                        "shape": [b, h, w],
                        "labels": labels,
                        "unexpected_labels": unexpected,
                        "alignment": "Shape only; visual spatial review required",
                        "cube_values_scanned": False,
                    }
                )
        except Exception as e:
            errors.append({"archive": path.name, "error": str(e)})
    unexpected = {
        r["id"]: r["unexpected_labels"] for r in rows if r["unexpected_labels"]
    }
    missing = [
        f"Patient_{i}"
        for i in range(1, 78)
        if f"Patient_{i}" not in {r["id"] for r in rows}
    ]
    write_json(
        output,
        {
            "count": len(rows),
            "patients": rows,
            "errors": errors,
            "missing_expected_ids": missing,
            "unexpected_labels_by_patient": unexpected,
        },
    )
    print(
        "Audited:",
        len(rows),
        "Errors:",
        errors,
        "Missing:",
        missing,
        "Patients with unexpected labels:",
        len(unexpected),
    )
    print("All observed IDs:", sorted({int(k) for r in rows for k in r["labels"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    audit(a.root, a.output)
