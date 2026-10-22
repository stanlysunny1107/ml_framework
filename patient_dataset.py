"""Discover and import patient datasets without replacing another patient's files."""

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile


def load_cube(path):
    """Read an ENVI cube as a memory-mapped array in band, row, column order.

    Read dimensions, data type, byte order, and interleave from the adjacent .hdr file.
    """
    import numpy as np

    path = Path(path)
    text = path.with_suffix(".hdr").read_text()
    meta = dict(
        (
            (k.strip().lower(), v.strip())
            for k, v in re.findall("^([^=\\n]+)=\\s*([^\\n]+)", text, re.M)
        )
    )
    h, w, b = (int(meta[k]) for k in ("lines", "samples", "bands"))
    types = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4"}
    dtype = np.dtype(
        ("<" if int(meta.get("byte order", 0)) == 0 else ">")
        + types[int(meta["data type"])]
    )
    offset = int(meta.get("header offset", 0))
    if path.stat().st_size != offset + h * w * b * dtype.itemsize:
        raise ValueError("Cube file size disagrees with ENVI header.")
    raw = np.memmap(path, dtype=dtype, mode="r", offset=offset)
    layout = meta["interleave"].lower()
    if layout == "bsq":
        return raw.reshape(b, h, w)
    if layout == "bil":
        return raw.reshape(h, b, w).transpose(1, 0, 2)
    if layout == "bip":
        return raw.reshape(h, w, b).transpose(2, 0, 1)
    raise ValueError(f"Unsupported interleave: {layout}")


def patient_record(folder):
    """Validate required files and return absolute paths for one patient."""
    folder = Path(folder).resolve()
    cube = folder / "cube.dat"
    header = folder / "cube.hdr"
    annotations = list(folder.glob("*_GT.mat"))
    if not cube.is_file() or not header.is_file() or len(annotations) != 1:
        raise ValueError(f"{folder}: expected cube.dat, cube.hdr, and one *_GT.mat")
    patient_id = annotations[0].name[:-7]
    if folder.name != patient_id:
        raise ValueError(
            f"Patient folder {folder.name} disagrees with annotation ID {patient_id}"
        )
    record = {
        "id": patient_id,
        "cube": str(cube),
        "header": str(header),
        "gt": str(annotations[0]),
        "gt_key": "annotation_map",
    }
    for key, suffix in [
        ("rgb", "_RGB.png"),
        ("gt_png", "_GT.png"),
        ("diagnostic", "_Diagnostic.txt"),
    ]:
        path = folder / (patient_id + suffix)
        record[key] = str(path) if path.is_file() else None
    return record


def discover_patients(root):
    """Locate patient folders recursively and reject duplicate patient IDs."""
    records = []
    seen = set()
    for cube in sorted(Path(root).resolve().rglob("cube.dat")):
        record = patient_record(cube.parent)
        if record["id"] in seen:
            raise ValueError(f'Duplicate patient ID: {record["id"]}')
        seen.add(record["id"])
        records.append(record)
    return records


def import_patient_zip(archive, destination):
    """Extract a single patient archive into its own directory; never overwrite data."""
    archive, destination = Path(archive), Path(destination).resolve()
    if not re.fullmatch(r"Patient_\d+", archive.stem):
        raise ValueError(f"Expected archive named Patient_<number>.zip: {archive.name}")
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / archive.stem
    with zipfile.ZipFile(archive) as bundle:
        members = []
        names = set()
        for info in bundle.infolist():
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name).parts
            if (
                not parts
                or parts[0] != archive.stem
                or any(p in (".", "..") or ":" in p for p in parts)
            ):
                raise ValueError(f"Unexpected archive path: {name}")
            resolved = destination.joinpath(*parts).resolve()
            if not resolved.is_relative_to(target.resolve()):
                raise ValueError(f"Archive path escapes patient folder: {name}")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError(
                    "Symbolic links are not supported in patient archives."
                )
            if name.casefold() in names:
                raise ValueError(f"Duplicate archive member: {name}")
            names.add(name.casefold())
            members.append((info, resolved))
        required = {
            f"{archive.stem}/cube.dat",
            f"{archive.stem}/cube.hdr",
            f"{archive.stem}/{archive.stem}_GT.mat",
        }
        if not required.issubset({i.filename.replace("\\", "/") for i, _ in members}):
            raise ValueError(f"{archive.name}: required dataset files are missing")
        if target.exists():
            for info, path in members:
                if not info.is_dir() and (
                    not path.is_file() or path.stat().st_size != info.file_size
                ):
                    raise ValueError(
                        f"Existing patient differs or is incomplete: {target}; use another destination"
                    )
            return patient_record(target)
        # Exclusive file creation also prevents accidental replacement during import.
        for info, path in members:
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, path.open("xb") as output:
                    shutil.copyfileobj(source, output)
    return patient_record(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--zip-dir", help="Optional folder containing Patient_*.zip archives"
    )
    parser.add_argument("--manifest", default="patients.json")
    args = parser.parse_args()
    if args.zip_dir:
        archives = sorted(Path(args.zip_dir).glob("Patient_*.zip"))
        if not archives:
            parser.error("No Patient_*.zip files found in --zip-dir")
        for archive in archives:
            record = import_patient_zip(archive, args.data_root)
            print(f'Imported or verified {record["id"]}', flush=True)
    patients = discover_patients(args.data_root)
    if not patients:
        parser.error("No patient datasets found")
    Path(args.manifest).write_text(
        json.dumps({"version": 1, "patients": patients}, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(patients)} patient records to {Path(args.manifest).resolve()}")


if __name__ == "__main__":
    main()
