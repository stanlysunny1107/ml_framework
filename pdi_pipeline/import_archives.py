"""Safe local import of patient ZIPs. Extract only required data, never execute contents."""

import argparse
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from .data import write_json


def import_archives(source, destination, patients=None):
    source = Path(source)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    imported = []
    for archive in sorted(source.glob("Patient_*.zip")):
        pid = archive.stem
        if patients and pid not in patients:
            continue
        with zipfile.ZipFile(archive) as z:
            names = z.namelist()
            expected = [f"{pid}/cube.dat", f"{pid}/cube.hdr", f"{pid}/{pid}_GT.mat"]
            if any(names.count(n) != 1 for n in expected):
                raise ValueError(f"{archive}: required files absent or duplicated")
            for name in expected:
                parts = PurePosixPath(name).parts
                if ".." in parts or PurePosixPath(name).is_absolute():
                    raise ValueError("Unsafe ZIP member")
                target = (destination / Path(*parts)).resolve()
                if destination not in target.parents:
                    raise ValueError("ZIP path escapes destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    if target.stat().st_size != z.getinfo(name).file_size:
                        raise ValueError(
                            f"Existing partial or conflicting file: {target}"
                        )
                    with target.open("rb") as existing:
                        import binascii

                        crc = 0
                        while chunk := existing.read(1024 * 1024):
                            crc = binascii.crc32(chunk, crc)
                    if (crc & 0xFFFFFFFF) != z.getinfo(name).CRC:
                        raise ValueError(f"Existing content differs: {target}")
                    continue
                partial = target.with_suffix(target.suffix + ".partial")
                with z.open(name) as src, partial.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                partial.replace(target)
        imported.append(pid)
        print("Imported", pid, flush=True)
    missing = [f"Patient_{i}" for i in range(1, 78) if f"Patient_{i}" not in imported]
    write_json(
        destination / "archive_inventory.json",
        {
            "patient_ids": imported,
            "count": len(imported),
            "missing_expected_ids": missing,
        },
    )
    print("Missing expected IDs:", missing)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--destination", required=True)
    p.add_argument("--patients", nargs="+")
    a = p.parse_args()
    import_archives(a.source, a.destination, a.patients)
