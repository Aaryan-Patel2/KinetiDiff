#!/usr/bin/env python3
"""Idempotent data download script for KinetiDiff.

Downloads:
1. PDB 3MTF (ACVR1/ALK2 crystal structure) from RCSB — public domain
2. Converts 3MTF to docking-ready PDBQT via obabel
3. GCDM checkpoint from Zenodo record 13375913
4. Writes data/checksums.txt

PDBBind v2020 requires manual registration — see printed instructions.
"""

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_STRUCTURES = REPO_ROOT / "data" / "structures"
DATA_CHECKPOINTS = REPO_ROOT / "data" / "checkpoints"

PDB_ID = "3MTF"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
PDB_PATH = DATA_STRUCTURES / f"{PDB_ID.lower()}.pdb"
PDBQT_PATH = DATA_STRUCTURES / "receptor.pdbqt"

ZENODO_RECORD = "13375913"
CHECKPOINT_NAME = "bindingmoad_ca_cond_gcpnet.ckpt"
ZENODO_URL = f"https://zenodo.org/record/{ZENODO_RECORD}/files/{CHECKPOINT_NAME}"
CHECKPOINT_PATH = DATA_CHECKPOINTS / CHECKPOINT_NAME

CHECKSUMS_PATH = REPO_ROOT / "data" / "checksums.txt"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, label: str, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"  [skip] {label} already exists at {dest.relative_to(REPO_ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {label} ...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()
    print(f"  Saved to {dest.relative_to(REPO_ROOT)}")


def _progress(count: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        sys.stdout.write(f"\r    {pct}%")
        sys.stdout.flush()


def convert_to_pdbqt(pdb: Path, pdbqt: Path, force: bool = False) -> None:
    if pdbqt.exists() and not force:
        print(f"  [skip] PDBQT already exists at {pdbqt.relative_to(REPO_ROOT)}")
        return
    print("  Converting PDB → PDBQT via obabel ...")
    result = subprocess.run(
        ["obabel", str(pdb), "-O", str(pdbqt), "-xr"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: obabel exited with code {result.returncode}")
        print(f"  stderr: {result.stderr.strip()}")
        print("  Install obabel: conda install -c conda-forge openbabel")
    else:
        print(f"  Saved to {pdbqt.relative_to(REPO_ROOT)}")


def write_checksums(paths: list[Path]) -> None:
    lines = []
    for path in paths:
        if path.exists():
            lines.append(f"{sha256(path)}  {path.relative_to(REPO_ROOT)}")
    CHECKSUMS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECKSUMS_PATH.write_text("\n".join(lines) + "\n")
    print(f"  Checksums written to {CHECKSUMS_PATH.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download KinetiDiff data assets (3MTF, checkpoint)")
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    force = args.force

    print("=== KinetiDiff data download ===\n")

    print("1/3  PDB 3MTF (ACVR1 crystal structure)")
    download(PDB_URL, PDB_PATH, "3MTF PDB", force=force)

    print("\n2/3  Receptor PDBQT (obabel conversion)")
    if PDB_PATH.exists():
        convert_to_pdbqt(PDB_PATH, PDBQT_PATH, force=force)
    else:
        print("  [skip] PDB not downloaded; skipping PDBQT conversion")

    print("\n3/3  GCDM checkpoint (Zenodo record 13375913)")
    download(ZENODO_URL, CHECKPOINT_PATH, "bindingmoad_ca_cond_gcpnet.ckpt", force=force)

    print("\nWriting checksums ...")
    write_checksums([PDB_PATH, PDBQT_PATH, CHECKPOINT_PATH])

    print(
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "PDBBind v2020 (required for affinity predictor training)\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "PDBBind requires free manual registration at:\n"
        "  http://www.pdbbind.org.cn/\n"
        "Download the 'general set' and extract under:\n"
        "  data/pdbbind/\n"
        "Then run: python -m kinetidiff.train.train_affinity --help\n"
    )

    print("Done.")


if __name__ == "__main__":
    main()
