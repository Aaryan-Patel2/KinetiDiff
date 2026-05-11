#!/usr/bin/env python3
"""Receptor PDB preparation for AutoDock Vina docking.

Cleans a raw PDB file (removes water, HETATM) and optionally converts it to
PDBQT format using obabel. The binding site centroid is printed at the end
for direct use in Vina config.

Usage:
    python scripts/prepare_receptor.py \
        --input data/structures/acvr1_wt.pdb \
        --output data/structures/receptor_siteA.pdb \
        --site A

    # Also generate PDBQT:
    python scripts/prepare_receptor.py \
        --input data/structures/acvr1_wt.pdb \
        --output data/structures/receptor_siteA.pdb \
        --site A --pdbqt

    # Force overwrite:
    python scripts/prepare_receptor.py ... --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Known binding sites for ACVR1/ALK2 (PDB 3MTF).
# Centroid values from CLAUDE.md and paper/SI.
SITE_CONFIG: dict[str, dict] = {
    "A": {
        "description": "ACVR1/ALK2 ATP-binding pocket (PDB 3MTF, chain A)",
        "centroid": (24.87, -12.54, 38.40),
        "chains": ["A"],
        "box_size": (20.0, 20.0, 20.0),
    },
}

# Water residue names to strip from PDB
_WATER_NAMES = {"HOH", "WAT", "H2O", "DOD", "TIP", "TIP3"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare receptor PDB for AutoDock Vina",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path, help="Raw input PDB file")
    p.add_argument("--output", required=True, type=Path, help="Cleaned output PDB file")
    p.add_argument(
        "--site",
        default="A",
        choices=sorted(SITE_CONFIG),
        help="Binding site (default: A — ACVR1 ATP pocket)",
    )
    p.add_argument(
        "--chain",
        default=None,
        metavar="CHAIN_ID",
        help="Override chain filter (e.g. B). Default: use site config.",
    )
    p.add_argument(
        "--keep-hetatm",
        action="store_true",
        help="Keep non-water HETATM records (co-factors, ions). Default: remove all.",
    )
    p.add_argument(
        "--pdbqt",
        action="store_true",
        help="Also convert output PDB to PDBQT via obabel (requires openbabel)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files",
    )
    return p.parse_args()


def _resolve(path: Path) -> Path:
    """Make path absolute relative to repo root if not already absolute."""
    return path if path.is_absolute() else REPO_ROOT / path


def clean_pdb(
    input_path: Path,
    output_path: Path,
    chains: list[str],
    keep_hetatm: bool,
    force: bool,
) -> bool:
    """Strip water/HETATM, filter chains, write cleaned PDB.

    Returns True on success (including skip-because-exists).
    """
    if output_path.exists() and not force:
        rel = output_path.relative_to(REPO_ROOT)
        print(f"  [skip] {rel} already exists  (use --force to overwrite)")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []
    n_kept = n_water = n_hetatm = n_chain = 0

    with open(input_path) as fh:
        for line in fh:
            record = line[:6].strip()

            if record in {"ATOM", "HETATM"}:
                res_name = line[17:20].strip()

                # Drop water unconditionally
                if res_name in _WATER_NAMES:
                    n_water += 1
                    continue

                # Drop non-water HETATM unless caller asked to keep them
                if record == "HETATM" and not keep_hetatm:
                    n_hetatm += 1
                    continue

                # Filter by chain
                chain_id = line[21]
                if chains and chain_id not in chains:
                    n_chain += 1
                    continue

                n_kept += 1
                out_lines.append(line)

            elif record in {
                "TER", "END", "REMARK", "HEADER", "TITLE",
                "CRYST1", "SEQRES",
                "SCALE1", "SCALE2", "SCALE3",
                "ORIGX1", "ORIGX2", "ORIGX3",
            }:
                out_lines.append(line)
            # CONECT, MASTER, ANISOU, etc. are dropped intentionally

    if n_kept == 0:
        print("  ERROR: No ATOM records kept — check input file and chain filter")
        return False

    # Guarantee END record
    if not out_lines or out_lines[-1].strip() != "END":
        out_lines.append("END\n")

    output_path.write_text("".join(out_lines))

    rel = output_path.relative_to(REPO_ROOT)
    print(f"  Wrote {n_kept} coordinate records to {rel}")
    if n_water:
        print(f"  Removed: {n_water} water, {n_hetatm} HETATM ligand, {n_chain} wrong-chain atoms")
    return True


def convert_to_pdbqt(pdb_path: Path, force: bool) -> Path | None:
    """Convert cleaned PDB → PDBQT via obabel for Vina input.

    Uses -xr (rigid receptor output, no flexible residues). Requires obenbabel
    installed: conda install -c conda-forge openbabel
    """
    pdbqt_path = pdb_path.with_suffix(".pdbqt")

    if pdbqt_path.exists() and not force:
        rel = pdbqt_path.relative_to(REPO_ROOT)
        print(f"  [skip] {rel} already exists")
        return pdbqt_path

    print(f"  Converting {pdb_path.name} → {pdbqt_path.name} via obabel ...")
    result = subprocess.run(
        ["obabel", str(pdb_path), "-O", str(pdbqt_path), "-xr"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  WARNING: obabel exited with code {result.returncode}")
        if result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()}")
        print("  Install obabel: conda install -c conda-forge openbabel")
        return None

    rel = pdbqt_path.relative_to(REPO_ROOT)
    print(f"  PDBQT written to {rel}")
    return pdbqt_path


def main() -> None:
    args = parse_args()

    input_path = _resolve(args.input)
    output_path = _resolve(args.output)

    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}")
        sys.exit(1)

    site = SITE_CONFIG[args.site]
    chains = [args.chain] if args.chain else site["chains"]
    n_steps = 2 if args.pdbqt else 1

    print("=== Receptor preparation ===")
    print(f"  Site: {args.site} — {site['description']}")
    print(f"  Chain filter: {chains}")
    print(f"  Pocket centroid: {site['centroid']}")
    print()

    print(f"1/{n_steps}  Clean PDB")
    ok = clean_pdb(input_path, output_path, chains, args.keep_hetatm, args.force)
    if not ok:
        sys.exit(1)

    if args.pdbqt:
        print(f"\n2/{n_steps}  Convert to PDBQT")
        convert_to_pdbqt(output_path, args.force)

    cx, cy, cz = site["centroid"]
    sx, sy, sz = site["box_size"]
    print(
        "\nDone.\n"
        f"\nVina config for site {args.site}:\n"
        f"  --center_x {cx} --center_y {cy} --center_z {cz}\n"
        f"  --size_x {sx:.0f} --size_y {sy:.0f} --size_z {sz:.0f}"
    )


if __name__ == "__main__":
    main()
