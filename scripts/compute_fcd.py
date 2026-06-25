#!/usr/bin/env python3
"""Compute Fréchet ChemNet Distance (FCD) for generated molecules.

fcd_torch does not bundle ChEMBL SMILES, only the ChemNet model weights.
We compute:
  (1) Self-FCD (sanity check, should be ~0)
  (2) Internal-split FCD (first-half vs second-half) — measures set diversity
  (3) Guided vs Unguided FCD — measures how much guidance shifts the distribution

Usage:
    python scripts/compute_fcd.py
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SMILES_FILES = {
    "guided_multi_objective": REPO_ROOT / "results/multi_objective/multi_objective_5k_molecules.smi",
    "no_guidance": REPO_ROOT / "results/no_guidance/generated10_molecules.smi",
    "unguided_1k": REPO_ROOT / "results/no_guidance/unguided_1k_molecules.smi",
}
OUT_DIR = REPO_ROOT / "results/fcd"


def load_smiles(path: Path) -> list[str]:
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def main() -> None:
    try:
        import fcd_torch
    except ImportError:
        print("fcd_torch not installed. Run: pip install fcd_torch")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fcd_model = fcd_torch.FCD(device="cpu", n_jobs=4)

    sets: dict[str, list[str]] = {}
    for label, path in SMILES_FILES.items():
        if path.exists():
            smiles = load_smiles(path)
            if len(smiles) >= 2:
                sets[label] = smiles
                print(f"Loaded {len(smiles)} SMILES [{label}]")
            else:
                print(f"Skipping [{label}]: only {len(smiles)} SMILES")
        else:
            print(f"Skipping [{label}]: file not found")

    results: list[str] = ["Fréchet ChemNet Distance Report", "=" * 50]

    # Self-FCD sanity checks
    for label, smiles in sets.items():
        score = fcd_model(ref=smiles, gen=smiles)
        line = f"Self-FCD [{label}]: {score:.4f}  (should be ~0)"
        print(line); results.append(line)

    # Internal diversity: first-half vs second-half
    for label, smiles in sets.items():
        if len(smiles) >= 100:
            half = len(smiles) // 2
            score = fcd_model(ref=smiles[:half], gen=smiles[half:])
            line = f"Internal-split FCD [{label}]: {score:.4f}  (diversity of set; lower = more uniform)"
            print(line); results.append(line)

    # Guided vs Unguided (the key comparison for the paper)
    if "guided_multi_objective" in sets and "unguided_1k" in sets:
        score = fcd_model(ref=sets["guided_multi_objective"], gen=sets["unguided_1k"])
        line = f"Guided vs Unguided-1k FCD: {score:.4f}  (distribution shift from guidance)"
        print(line); results.append(line)

    results += [
        "",
        "Interpretation:",
        "  Internal-split FCD < 1: set is very uniform (low diversity)",
        "  Internal-split FCD 1-5: moderate diversity within the set",
        "  Guided vs Unguided FCD: how much Vina guidance shifts the chemical space",
        "  For ChEMBL reference FCD, download ChEMBL25 SMILES and pass as ref=.",
    ]

    report = OUT_DIR / "fcd_report.txt"
    report.write_text("\n".join(results))
    print(f"\nReport written to: {report}")


if __name__ == "__main__":
    sys.exit(main())
