#!/usr/bin/env python3
"""Dock known ALK2/ACVR1 inhibitors (clinical benchmark) against WT and R206H.

Fetches canonical SMILES from PubChem by name, generates 3D conformers, docks
with the same Vina parameters used for the KinetiDiff leads (pocket center
(-17.66, -13.65, 38.41), exhaustiveness=16, seed=42).

Results written to results/clinical_benchmark/clinical_docking_scores.csv.

Usage:
    python scripts/clinical_benchmark_docking.py
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results" / "clinical_benchmark"

POCKET_CENTER = (-17.66, -13.65, 38.41)
BOX_SIZE = (25.0, 25.0, 25.0)
EXHAUSTIVENESS = 16
SEED = 42

RECEPTOR_PDBQTS = {
    "WT":    REPO_ROOT / "results" / "redocked_poses" / "receptor_WT.pdbqt",
    "R206H": REPO_ROOT / "results" / "redocked_poses" / "receptor_R206H.pdbqt",
}

# Known ALK2/ACVR1 inhibitors for benchmark comparison.
# SMILES fetched dynamically from PubChem; fallback hardcoded from ChEMBL.
CLINICAL_COMPOUNDS = [
    {
        "name": "LDN-193189",
        "pubchem_name": "LDN-193189",
        "smiles_fallback": "C1CN(CCN1)c1ccc(-c2cn3nccc3n2)cc1",
        "notes": "Selective BMP type-I receptor inhibitor; ALK2 IC50 ~5 nM",
    },
    {
        "name": "Saracatinib",
        "pubchem_name": "Saracatinib",
        "smiles_fallback": (
            "COc1cc2c(Nc3cccc(Cl)c3F)nc(Nc3ccc(F)cc3)nc2cc1OCCCN1CCOCC1"
        ),
        "notes": "Dual Src/ABL inhibitor; ALK2/ACVR1 kinase activity",
    },
    {
        "name": "Zilurgisertib",
        "pubchem_name": "Zilurgisertib",
        "smiles_fallback": (
            "O=C(NC1CCN(c2ccc(-c3ccc4[nH]cc(C#N)c4c3)cc2)CC1)"
            "c1cc2c(cn1)cc(F)cn2"
        ),
        "notes": "Selective ALK2 inhibitor; FOP clinical trial (RESILIENT)",
    },
    {
        "name": "LDN-212854",
        "pubchem_name": "LDN-212854",
        "smiles_fallback": (
            "Fc1ccc(-c2cnc3cc(-c4ccncc4)ccc3n2)cc1"
        ),
        "notes": "Second-gen ALK2-selective inhibitor; improved selectivity over LDN-193189",
    },
]


def fetch_smiles_pubchem(name: str) -> str | None:
    """Fetch canonical SMILES from PubChem REST API by compound name."""
    url = (
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{urllib.request.quote(name)}/property/CanonicalSMILES/JSON"
    )
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception as exc:
        print(f"  PubChem lookup failed for {name!r}: {exc}")
        return None


def smiles_to_3d_mol(smiles: str, name: str):
    """Generate a 3D RDKit Mol from SMILES using ETKDG."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        # Fallback: try without ETKDGv3
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result != 0:
        raise RuntimeError(f"3D embedding failed for {name}")
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def mol_to_pdbqt(mol) -> str:
    """Convert RDKit Mol (with 3D) to PDBQT string via meeko."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore[import]

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    for setup in setups:
        pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(setup)
        if ok:
            return pdbqt_str
    raise RuntimeError("meeko failed to produce PDBQT")


def dock(ligand_pdbqt: str, receptor_pdbqt: Path) -> float:
    """Dock ligand PDBQT against receptor; return best Vina score (kcal/mol)."""
    from vina import Vina  # type: ignore[import]

    v = Vina(sf_name="vina", seed=SEED, verbosity=0)
    v.set_receptor(str(receptor_pdbqt))
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as fh:
        fh.write(ligand_pdbqt)
        lig_tmp = Path(fh.name)
    try:
        v.set_ligand_from_file(str(lig_tmp))
        v.compute_vina_maps(center=list(POCKET_CENTER), box_size=list(BOX_SIZE))
        v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=5)
        return float(v.energies(n_poses=1)[0][0])
    finally:
        lig_tmp.unlink(missing_ok=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for compound in CLINICAL_COMPOUNDS:
        name = compound["name"]
        print(f"\n{'─'*60}")
        print(f"  {name}")

        # 1. Get SMILES
        smiles = fetch_smiles_pubchem(compound["pubchem_name"])
        if smiles is None:
            smiles = compound["smiles_fallback"]
            smiles_source = "fallback"
            print(f"  Using fallback SMILES: {smiles[:50]}...")
        else:
            smiles_source = "pubchem"
            print(f"  PubChem SMILES: {smiles[:50]}...")

        # 2. 3D conformer
        try:
            mol = smiles_to_3d_mol(smiles, name)
            pdbqt = mol_to_pdbqt(mol)
        except Exception as exc:
            print(f"  Conformer/PDBQT failed: {exc}")
            rows.append({"compound": name, "smiles": smiles,
                         "smiles_source": smiles_source,
                         "WT_vina": None, "R206H_vina": None,
                         "delta_delta_G": None, "notes": compound["notes"],
                         "error": str(exc)})
            continue

        # 3. Dock against both receptors
        scores: dict[str, float | None] = {}
        for rec_name, rec_pdbqt in RECEPTOR_PDBQTS.items():
            print(f"  Docking vs {rec_name} …", end=" ", flush=True)
            try:
                score = dock(pdbqt, rec_pdbqt)
                scores[rec_name] = score
                print(f"{score:.2f} kcal/mol")
            except Exception as exc:
                print(f"FAILED: {exc}")
                scores[rec_name] = None

        ddg = None
        if scores.get("R206H") is not None and scores.get("WT") is not None:
            ddg = scores["R206H"] - scores["WT"]

        rows.append({
            "compound": name,
            "smiles": smiles,
            "smiles_source": smiles_source,
            "WT_vina": scores.get("WT"),
            "R206H_vina": scores.get("R206H"),
            "delta_delta_G": ddg,
            "notes": compound["notes"],
            "error": "",
        })

    # Write CSV
    out_csv = OUT_DIR / "clinical_docking_scores.csv"
    fieldnames = ["compound", "WT_vina", "R206H_vina", "delta_delta_G",
                  "smiles", "smiles_source", "notes", "error"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'═'*60}")
    print("CLINICAL BENCHMARK DOCKING RESULTS")
    print(f"{'═'*60}")
    print(f"{'Compound':<20} {'WT (kcal/mol)':>14} {'R206H':>8} {'ΔΔG':>8}")
    print("─" * 54)
    for row in rows:
        wt = f"{row['WT_vina']:.2f}" if row["WT_vina"] is not None else "FAIL"
        r2 = f"{row['R206H_vina']:.2f}" if row["R206H_vina"] is not None else "FAIL"
        ddg_str = f"{row['delta_delta_G']:+.2f}" if row["delta_delta_G"] is not None else "N/A"
        print(f"{row['compound']:<20} {wt:>14} {r2:>8} {ddg_str:>8}")
    print(f"\nResults saved to: {out_csv}")


if __name__ == "__main__":
    sys.exit(main())
