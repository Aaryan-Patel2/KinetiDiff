"""Re-dock protonated (+1) salt forms of top MD hits against their target receptors.

L1p (piperidine [NH2+]) → WT receptor
L2p (piperidine [NH2+]) → R206H receptor

Reuses helpers from redock_leads.py (pocket center, box, exhaustiveness, PDBQT
conversion, and pdbqt_pose_to_rdkit atom-order grafting).

Output: results/redocked_poses/protonated_leads_ranked.sdf
        (2 records, priority order L1p → L2p, with explicit Hs, vina_score prop set)

Usage:
    cd ~/KinetiDiff
    ~/.conda/envs/kinetidiff-md/bin/python3 scripts/md/redock_protonated_leads.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing helpers from redock_leads.py in the same directory
sys.path.insert(0, str(Path(__file__).parent))
from redock_leads import (  # noqa: E402
    POCKET_CENTER,
    BOX_SIZE,
    EXHAUSTIVENESS,
    SEED,
    sdf_mol_to_pdbqt,
    dock_one,
    pdbqt_pose_to_rdkit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR   = REPO_ROOT / "results" / "redocked_poses"

# Protonated forms: piperidine secondary amine → [NH2+] (formal charge +1)
JOBS: list[tuple[str, str, str]] = [
    (
        "L1p",
        "O=C(NC1CC[NH2+]CC1c1ccc[nH]1)c1c(C2CCCOC2)ccc2ccccc12",
        "receptor_WT.pdbqt",
    ),
    (
        "L2p",
        "O=S(=O)(Cc1cccnc1)Nc1ccc(F)cc1-c1ccc2ccccc2c1C1CC[NH2+]CC1",
        "receptor_R206H.pdbqt",
    ),
]


def main() -> None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    poses: list[tuple[str, float, Chem.Mol]] = []

    for lig_id, smiles, rec_fname in JOBS:
        rec_pdbqt = OUT_DIR / rec_fname
        if not rec_pdbqt.exists():
            raise FileNotFoundError(
                f"Receptor PDBQT not found: {rec_pdbqt}\n"
                "Run scripts/md/redock_leads.py first to generate receptor PDBQTs."
            )

        print(f"\n[{lig_id}] Building 3D conformer from SMILES …")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit could not parse SMILES for {lig_id}: {smiles}")
        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, randomSeed=SEED)
        if result != 0:
            raise RuntimeError(f"[{lig_id}] EmbedMolecule failed (result={result})")
        AllChem.MMFFOptimizeMolecule(mol)

        # Sanity check: formal charge must be +1
        fc = Chem.GetFormalCharge(mol)
        assert fc == 1, f"[{lig_id}] Expected formal charge +1, got {fc}"
        print(f"[{lig_id}] Formal charge confirmed: {fc:+d}")

        print(f"[{lig_id}] Docking against {rec_fname} …", end=" ", flush=True)
        lig_pdbqt = sdf_mol_to_pdbqt(mol)
        score, pose_pdbqt = dock_one(lig_pdbqt, rec_pdbqt, lig_id)
        print(f"{score:.2f} kcal/mol")

        pose_mol = pdbqt_pose_to_rdkit(pose_pdbqt, mol)
        if pose_mol is None:
            # Fallback: translate conformer centroid to pocket centre
            print(f"  WARNING: meeko pose conversion failed — using translated 3D conformer")
            from rdkit.Chem import rdGeometry
            conf = mol.GetConformer()
            centroid = conf.GetPositions().mean(axis=0)
            shift = [POCKET_CENTER[i] - centroid[i] for i in range(3)]
            edit = Chem.RWMol(mol)
            c = edit.GetConformer()
            for idx in range(edit.GetNumAtoms()):
                pos = c.GetAtomPosition(idx)
                c.SetAtomPosition(idx, (pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]))
            pose_mol = edit.GetMol()

        pose_mol.SetProp("_Name", lig_id)
        pose_mol.SetProp("vina_score", f"{score:.3f}")
        pose_mol.SetProp("smiles", smiles)
        pose_mol.SetProp("receptor", rec_fname.replace(".pdbqt", "").replace("receptor_", ""))

        # Save individual SDF
        out_single = OUT_DIR / f"{lig_id}_docked.sdf"
        w = Chem.SDWriter(str(out_single))
        w.write(pose_mol)
        w.close()
        print(f"[{lig_id}] Pose written → {out_single.name}")

        poses.append((lig_id, score, pose_mol))

    # Combined SDF in priority order (L1p first, L2p second) → used as sdf_input
    out_combined = OUT_DIR / "protonated_leads_ranked.sdf"
    w = Chem.SDWriter(str(out_combined))
    for lig_id, score, mol in poses:
        w.write(mol)
    w.close()

    print(f"\n{'─' * 52}")
    print(f"{'Ligand':<10} {'Receptor':<12} {'Vina (kcal/mol)':>16}")
    print(f"{'─' * 52}")
    for lig_id, score, _ in poses:
        rec = next(r for (i, _, r) in JOBS if i == lig_id).replace(".pdbqt", "").replace("receptor_", "")
        print(f"{lig_id:<10} {rec:<12} {score:>16.2f}")
    print(f"{'─' * 52}")
    print(f"\nCombined SDF → {out_combined}")
    print("\nNext step: copy the Vina scores above into configs/simulation_protonated.yaml")
    print("           then run the smoke test before submitting the full campaign.")


if __name__ == "__main__":
    main()
