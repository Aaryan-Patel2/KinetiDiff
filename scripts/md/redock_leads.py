"""Re-dock top5 leads against WT and R206H ACVR1 receptors.

Uses the corrected pocket center (-17.66, -13.65, 38.41) and saves one best-pose
SDF per (ligand, receptor) combination for use as MD starting geometries.

Usage:
    python scripts/md/redock_leads.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Corrected pocket centre (2026-06-08 fix — old value was (24.87, -12.54, 38.40))
POCKET_CENTER = (-17.66, -13.65, 38.41)
BOX_SIZE      = (25.0, 25.0, 25.0)   # Å — covers full kinase ATP site
EXHAUSTIVENESS = 16
SEED           = 42

RECEPTORS = {
    "WT":    REPO_ROOT / "data/structures/prepped/ACVR1_WT_prepped.pdb",
    "R206H": REPO_ROOT / "data/structures/prepped/ACVR1_R206H_prepped.pdb",
}
LIGANDS_SDF = REPO_ROOT / "results/vina_direct/top5_leads_3D.sdf"
OUT_DIR     = REPO_ROOT / "results/redocked_poses"


def pdb_to_pdbqt(pdb_path: Path, out_path: Path) -> None:
    """Convert protein PDB → PDBQT via openbabel.

    openbabel assigns AutoDock atom types and Gasteiger partial charges.
    The receptor must already be protonated and have no water/heterogens.
    """
    obabel = Path(sys.executable).parent / "obabel"
    result = subprocess.run(
        [str(obabel), str(pdb_path), "-O", str(out_path), "-xr"],
        capture_output=True, text=True, timeout=120,
    )
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(
            f"obabel failed to convert {pdb_path}:\n{result.stderr}"
        )


def sdf_mol_to_pdbqt(mol, seed: int = 42) -> str:
    """Convert an RDKit Mol (with 3D coords) to PDBQT string via meeko."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore[import]

    prep = MoleculePreparation()
    setups = prep.prepare(mol)
    for setup in setups:
        pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(setup)
        if ok:
            return pdbqt_str
    raise RuntimeError("meeko failed to prepare ligand PDBQT")


def dock_one(
    ligand_pdbqt: str,
    receptor_pdbqt: Path,
    name: str,
) -> tuple[float, str]:
    """Dock a single ligand PDBQT string. Returns (score_kcal_mol, pose_pdbqt_str)."""
    from vina import Vina  # type: ignore[import]

    v = Vina(sf_name="vina", seed=SEED, verbosity=0)
    v.set_receptor(str(receptor_pdbqt))

    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as fh:
        fh.write(ligand_pdbqt)
        lig_tmp = Path(fh.name)

    try:
        v.set_ligand_from_file(str(lig_tmp))
        v.compute_vina_maps(
            center=list(POCKET_CENTER),
            box_size=list(BOX_SIZE),
        )
        v.dock(exhaustiveness=EXHAUSTIVENESS, n_poses=5)
        energies = v.energies(n_poses=1)
        best_score = float(energies[0][0])
        best_pose_pdbqt = v.poses(n_poses=1)
    finally:
        lig_tmp.unlink(missing_ok=True)

    return best_score, best_pose_pdbqt


def pdbqt_pose_to_rdkit(pose_pdbqt: str, ref_mol):
    """Parse Vina-placed PDBQT coordinates and graft them onto ref_mol.

    Meeko writes heavy atoms in the same order as the input RDKit mol, so we
    can directly map PDBQT ATOM coordinates (skipping H lines) onto the
    heavy-atom conformer of ref_mol, preserving bond orders and stereo.
    """
    from rdkit import Chem  # type: ignore[import]

    # Parse heavy-atom coordinates from the PDBQT (columns 31-54 per PDB spec)
    coords = []
    for line in pose_pdbqt.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name.upper().startswith("H"):
            continue
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            coords.append((x, y, z))
        except ValueError:
            continue

    ref_no_h = Chem.RemoveHs(ref_mol)
    n_heavy = ref_no_h.GetNumAtoms()
    if len(coords) != n_heavy:
        return None

    edit = Chem.RWMol(ref_no_h)
    edit.RemoveAllConformers()
    conf = Chem.Conformer(n_heavy)
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, (x, y, z))
    edit.AddConformer(conf, assignId=True)
    # AddHs with addCoords=True generates H positions from the heavy-atom
    # conformer geometry; OpenFF molecules from SMILES include explicit Hs,
    # so the SDF must also have Hs for atom counts to agree in load_sdf_molecules.
    return Chem.AddHs(edit.GetMol(), addCoords=True)


def main() -> None:
    from rdkit import Chem  # type: ignore[import]
    from rdkit.Chem import AllChem  # type: ignore[import]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Convert receptors to PDBQT ────────────────────────────────────────
    receptor_pdbqts: dict[str, Path] = {}
    for rec_name, rec_pdb in RECEPTORS.items():
        pdbqt_path = OUT_DIR / f"receptor_{rec_name}.pdbqt"
        if pdbqt_path.exists():
            print(f"  [receptor] {rec_name}: using cached {pdbqt_path.name}")
        else:
            print(f"  [receptor] Converting {rec_name} PDB → PDBQT …")
            pdb_to_pdbqt(rec_pdb, pdbqt_path)
            print(f"  [receptor] {rec_name}: written {pdbqt_path.name}")
        receptor_pdbqts[rec_name] = pdbqt_path

    # ── 2. Load leads ─────────────────────────────────────────────────────────
    suppl = list(Chem.SDMolSupplier(str(LIGANDS_SDF), removeHs=False))
    leads = [(m.GetProp("_Name") if m.HasProp("_Name") else f"mol_{i}", m)
             for i, m in enumerate(suppl) if m is not None]
    print(f"\nLoaded {len(leads)} leads from {LIGANDS_SDF.name}")

    # ── 3. Dock each lead against each receptor ────────────────────────────────
    results = []

    for lig_name, mol in leads:
        # Meeko needs Hs and a conformer; the SDF already has both
        lig_pdbqt = sdf_mol_to_pdbqt(mol)

        for rec_name, rec_pdbqt in receptor_pdbqts.items():
            print(f"  Docking {lig_name} vs {rec_name} …", end=" ", flush=True)
            try:
                score, pose_pdbqt = dock_one(lig_pdbqt, rec_pdbqt, lig_name)
                print(f"{score:.2f} kcal/mol")

                # Convert pose back to RDKit mol
                pose_mol = pdbqt_pose_to_rdkit(pose_pdbqt, mol)
                if pose_mol is None:
                    # Fallback: translate original mol centroid to pocket centre
                    print(f"    WARNING: meeko pose conversion failed; using translated original")
                    pose_mol = Chem.RWMol(mol)
                    conf = pose_mol.GetConformer()
                    centroid = conf.GetPositions().mean(axis=0)
                    target = POCKET_CENTER
                    shift = [target[i] - centroid[i] for i in range(3)]
                    for atom_idx in range(pose_mol.GetNumAtoms()):
                        pos = conf.GetAtomPosition(atom_idx)
                        conf.SetAtomPosition(atom_idx, (pos.x + shift[0], pos.y + shift[1], pos.z + shift[2]))
                    pose_mol = pose_mol.GetMol()

                results.append((lig_name, rec_name, score, pose_mol))

                # Save individual SDF
                out_sdf = OUT_DIR / f"{lig_name}_{rec_name}_docked.sdf"
                w = Chem.SDWriter(str(out_sdf))
                w.write(pose_mol)
                w.close()

            except Exception as exc:
                print(f"FAILED: {exc}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'Lead':<20} {'Receptor':<8} {'Vina (kcal/mol)':>16}")
    print("-" * 48)
    for lig_name, rec_name, score, _ in sorted(results, key=lambda x: x[2]):
        print(f"{lig_name:<20} {rec_name:<8} {score:>16.2f}")

    # Save combined SDF per receptor
    for rec_name in receptor_pdbqts:
        rec_results = [(n, s, m) for n, r, s, m in results if r == rec_name]
        out_combined = OUT_DIR / f"all_leads_{rec_name}_docked.sdf"
        w = Chem.SDWriter(str(out_combined))
        for lig_name, score, pose_mol in sorted(rec_results, key=lambda x: x[1]):
            pose_mol.SetProp("_Name", lig_name)
            pose_mol.SetProp("vina_score", f"{score:.3f}")
            pose_mol.SetProp("receptor", rec_name)
            w.write(pose_mol)
        w.close()
        print(f"\nCombined poses → {out_combined}")

    print("\nDone. Next step: rebuild MD systems from results/redocked_poses/*_docked.sdf")


if __name__ == "__main__":
    main()
