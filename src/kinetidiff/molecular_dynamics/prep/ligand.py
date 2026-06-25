"""Ligand preparation: SMILES → OpenFF Molecule and SDF loading.

All ligands must be parametrized with OpenFF Sage 2.0 (SMIRNOFF).
GAFF2 is explicitly excluded per the locked force-field stack in MD-PIPELINE.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kinetidiff._logging import get_logger

logger = get_logger(__name__)


def smiles_to_openff_molecule(
    smiles: str,
    name: str,
    resname: str,
) -> Any:
    """Convert a SMILES string to an OpenFF Molecule with a unique name.

    Generates a single conformer (needed for OpenMM Modeller placement).
    The caller is responsible for placing the molecule at the docked pose
    coordinates extracted from the SDF.

    Args:
        smiles: SMILES string of the ligand.
        name: Human-readable name (e.g. ``"L1"``).
        resname: PDB residue name (3–4 chars, e.g. ``"L01"``).

    Returns:
        An ``openff.toolkit.topology.Molecule`` ready for SMIRNOFF parametrization.
    """
    try:
        from openff.toolkit.topology import Molecule  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "openff-toolkit must be installed in the kinetidiff-md conda env."
        ) from exc

    logger.info("Building OpenFF Molecule for %s (%s) from SMILES.", name, resname)
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    mol.name = name
    mol.generate_conformers(n_conformers=1)
    return mol


def load_sdf_molecules(
    sdf_path: Path,
    ligand_configs: dict[str, dict],
) -> dict[str, Any]:
    """Load docked poses from an SDF and pair with per-ligand OpenFF molecules.

    Matches SDF records to *ligand_configs* by order (rank 1→L1, rank 2→L2,
    etc.) and sets the OpenFF molecule conformer to the docked pose coordinates
    so that the MD system starts from the docked geometry.

    Args:
        sdf_path: Path to the multi-molecule SDF (e.g. ``results/vina_direct/top5_leads_3D.sdf``).
        ligand_configs: Dict mapping ligand IDs (``"L1"``, ...) to config dicts
            with keys ``smiles``, ``resname``, etc.

    Returns:
        Dict mapping ligand ID → OpenFF Molecule with docked-pose conformer.
    """
    try:
        from rdkit import Chem  # type: ignore[import]
        from rdkit.Chem import AllChem  # type: ignore[import]
        import numpy as np
        from openff.toolkit.topology import Molecule  # type: ignore[import]
        from openff.units import unit  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "rdkit and openff-toolkit must be installed in the kinetidiff-md conda env."
        ) from exc

    sdf_supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    rdkit_mols = [m for m in sdf_supplier if m is not None]
    if not rdkit_mols:
        raise FileNotFoundError(f"No valid molecules found in {sdf_path}")

    priority_order = list(ligand_configs.keys())
    result: dict[str, Any] = {}

    for idx, lig_id in enumerate(priority_order):
        cfg = ligand_configs[lig_id]
        off_mol = smiles_to_openff_molecule(cfg["smiles"], lig_id, cfg["resname"])

        # If a docked pose is available, override the conformer.
        if idx < len(rdkit_mols):
            rdmol = rdkit_mols[idx]
            if rdmol.GetNumConformers() > 0:
                conf = rdmol.GetConformer(0)
                positions_A = np.array([
                    [conf.GetAtomPosition(i).x,
                     conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z]
                    for i in range(rdmol.GetNumAtoms())
                ])
                try:
                    # Store in Å — _off_molecule_to_openmm expects Å and converts to nm itself.
                    # Do NOT pre-convert to nm here; doing so causes a double divide-by-10.
                    off_mol._conformers = [positions_A * unit.angstrom]
                    logger.info(
                        "[%s] Docked pose loaded from SDF (record %d, %d atoms).",
                        lig_id, idx + 1, rdmol.GetNumAtoms(),
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] Could not set docked conformer: %s. Using generated conformer.",
                        lig_id, exc,
                    )
        else:
            logger.warning(
                "[%s] No SDF record at index %d; using RDKit-generated conformer.", lig_id, idx,
            )

        result[lig_id] = off_mol

    return result
