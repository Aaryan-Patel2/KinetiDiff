"""Build the solvated OpenMM system for a protein-ligand complex.

Uses OpenFF Sage 2.0 (SMIRNOFF) for the ligand via SMIRNOFFTemplateGenerator,
and AMBER ff14SB + TIP3P for the protein/water.  No GAFF2.
"""

from __future__ import annotations

import fcntl
from pathlib import Path
from typing import Any

from kinetidiff._logging import get_logger

logger = get_logger(__name__)


def build_solvated_system(
    protein_pdb_path: Path,
    off_molecule: Any,
    ligand_id: str,
    receptor_type: str,
    cfg: dict,
    out_dir: Path,
) -> tuple[Any, Any, Any]:
    """Combine protein + ligand, solvate, and return the OpenMM system.

    Why:
        No runtime pip installs — the conda env pins openmmforcefields and
        openff-toolkit.  We load the force field once per call so the function
        is safely parallelisable across SLURM tasks.

    Args:
        protein_pdb_path: Prepped apo protein PDB (A3F removed).
        off_molecule: OpenFF Molecule with a conformer at the docked pose.
        ligand_id: e.g. ``"L1"``.
        receptor_type: ``"WT"`` or ``"R206H"``.
        cfg: Resolved simulation config dict (from ``configs/simulation.yaml``).
        out_dir: Directory to write the topology PDB and system XML.

    Returns:
        ``(system, modeller, topology_pdb_path)`` where *system* is the
        ``openmm.System``, *modeller* is the ``Modeller`` with solvated
        positions, and *topology_pdb_path* is the path to the written PDB.
    """
    try:
        import openmm as mm  # type: ignore[import]
        import openmm.app as app  # type: ignore[import]
        import openmm.unit as unit  # type: ignore[import]
        from openmmforcefields.generators import SMIRNOFFTemplateGenerator  # type: ignore[import]
        from openff.toolkit.topology import Molecule as OFFMolecule  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "openmm, openmmforcefields, and openff-toolkit must be in kinetidiff-md."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{ligand_id}_{receptor_type}"
    topology_pdb_path = out_dir / f"topology_{tag}.pdb"
    system_xml_path   = out_dir / f"system_{tag}.xml"

    # Exclusive file lock prevents concurrent SLURM tasks from each calling
    # addSolvent() on the same system and writing different atom counts to the
    # same cached path (non-deterministic solvation → topology mismatch).
    lock_path = out_dir / f".{tag}.lock"
    with open(lock_path, "w") as _lock_fh:
        fcntl.flock(_lock_fh, fcntl.LOCK_EX)

        if topology_pdb_path.exists() and system_xml_path.exists():
            logger.info("[%s] System files already exist; loading cached system.", tag)
            pdb = app.PDBFile(str(topology_pdb_path))
            with open(system_xml_path) as fh:
                system = mm.XmlSerializer.deserialize(fh.read())
            modeller = app.Modeller(pdb.topology, pdb.positions)
            return system, modeller, topology_pdb_path

        logger.info("[%s] Building solvated OpenMM system...", tag)

        # ── Load protein ──────────────────────────────────────────────────────
        pdb = app.PDBFile(str(protein_pdb_path))

        # ── Pre-assign partial charges ────────────────────────────────────────
        # SMIRNOFFTemplateGenerator needs partial charges on the molecule or it
        # falls back to AM1-BCC (requires AmberTools/sqm, not in this env).
        # Priority: NAGL (AM1-BCC quality, no AmberTools) → Gasteiger (smoke/dev).
        # Gasteiger is NOT acceptable for production — raises a hard error so the
        # caller can install openff-nagl-models or ambertools before running.
        if off_molecule.partial_charges is None:
            _charge_assigned = False
            try:
                from openff.nagl_models import list_available_nagl_models  # type: ignore[import]
                from openff.toolkit import NAGLToolkitWrapper  # type: ignore[import]
                _nagl_models = list_available_nagl_models()
                _am1_models = [m for m in _nagl_models if "am1bcc" in str(m).lower()]
                if _am1_models:
                    off_molecule.assign_partial_charges(
                        str(_am1_models[0]), toolkit_registry=NAGLToolkitWrapper()
                    )
                    logger.info("[%s] NAGL partial charges assigned (%s).", tag, _am1_models[0])
                    _charge_assigned = True
            except Exception as _e:
                logger.debug("[%s] NAGL charge attempt failed: %s", tag, _e)

            if not _charge_assigned:
                allow_gasteiger = cfg.get("forcefield", {}).get("allow_gasteiger_charges", False)
                if not allow_gasteiger:
                    raise RuntimeError(
                        f"[{tag}] Cannot assign AM1-BCC charges: AmberTools and NAGL models "
                        "are both unavailable. Install 'openff-nagl-models' in the "
                        "kinetidiff-md conda env, or set "
                        "forcefield.allow_gasteiger_charges=true in simulation.yaml "
                        "(smoke/dev only — not acceptable for production MD)."
                    )
                logger.warning(
                    "[%s] NAGL unavailable; using Gasteiger charges. "
                    "NOT acceptable for production — install openff-nagl-models.",
                    tag,
                )
                from openff.toolkit.utils import RDKitToolkitWrapper  # type: ignore[import]
                off_molecule.assign_partial_charges("gasteiger", toolkit_registry=RDKitToolkitWrapper())

        # ── Set up force field with SMIRNOFF ligand template ─────────────────
        forcefield = app.ForceField(
            cfg["forcefield"]["protein"],
            cfg["forcefield"]["water"],
        )
        smirnoff_gen = SMIRNOFFTemplateGenerator(
            molecules=[off_molecule],
            forcefield=cfg["forcefield"]["ligand"],
        )
        forcefield.registerTemplateGenerator(smirnoff_gen.generator)

        # ── Add ligand to Modeller ────────────────────────────────────────────
        # Use the config resname (e.g. "L01") — not mol.name ("L1") — so the PDB
        # topology residue name matches what the analysis scripts query.
        lig_resname = cfg["campaign"]["ligands"][ligand_id]["resname"]
        modeller = app.Modeller(pdb.topology, pdb.positions)
        lig_topology, lig_positions = _off_molecule_to_openmm(off_molecule, lig_resname)
        modeller.add(lig_topology, lig_positions)
        logger.info("[%s] Ligand %s added to modeller.", tag, ligand_id)

        # ── Add hydrogens ─────────────────────────────────────────────────────
        modeller.addHydrogens(forcefield, pH=7.4)

        # ── Solvate ───────────────────────────────────────────────────────────
        padding = cfg["solvent"]["padding_nm"] * unit.nanometers
        ionic_str = cfg["solvent"]["ionic_strength_M"] * unit.molar
        modeller.addSolvent(
            forcefield,
            model="tip3p",
            padding=padding,
            ionicStrength=ionic_str,
            neutralize=True,
        )
        logger.info(
            "[%s] Solvated system: %d atoms total.",
            tag, modeller.topology.getNumAtoms(),
        )

        # ── Build OpenMM System ───────────────────────────────────────────────
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
        )

        # ── Persist topology PDB + system XML (inside lock — atomic write) ───
        with open(topology_pdb_path, "w") as fh:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)

        with open(system_xml_path, "w") as fh:
            fh.write(mm.XmlSerializer.serialize(system))

        logger.info("[%s] System written: %s, %s", tag, topology_pdb_path.name, system_xml_path.name)
        return system, modeller, topology_pdb_path


def _off_molecule_to_openmm(off_molecule: Any, resname: str) -> tuple[Any, Any]:
    """Convert an OpenFF Molecule to an OpenMM topology + positions.

    Why:
        OpenMM Modeller.add() needs an openmm.app.Topology, not an OpenFF one.
        We extract it via the OpenFF→OpenMM topology conversion API.

    Args:
        off_molecule: OpenFF Molecule with a conformer.
        resname: Residue name to assign in the OpenMM topology.

    Returns:
        ``(openmm_topology, positions_in_nanometers)``
    """
    import openmm.unit as unit  # type: ignore[import]
    import numpy as np

    off_topology = off_molecule.to_topology()
    openmm_topology = off_topology.to_openmm()

    # Fix residue names to the configured resname
    for residue in openmm_topology.residues():
        residue.name = resname

    if off_molecule.n_conformers == 0:
        raise ValueError(
            f"OpenFF Molecule '{resname}' has no conformers. "
            "Call generate_conformers() before building the system."
        )

    # Off-toolkit stores positions in angstroms; OpenMM needs nm
    positions_A = off_molecule.conformers[0].magnitude  # (n_atoms, 3) in Å
    positions_nm = positions_A / 10.0
    openmm_positions = [
        (float(p[0]), float(p[1]), float(p[2])) * unit.nanometers
        for p in positions_nm
    ]
    return openmm_topology, openmm_positions
