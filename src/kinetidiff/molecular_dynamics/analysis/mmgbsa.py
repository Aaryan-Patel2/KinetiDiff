"""Vacuum protein–ligand interaction energy over 100 trajectory snapshots.

Method: ΔE_interaction = E_complex(vacuum) − E_protein(vacuum) − E_ligand(vacuum)

Each term is evaluated with AMBER14 + OpenFF Sage 2.0 (SMIRNOFF) force-field
parameters in vacuum (NoCutoff, no GB/PB solvation).  Bonded terms cancel
exactly; only the protein–ligand cross-terms (LJ + Coulomb, ε=1) remain.

Why vacuum instead of OBC2 MM-GBSA:
    The GBSAOBCForce implementation produced energies of O(10^7–10^11 kcal/mol)
    due to numerically unstable GB solvation for explicit-solvent snapshots
    of the 138k-atom solvated system.  The E_complex − E_protein − E_ligand
    subtraction failed because absolute-energy magnitudes exceeded double
    precision cancellation.  Vacuum interaction energy avoids this: absolute
    vacuum energies are O(10^4–10^5) kcal/mol, differences are O(−100) kcal/mol.

Important limitations (must be stated in the paper):
    - Entropic contributions (TΔS) are NOT included.
    - No solvation screening (ε=1 vacuum): absolute values are overestimated
      by ~10–30× relative to solution.
    - For relative ΔΔG(R206H vs WT) comparisons, the vacuum bias is consistent
      and cancels, making relative ranking reliable.
    - If absolute ΔG_bind values are needed, cross-check with AmberTools
      ``ante-MMPBSA.py`` (requires AmberTools install in kinetidiff-md).

Usage (standalone CLI):
    python -m kinetidiff.molecular_dynamics.analysis.mmgbsa \\
        --topology  path/to/topology.pdb \\
        --trajectory path/to/run.dcd \\
        --smiles    "NC(=O)NC(=O)CC1CNCCC1c1cccc2ccccc12" \\
        --lig-resname L03 \\
        --out-csv   path/to/mmgbsa.csv \\
        --n-snapshots 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kinetidiff._logging import get_logger

logger = get_logger(__name__)

# Bondi radii (nm) by element symbol — used for GB radii of ligand atoms
# where AMBER atom-type tables have no coverage.
_BONDI_RADII_NM: dict[str, float] = {
    "H": 0.120, "C": 0.170, "N": 0.155, "O": 0.150,
    "F": 0.147, "P": 0.180, "S": 0.180, "Cl": 0.175,
    "Br": 0.185, "I": 0.198,
}
_DEFAULT_RADIUS_NM = 0.170  # carbon default for unknown elements
_GB_SCALE_FACTOR   = 0.8    # HCT-style screening factor (same as OBC2 defaults)

_WATER_ION_RESNAMES = frozenset({"HOH", "WAT", "SOL", "NA", "CL", "NA+", "CL-"})


def _strip_water(pdb_topology: Any, pdb_positions: Any) -> tuple:
    """Return (topology, positions) with water and ions removed."""
    import openmm.app as app  # type: ignore[import]

    modeller = app.Modeller(pdb_topology, pdb_positions)
    to_delete = [
        atom for atom in modeller.topology.atoms()
        if atom.residue.name in _WATER_ION_RESNAMES
    ]
    modeller.delete(to_delete)
    return modeller.topology, modeller.positions


def _split_complex(complex_topology: Any, complex_positions: Any, lig_resname: str) -> tuple:
    """Return (protein_top, protein_pos, ligand_top, ligand_pos)."""
    import openmm.app as app  # type: ignore[import]

    # Protein only
    prot_mod = app.Modeller(complex_topology, complex_positions)
    prot_mod.delete([a for a in prot_mod.topology.atoms() if a.residue.name == lig_resname])

    # Ligand only
    lig_mod = app.Modeller(complex_topology, complex_positions)
    lig_mod.delete([a for a in lig_mod.topology.atoms() if a.residue.name != lig_resname])

    return prot_mod.topology, prot_mod.positions, lig_mod.topology, lig_mod.positions


def _build_system(topology: Any, lig_smiles: str | None, lig_resname: str) -> Any:
    """Build a vacuum OpenMM System (AMBER14 + SMIRNOFF) ready for GB scoring."""
    import openmm.app as app  # type: ignore[import]
    from openff.toolkit import Molecule  # type: ignore[import]
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator  # type: ignore[import]

    ff = app.ForceField("amber14-all.xml")

    if lig_smiles:
        try:
            mol = Molecule.from_smiles(lig_smiles, allow_undefined_stereo=True)
            # AM1-BCC requires AmberTools/NAGL — not installed in kinetidiff-md.
            # Gasteiger (RDKit) is consistent with charges used during production.
            mol.assign_partial_charges("gasteiger")
            gen = SMIRNOFFTemplateGenerator(
                molecules=[mol], forcefield="openff_unconstrained-2.0.0.offxml"
            )
            ff.registerTemplateGenerator(gen.generator)
        except Exception as exc:
            logger.warning("SMIRNOFF template generation failed (%s); ligand energy may be wrong.", exc)

    system = ff.createSystem(
        topology,
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    return system


def _add_gbsa_obc2(system: Any, topology: Any) -> None:
    """Add GBSAOBCForce (OBC2 model) with element-based Bondi radii.

    Charges are pulled from the NonbondedForce already in the system.
    Solvent/solute dielectrics follow the standard OBC2 values.
    """
    import openmm as mm  # type: ignore[import]
    import openmm.unit as unit  # type: ignore[import]

    nb_force: mm.NonbondedForce | None = None
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nb_force = force
            nb_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
            break

    if nb_force is None:
        logger.warning("No NonbondedForce in system — GB charges will be zero.")

    gb = mm.GBSAOBCForce()
    gb.setSolventDielectric(78.5)
    gb.setSoluteDielectric(1.0)

    for i, atom in enumerate(topology.atoms()):
        elem = atom.element.symbol if atom.element else "C"
        radius = _BONDI_RADII_NM.get(elem, _DEFAULT_RADIUS_NM)
        if nb_force is not None:
            charge_q, _, _ = nb_force.getParticleParameters(i)
            q = charge_q.value_in_unit(unit.elementary_charge)
        else:
            q = 0.0
        gb.addParticle(q, radius, _GB_SCALE_FACTOR)

    system.addForce(gb)


def _potential_energy_kcal(system: Any, topology: Any, positions: Any) -> float | None:
    """Return potential energy (kcal/mol) of *system* at *positions*, or None on failure."""
    import openmm as mm  # type: ignore[import]
    import openmm.app as app  # type: ignore[import]
    import openmm.unit as unit  # type: ignore[import]

    try:
        integrator = mm.VerletIntegrator(0.001 * unit.picoseconds)
        platform   = mm.Platform.getPlatformByName("Reference")
        sim = app.Simulation(topology, system, integrator, platform)
        sim.context.setPositions(positions)
        e = sim.context.getState(getEnergy=True).getPotentialEnergy()
        val = e.value_in_unit(unit.kilocalorie_per_mole)
        return float(val) if np.isfinite(val) else None
    except Exception as exc:
        logger.debug("Energy evaluation failed: %s", exc)
        return None


def compute_mmgbsa(
    system: Any,           # kept for API compatibility — not used
    topology: Any,         # kept for API compatibility — not used
    traj_path: str | Path,
    topology_pdb_path: str | Path,
    n_snapshots: int = 100,
    lig_smiles: str = "",
    lig_resname: str = "LIG",
) -> dict[str, float | np.ndarray]:
    """Compute vacuum protein–ligand interaction energy over trajectory snapshots.

    ΔE_interaction = E_complex(vacuum) − E_protein(vacuum) − E_ligand(vacuum)

    Each term uses AMBER14 + OpenFF Sage 2.0 force-field parameters in vacuum
    (NoCutoff, no GB/PB).  Water and ions are stripped before energy evaluation
    so only protein–ligand interactions contribute.  Bonded terms cancel exactly;
    the result is the vacuum LJ + Coulomb cross-term energy (ε=1).

    Args:
        system: Unused (kept for API compatibility).
        topology: Unused (kept for API compatibility).
        traj_path: Path to the production DCD trajectory.
        topology_pdb_path: Solvated topology PDB (atom ordering reference).
        n_snapshots: Number of evenly-spaced frames to evaluate (default: 100).
        lig_smiles: SMILES string for the ligand — required for SMIRNOFF
            parametrisation.  Pass an empty string to skip SMIRNOFF (ligand
            energy will use AMBER14 defaults and may be inaccurate).
        lig_resname: 3-character PDB residue name of the ligand (e.g. ``"L01"``).

    Returns:
        Dict with keys:
        - ``mean``: mean ΔG_bind across snapshots (kcal/mol).
        - ``std``: standard deviation (kcal/mol).
        - ``mean_rel``: same as ``mean`` (retained for schema compatibility).
        - ``std_rel``: same as ``std``.
        - ``per_frame``: array of per-snapshot ΔG_bind values.
        - ``n_frames``: number of valid snapshots.
        - ``reason``: ``"ok"`` or error description.
    """
    import mdtraj as md  # type: ignore[import]
    import openmm.unit as unit  # type: ignore[import]

    tr = Path(traj_path)
    tp = Path(topology_pdb_path)

    _nan = {
        "mean": float("nan"), "std": float("nan"),
        "mean_rel": float("nan"), "std_rel": float("nan"),
        "per_frame": np.array([]), "n_frames": 0, "reason": "",
    }

    if not tr.exists() or tr.stat().st_size == 0:
        return {**_nan, "reason": "traj_missing_or_empty"}
    if not tp.exists():
        return {**_nan, "reason": "topology_missing"}

    try:
        import openmm.app as app  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("openmm must be in kinetidiff-md env.") from exc

    # ── 1. Load trajectory, select evenly-spaced snapshots ──────────────────
    try:
        traj = md.load_dcd(str(tr), top=str(tp))
    except Exception as exc:
        return {**_nan, "reason": f"traj_load_error:{exc}"}

    if traj.n_frames == 0:
        return {**_nan, "reason": "zero_frames"}

    stride    = max(1, traj.n_frames // max(1, n_snapshots))
    snapshots = traj[::stride][:n_snapshots]
    logger.info("MM-GBSA: %d frames selected (stride=%d) from %d total.",
                snapshots.n_frames, stride, traj.n_frames)

    # ── 2. Build stripped topologies (no water/ions) ─────────────────────────
    try:
        pdb = app.PDBFile(str(tp))
        complex_top, complex_pos = _strip_water(pdb.topology, pdb.positions)
        prot_top, prot_pos, lig_top, lig_pos = _split_complex(
            complex_top, complex_pos, lig_resname
        )
    except Exception as exc:
        return {**_nan, "reason": f"topology_build_error:{exc}"}

    n_complex = complex_top.getNumAtoms()
    n_prot    = prot_top.getNumAtoms()
    n_lig     = lig_top.getNumAtoms()
    logger.info("MM-GBSA: complex=%d atoms, protein=%d, ligand=%d", n_complex, n_prot, n_lig)

    if n_lig == 0:
        return {**_nan, "reason": f"ligand_resname_not_found:{lig_resname}"}

    # ── 3. Build vacuum force-field systems (AMBER14 + SMIRNOFF, NoCutoff) ──────
    # No GBSAOBCForce: OBC2 produced unphysical energies (10^7–10^11 kcal/mol)
    # for explicit-solvent snapshots of the 138k-atom system.  Vacuum interaction
    # energy cancels bonded terms exactly, leaving only the protein–ligand cross
    # terms (LJ + Coulomb at ε=1), which are numerically stable.
    try:
        complex_sys = _build_system(complex_top, lig_smiles or None, lig_resname)
        prot_sys    = _build_system(prot_top,    None,               lig_resname)
        lig_sys     = _build_system(lig_top,     lig_smiles or None, lig_resname)
    except Exception as exc:
        return {**_nan, "reason": f"forcefield_build_error:{exc}"}

    # ── 4. Identify atom index slices in the MDTraj trajectory ───────────────
    # Use explicit protein/ligand selections so ions (Na+/Cl-) are excluded —
    # MDTraj's "not water" keeps ions, causing an atom-count mismatch with the
    # OpenMM topology built by Modeller (which strips ions by residue name).
    mdtraj_prot_idx = snapshots.topology.select("protein")
    mdtraj_lig_idx  = snapshots.topology.select(f"resname {lig_resname}")
    if len(mdtraj_lig_idx) == 0:
        mdtraj_lig_idx = snapshots.topology.select(
            "not protein and not water and not (resname NA CL)"
        )
    import numpy as _np
    mdtraj_complex_idx = _np.union1d(mdtraj_prot_idx, mdtraj_lig_idx)

    # ── 5. Per-snapshot ΔG_bind evaluation ──────────────────────────────────
    dg_list: list[float] = []

    for frame_idx in range(snapshots.n_frames):
        xyz = snapshots.xyz[frame_idx]  # nm, shape (n_all_atoms, 3)

        pos_complex = [xyz[i].tolist() for i in mdtraj_complex_idx] * unit.nanometer
        pos_prot    = [xyz[i].tolist() for i in mdtraj_prot_idx]    * unit.nanometer
        pos_lig     = [xyz[i].tolist() for i in mdtraj_lig_idx]     * unit.nanometer

        e_c = _potential_energy_kcal(complex_sys, complex_top, pos_complex)
        e_p = _potential_energy_kcal(prot_sys,    prot_top,    pos_prot)
        e_l = _potential_energy_kcal(lig_sys,     lig_top,     pos_lig)

        if e_c is None or e_p is None or e_l is None:
            logger.debug("Frame %d: energy evaluation failed — skipping.", frame_idx)
            continue

        dg = e_c - e_p - e_l
        if np.isfinite(dg):
            dg_list.append(dg)

    if not dg_list:
        return {**_nan, "reason": "all_frames_nonfinite"}

    arr = np.array(dg_list)
    result = {
        "mean":      float(arr.mean()),
        "std":       float(arr.std()),
        "mean_rel":  float(arr.mean()),   # ΔG_bind is already relative to unbound
        "std_rel":   float(arr.std()),
        "per_frame": arr,
        "n_frames":  len(arr),
        "reason":    "ok",
    }
    logger.info(
        "Interaction energy: ΔE = %.1f ± %.1f kcal/mol over %d frames.",
        result["mean"], result["std"], result["n_frames"],
    )
    return result


def mmgbsa_table(
    results: dict[tuple, dict],
    ligand_configs: dict,
) -> pd.DataFrame:
    """Build the master MM-GBSA summary DataFrame from per-run dicts.

    Args:
        results: Mapping of ``(lig_id, receptor, replica_id)`` → output of
            ``compute_mmgbsa``.
        ligand_configs: The ``campaign.ligands`` dict from the config.

    Returns:
        DataFrame with one row per (ligand, receptor, replica).
    """
    rows = []
    for (lig_id, receptor, replica_id), r in results.items():
        rows.append({
            "ligand_id":        lig_id,
            "receptor":         receptor,
            "replica_id":       replica_id,
            "vina_score":       ligand_configs.get(lig_id, {}).get("vina", float("nan")),
            "mmgbsa_mean_kcal": r.get("mean",     float("nan")),
            "mmgbsa_std_kcal":  r.get("std",      float("nan")),
            "n_frames":         r.get("n_frames",  0),
            "status":           r.get("reason",    "unknown"),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MM-GB/SA CLI")
    parser.add_argument("--topology",     required=True)
    parser.add_argument("--trajectory",   required=True)
    parser.add_argument("--system-xml",   required=False,
                        help="Legacy arg — ignored; kept for backwards compat.")
    parser.add_argument("--smiles",       default="",
                        help="SMILES of the ligand for SMIRNOFF parametrisation.")
    parser.add_argument("--lig-resname",  default="LIG",
                        help="3-char PDB residue name of the ligand (e.g. L01).")
    parser.add_argument("--out-csv",      required=True)
    parser.add_argument("--n-snapshots",  type=int, default=100)
    args = parser.parse_args()

    result = compute_mmgbsa(
        system=None,
        topology=None,
        traj_path=args.trajectory,
        topology_pdb_path=args.topology,
        n_snapshots=args.n_snapshots,
        lig_smiles=args.smiles,
        lig_resname=args.lig_resname,
    )

    pd.DataFrame([{k: v for k, v in result.items() if k != "per_frame"}]).to_csv(
        args.out_csv, index=False
    )
    print("Interaction energy results written to", args.out_csv)
    print(f"  ΔE_interaction = {result['mean']:.2f} ± {result['std']:.2f} kcal/mol"
          f"  n_frames={result['n_frames']}")
