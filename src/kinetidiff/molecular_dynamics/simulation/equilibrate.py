"""Minimization → NVT → NPT equilibration stage.

Idempotent: returns immediately if ``equil_{tag}.xml`` and
``equil_{tag}.pdb`` already exist in *out_dir*.  Never re-runs completed
equilibration without an explicit ``force=True``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kinetidiff._logging import get_logger

logger = get_logger(__name__)


def _add_positional_restraints(
    system: Any,
    modeller: Any,
    k_kcal_per_mol_A2: float,
) -> None:
    """Add harmonic restraints on all heavy atoms (protein + ligand).

    Why:
        During NVT and early NPT equilibration we restrain heavy atoms to avoid
        clash-driven explosion before solvent density stabilises.  k=0 is a
        no-op (no force added).

    Args:
        system: OpenMM System.
        modeller: Modeller object (provides topology + positions for atom loop).
        k_kcal_per_mol_A2: Spring constant in kcal/mol/Å².  Typical: 2.0.
    """
    if k_kcal_per_mol_A2 == 0.0:
        return

    import openmm as mm  # type: ignore[import]
    import openmm.unit as unit  # type: ignore[import]

    k_kJ_per_nm2 = k_kcal_per_mol_A2 * 418.4  # kcal/mol/Å² → kJ/mol/nm²
    restraint = mm.CustomExternalForce("k*periodicdistance(x,y,z,x0,y0,z0)^2")
    restraint.addGlobalParameter("k", k_kJ_per_nm2)
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    positions = modeller.getPositions()
    for idx, (atom, pos) in enumerate(zip(modeller.topology.atoms(), positions)):
        if atom.element is not None and atom.element.symbol != "H":
            restraint.addParticle(idx, [pos.x, pos.y, pos.z])

    system.addForce(restraint)


def run_equilibration(
    system: Any,
    modeller: Any,
    ligand_id: str,
    receptor_type: str,
    cfg: dict,
    out_dir: Path,
    platform_name: str = "OpenCL",
    force: bool = False,
) -> Path:
    """Run the full equilibration pipeline (minimize → NVT → NPT).

    Idempotent: if the output state XML and PDB exist and *force* is False,
    returns immediately with the cached paths.

    Platform fallback: tries CUDA first; if it fails with PTX/compute capability
    mismatch, falls back to OpenCL. On HPC clusters, OpenCL is often the more
    robust option (validated on MIT Engaging H200 nodes).

    Args:
        system: Solvated OpenMM System (from ``prep.system.build_solvated_system``).
        modeller: Modeller with solvated positions.
        ligand_id: e.g. ``"L1"``.
        receptor_type: ``"WT"`` or ``"R206H"``.
        cfg: Resolved simulation config dict.
        out_dir: Directory to write equilibrated state XML + PDB.
        platform_name: OpenMM platform (``"CUDA"``, ``"CPU"``).
        force: If True, re-run even if output files exist.

    Returns:
        Path to the equilibrated state XML (for production stage to load).
    """
    import copy
    import openmm as mm  # type: ignore[import]
    import openmm.app as app  # type: ignore[import]
    import openmm.unit as unit  # type: ignore[import]

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{ligand_id}_{receptor_type}"
    state_path = out_dir / f"equil_{tag}.xml"
    top_path   = out_dir / f"equil_{tag}.pdb"

    if state_path.exists() and top_path.exists() and not force:
        logger.info("[%s] Equilibration outputs exist; skipping.", tag)
        return state_path

    sim_cfg = cfg["simulation"]
    temperature = sim_cfg["temperature_K"] * unit.kelvin
    pressure    = sim_cfg["pressure_bar"] * unit.bar
    dt          = sim_cfg["timestep_ps"] * unit.picoseconds
    restraint_k = sim_cfg["restraint_k_kcal"]

    # Try to get the platform; fall back to OpenCL on CUDA failure
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        platform_props = {"CudaPrecision": "mixed"} if platform_name == "CUDA" else {}
    except Exception as e:
        if platform_name == "CUDA" and "UNSUPPORTED_PTX" in str(e):
            logger.warning(
                "[%s] CUDA platform not available (PTX/compute capability mismatch); "
                "falling back to OpenCL.", tag
            )
            platform = mm.Platform.getPlatformByName("OpenCL")
            platform_props = {}
        else:
            raise

    # ── 1. Energy minimisation ─────────────────────────────────────────────
    logger.info("[%s] Energy minimisation (max %d iterations)...", tag, sim_cfg["minimize_max_iterations"])
    equil_system = copy.deepcopy(system)
    _add_positional_restraints(equil_system, modeller, restraint_k)

    integrator = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    sim = app.Simulation(modeller.topology, equil_system, integrator, platform, platform_props)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=sim_cfg["minimize_max_iterations"])
    logger.info("[%s] Minimisation complete.", tag)

    # ── 2. NVT equilibration (100 ps) ──────────────────────────────────────
    logger.info("[%s] NVT equilibration (%d steps)...", tag, sim_cfg["nvt_steps"])
    sim.context.setVelocitiesToTemperature(temperature)
    sim.step(sim_cfg["nvt_steps"])
    logger.info("[%s] NVT complete.", tag)

    # ── 3. NPT equilibration (100 ps, Barostat) ────────────────────────────
    logger.info("[%s] NPT equilibration (%d steps)...", tag, sim_cfg["npt_steps"])
    equil_system.addForce(mm.MonteCarloBarostat(pressure, temperature, 25))
    integrator2 = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    sim2 = app.Simulation(modeller.topology, equil_system, integrator2, platform, platform_props)
    state = sim.context.getState(getPositions=True, getVelocities=True)
    sim2.context.setPositions(state.getPositions())
    sim2.context.setVelocities(state.getVelocities())
    sim2.step(sim_cfg["npt_steps"])
    logger.info("[%s] NPT complete.", tag)

    # ── 4. Save state ──────────────────────────────────────────────────────
    # enforcePeriodicBox=True for the XML so positions are in the primary unit cell,
    # which keeps the state compact and valid for production.py to load directly.
    final_state = sim2.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=True)
    with open(state_path, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(final_state))

    # enforcePeriodicBox=False for the PDB so the protein+ligand complex is written
    # as a single connected unit rather than wrapping molecules to arbitrary images.
    # The PDB is only used for visual inspection and as a topology reference — it is
    # NOT loaded by production.py (which reads the XML state above).
    pdb_state = sim2.context.getState(getPositions=True, enforcePeriodicBox=False)
    with open(top_path, "w") as fh:
        app.PDBFile.writeFile(
            sim2.topology,
            pdb_state.getPositions(),
            fh,
        )

    logger.info("[%s] Equilibration state written: %s", tag, state_path)
    return state_path
