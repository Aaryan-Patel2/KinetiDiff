"""Production NPT MD — 100 ns per run, checkpoint-resumed, SIGTERM-safe.

Design contracts:
- Resumes from ``.chk`` if it exists; skips already-complete trajectories.
- Writes OpenMM CheckpointReporter every ``checkpoint_interval`` steps.
- Catches SIGTERM (SLURM preemption on ``mit_preemptable``) and saves a
  checkpoint before exiting cleanly so ``--requeue`` picks up exactly where
  the job left off.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path
from typing import Any

from kinetidiff._logging import get_logger

logger = get_logger(__name__)

_SHUTDOWN_REQUESTED: bool = False


def _handle_sigterm(signum: int, frame: Any) -> None:  # noqa: ANN401
    """Handle SIGTERM from SLURM preemption — checkpoint then exit cleanly."""
    global _SHUTDOWN_REQUESTED
    logger.warning("SIGTERM received (SLURM preemption). Will checkpoint after current step.")
    _SHUTDOWN_REQUESTED = True


def run_production(
    system: Any,
    modeller: Any,
    ligand_id: str,
    receptor_type: str,
    replica_id: int,
    seed: int,
    cfg: dict,
    scratch_dir: Path,
    topology_pdb_path: Path,
    equil_state_path: Path,
    platform_name: str = "OpenCL",
    smoke_test: bool = False,
) -> Path:
    """Run production MD with DCD trajectory output and checkpoint-resume.

    Args:
        system: Solvated OpenMM System.
        modeller: Modeller with solvated positions (for topology reference).
        ligand_id: Ligand identifier, e.g. ``"L1"``.
        receptor_type: ``"WT"`` or ``"R206H"``.
        replica_id: Replica index (1-based).
        seed: Random seed for velocity assignment (42, 43, or 44).
        cfg: Resolved config dict from ``configs/simulation.yaml``.
        scratch_dir: ORCD scratch directory root for this campaign.
        topology_pdb_path: PDB written during system build (defines atom ordering for MDTraj).
        equil_state_path: XML from equilibration stage.
        platform_name: OpenMM platform name (``"CUDA"`` on ORCD GPU nodes).
        smoke_test: If True, run only ``smoke_steps`` steps to validate the pipeline.

    Returns:
        Path to the written DCD trajectory file.
    """
    import openmm as mm  # type: ignore[import]
    import openmm.app as app  # type: ignore[import]
    import openmm.unit as unit  # type: ignore[import]

    sim_cfg = cfg["simulation"]
    n_steps = sim_cfg["smoke_steps"] if smoke_test else sim_cfg["production_steps"]
    report_interval     = sim_cfg["report_interval"]
    checkpoint_interval = sim_cfg["checkpoint_interval"]

    run_tag  = f"{ligand_id}_{receptor_type}_rep{replica_id}"
    traj_dir = scratch_dir / "trajectories"
    ckpt_dir = scratch_dir / "checkpoints"
    log_dir  = scratch_dir / "logs"

    for d in (traj_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    traj_path  = traj_dir / f"{run_tag}.dcd"
    ckpt_path  = ckpt_dir / f"{run_tag}.chk"
    state_path = ckpt_dir / f"{run_tag}_final.xml"
    run_log    = log_dir  / f"{run_tag}.log"

    # ── Idempotency: skip if DCD already has expected frames ─────────────────
    if _trajectory_is_complete(traj_path, n_steps, report_interval):
        logger.info("[%s] Trajectory already complete (%s). Skipping.", run_tag, traj_path)
        return traj_path

    # ── Register SIGTERM handler BEFORE setting up the simulation ─────────────
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGUSR1, _handle_sigterm)

    # ── Platform ──────────────────────────────────────────────────────────────
    # Fallback: CUDA → OpenCL on PTX/compute capability mismatch
    try:
        platform = mm.Platform.getPlatformByName(platform_name)
        platform_props = {"CudaPrecision": "mixed"} if platform_name == "CUDA" else {}
    except Exception as e:
        if platform_name == "CUDA" and "UNSUPPORTED_PTX" in str(e):
            logger.warning(
                "[%s] CUDA platform not available (PTX/compute capability mismatch); "
                "falling back to OpenCL.", run_tag
            )
            platform = mm.Platform.getPlatformByName("OpenCL")
            platform_props = {}
        else:
            raise

    temperature = sim_cfg["temperature_K"] * unit.kelvin
    pressure    = sim_cfg["pressure_bar"] * unit.bar
    dt          = sim_cfg["timestep_ps"] * unit.picoseconds
    friction    = sim_cfg["friction_per_ps"] / unit.picosecond

    # ── Add barostat (NPT production) ─────────────────────────────────────────
    import copy
    prod_system = copy.deepcopy(system)
    prod_system.addForce(mm.MonteCarloBarostat(pressure, temperature, 25))

    integrator = mm.LangevinMiddleIntegrator(temperature, friction, dt)
    integrator.setRandomNumberSeed(seed)

    sim = app.Simulation(modeller.topology, prod_system, integrator, platform, platform_props)

    # ── Load equilibrated state OR resume from checkpoint ─────────────────────
    if ckpt_path.exists():
        logger.info("[%s] Resuming from checkpoint %s", run_tag, ckpt_path)
        with open(ckpt_path, "rb") as fh:
            sim.context.loadCheckpoint(fh.read())
        resume_step = _count_dcd_frames(traj_path) * report_interval if traj_path.exists() else 0
        steps_done  = resume_step
    else:
        logger.info("[%s] Loading equilibration state from %s", run_tag, equil_state_path)
        with open(equil_state_path) as fh:
            state_xml = mm.XmlSerializer.deserialize(fh.read())
        sim.context.setState(state_xml)
        sim.context.setVelocitiesToTemperature(temperature, seed)
        steps_done = 0

    steps_remaining = n_steps - steps_done
    if steps_remaining <= 0:
        logger.info("[%s] No steps remaining. Trajectory complete.", run_tag)
        return traj_path

    logger.info(
        "[%s] Running production: %d / %d steps remaining (seed=%d, smoke=%s)",
        run_tag, steps_remaining, n_steps, seed, smoke_test,
    )

    # ── Reporters ──────────────────────────────────────────────────────────────
    dcd_append = traj_path.exists()
    sim.reporters.append(
        app.DCDReporter(str(traj_path), report_interval, append=dcd_append)
    )
    sim.reporters.append(
        app.StateDataReporter(
            str(run_log),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            append=dcd_append,
        )
    )
    sim.reporters.append(
        app.CheckpointReporter(str(ckpt_path), checkpoint_interval)
    )

    # ── Production loop (with SIGTERM checkpointing) ───────────────────────────
    global _SHUTDOWN_REQUESTED
    _SHUTDOWN_REQUESTED = False

    chunk = min(checkpoint_interval, steps_remaining)
    steps_run = 0

    while steps_run < steps_remaining:
        to_run = min(chunk, steps_remaining - steps_run)
        sim.step(to_run)
        steps_run += to_run

        if _SHUTDOWN_REQUESTED:
            logger.warning("[%s] SIGTERM — checkpointing and exiting.", run_tag)
            _save_final_state(sim, state_path)
            sys.exit(0)

    # ── Final state ────────────────────────────────────────────────────────────
    _save_final_state(sim, state_path)
    logger.info("[%s] Production complete. Trajectory: %s", run_tag, traj_path)
    return traj_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_final_state(sim: Any, state_path: Path) -> None:
    """Serialize final simulation state to XML."""
    import openmm as mm  # type: ignore[import]

    final_state = sim.context.getState(
        getPositions=True, getVelocities=True, enforcePeriodicBox=True
    )
    with open(state_path, "w") as fh:
        fh.write(mm.XmlSerializer.serialize(final_state))
    logger.info("Final state written: %s", state_path)


def _count_dcd_frames(dcd_path: Path) -> int:
    """Count frames in an existing DCD; returns 0 if missing or unreadable."""
    if not dcd_path.exists() or dcd_path.stat().st_size == 0:
        return 0
    try:
        import struct
        with open(dcd_path, "rb") as f:
            f.seek(8)
            n_frames = struct.unpack("i", f.read(4))[0]
        return max(0, n_frames)
    except Exception:
        return 0


def _trajectory_is_complete(traj_path: Path, n_steps: int, report_interval: int) -> bool:
    """Return True if the DCD already contains the expected number of frames."""
    expected = n_steps // report_interval
    actual   = _count_dcd_frames(traj_path)
    if actual >= expected * 0.99:  # 1% tolerance for edge-step rounding
        return True
    return False
