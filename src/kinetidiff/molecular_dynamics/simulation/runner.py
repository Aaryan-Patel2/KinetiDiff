"""SLURM-aware entry point: maps SLURM_ARRAY_TASK_ID → one (lead, receptor, replica).

Execution model:
- The job array has 30 tasks (5 leads × 2 receptors × 3 replicas).
- Each task calls ``python -m kinetidiff.molecular_dynamics.simulation.runner``
  inside the SLURM job; it reads its task ID, resolves the (L, R, rep) triple,
  runs equilibration + production, and exits.
- All paths and config are read from ``configs/simulation.yaml``.
- No SLURM commands are invoked from inside the script (those live in
  ``scripts/md/submit_prod_array.sh``, managed by ``engagebot-hpc``).

Usage (inside SLURM batch script):
    python -m kinetidiff.molecular_dynamics.simulation.runner \\
        --config configs/simulation.yaml \\
        --repo-root /home/$USER/KinetiDiff \\
        [--smoke-test]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from kinetidiff._logging import get_logger

logger = get_logger(__name__)


def build_run_matrix(campaign: dict) -> list[tuple[str, str, int, int]]:
    """Build the ordered list of (ligand_id, receptor, replica_id, seed) tuples.

    Two modes depending on whether ``campaign`` contains a ``complexes`` key:

    **Explicit-pairs mode** (``complexes`` present):
        Each entry in ``complexes`` is a ``[ligand_id, receptor]`` pair.
        The matrix is built as ``complexes × seeds``.
        Use this for focused campaigns (e.g. protonated forms of specific hits)
        where you do *not* want the full cross-product of all ligands × receptors.

    **Cross-product mode** (no ``complexes`` key — default, backward-compatible):
        Ordering is ``priority_order × receptors × replicas`` (cyclic, preserving
        higher-priority leads at lower SLURM task IDs so they are scheduled first).

    Args:
        campaign: The ``campaign`` block from ``configs/simulation.yaml``.

    Returns:
        List of ``(lig_id, receptor, replica_id, seed)`` tuples in priority order.
    """
    matrix = []
    pairs = campaign.get("complexes")  # explicit-pairs mode (optional)
    if pairs:
        for lig_id, receptor in pairs:
            for rep_idx, seed in enumerate(campaign["seeds"], start=1):
                matrix.append((lig_id, receptor, rep_idx, seed))
        return matrix
    # Default: full cross-product over priority_order × receptors × seeds
    for lig_id in campaign["priority_order"]:
        for receptor in campaign["receptors"]:
            for rep_idx, seed in enumerate(campaign["seeds"], start=1):
                matrix.append((lig_id, receptor, rep_idx, seed))
    return matrix


def get_my_run(
    matrix: list[tuple[str, str, int, int]],
    task_id: int,
    num_tasks: int,
) -> tuple[str, str, int, int]:
    """Cyclic work assignment: task *task_id* owns ``matrix[task_id::num_tasks]``.

    For a single-task-per-run array (``num_tasks == len(matrix)``), this is
    a direct index lookup.  For smaller arrays, each task handles multiple runs
    in sequence.

    Args:
        matrix: Output of ``build_run_matrix``.
        task_id: ``SLURM_ARRAY_TASK_ID`` (0-based).
        num_tasks: ``SLURM_ARRAY_TASK_COUNT``.

    Returns:
        The ``(lig_id, receptor, replica_id, seed)`` tuple for this task.
        If the task owns multiple runs the *first* is returned; callers should
        iterate ``matrix[task_id::num_tasks]`` to process all.
    """
    my_runs = matrix[task_id::num_tasks]
    if not my_runs:
        raise IndexError(f"task_id={task_id} has no runs in matrix of size {len(matrix)}")
    return my_runs  # type: ignore[return-value]


def _get_git_hash() -> str:
    """Return the current git commit hash for provenance logging."""
    try:
        return subprocess.check_output(
            ["git", "log", "--oneline", "-1"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    """Entry point — called by SLURM batch script via ``python -m ...runner``."""
    # ── Parse args ────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="KinetiDiff MD production runner")
    parser.add_argument("--config",    required=True,  help="Path to configs/simulation.yaml")
    parser.add_argument("--repo-root", required=True,  help="Absolute path to repo root")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only smoke_steps steps to validate pipeline")
    parser.add_argument("--equil-only", action="store_true",
                        help="Run prep + equilibration only; skip production")
    args = parser.parse_args()

    # ── SLURM environment (shim: defaults for local runs) ─────────────────────
    task_id   = int(os.environ.get("SLURM_ARRAY_TASK_ID",   0))
    num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    n_cpus    = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    # ── Logging ───────────────────────────────────────────────────────────────
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s | job={job_id} task={task_id} | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    logger.info("Git: %s", _get_git_hash())
    logger.info("SLURM: job=%s task=%d/%d cpus=%d", job_id, task_id, num_tasks, n_cpus)

    # ── Load config ───────────────────────────────────────────────────────────
    from omegaconf import OmegaConf  # type: ignore[import]
    repo_root = Path(args.repo_root).resolve()
    cfg = OmegaConf.load(args.config)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # ── Resolve storage paths ─────────────────────────────────────────────────
    scratch_root = Path(
        os.environ.get("ORCD_SCRATCH", Path.home() / "orcd" / "scratch")
    ) / "kinetidiff-md"
    structures_dir = repo_root / cfg_dict["paths"]["structures_dir"]
    sdf_input      = repo_root / cfg_dict["paths"]["sdf_input"]

    # ── Build run matrix and select this task's runs ──────────────────────────
    matrix = build_run_matrix(cfg_dict["campaign"])
    my_runs = matrix[task_id::num_tasks]
    logger.info("Task %d/%d is responsible for %d run(s).", task_id, num_tasks, len(my_runs))

    # ── Lazy imports (heavy — only after config is loaded) ────────────────────
    from kinetidiff.molecular_dynamics.prep.protein import (
        download_pdb, prep_protein, validate_pocket_centroid,
    )
    from kinetidiff.molecular_dynamics.prep.ligand import load_sdf_molecules
    from kinetidiff.molecular_dynamics.prep.system import build_solvated_system
    from kinetidiff.molecular_dynamics.simulation.equilibrate import run_equilibration
    from kinetidiff.molecular_dynamics.simulation.production import run_production

    # ── One-time: download + prep required protein variants ───────────────────
    # Derive unique receptors from the run matrix (supports both cross-product
    # configs with a top-level `receptors` key and focused `complexes` configs
    # that may omit `receptors`).
    raw_pdb = download_pdb("3MTF", structures_dir)
    required_receptors = list(dict.fromkeys(rec for _, rec, _, _ in matrix))
    protein_paths: dict[str, Path] = {}
    for receptor in required_receptors:
        protein_paths[receptor] = prep_protein(raw_pdb, receptor, structures_dir / "prepped")
        validate_pocket_centroid(
            protein_paths[receptor],
            tolerance_A=cfg_dict["pocket"]["centroid_tolerance_A"],
        )

    # ── Load docked ligand poses ──────────────────────────────────────────────
    off_molecules = load_sdf_molecules(sdf_input, cfg_dict["campaign"]["ligands"])

    # ── Per-run loop ──────────────────────────────────────────────────────────
    systems_dir = scratch_root / "systems"
    equil_dir   = scratch_root / "equil_states"

    for (lig_id, receptor, replica_id, seed) in my_runs:
        run_tag = f"{lig_id}_{receptor}_rep{replica_id}"
        logger.info("== Starting run: %s ==", run_tag)

        # Build solvated system (idempotent — cached by tag)
        system, modeller, topology_pdb = build_solvated_system(
            protein_pdb_path=protein_paths[receptor],
            off_molecule=off_molecules[lig_id],
            ligand_id=lig_id,
            receptor_type=receptor,
            cfg=cfg_dict,
            out_dir=systems_dir,
        )

        # Platform: honour OPENMM_DEFAULT_PLATFORM env var (set by sbatch scripts).
        # Defaults to OpenCL (GPU; RTX Pro 6000 Blackwell requires it over CUDA 11.8).
        # Use CPU when running on CPU-only nodes (e.g. mit_quicktest smoke tests).
        platform_name = os.environ.get("OPENMM_DEFAULT_PLATFORM", "OpenCL")

        # Equilibration (idempotent)
        equil_state = run_equilibration(
            system=system,
            modeller=modeller,
            ligand_id=lig_id,
            receptor_type=receptor,
            cfg=cfg_dict,
            out_dir=equil_dir,
            platform_name=platform_name,
        )

        if args.equil_only:
            logger.info("== Equil-only mode: skipping production for %s ==", run_tag)
            continue

        # Production MD
        run_production(
            system=system,
            modeller=modeller,
            ligand_id=lig_id,
            receptor_type=receptor,
            replica_id=replica_id,
            seed=seed,
            cfg=cfg_dict,
            scratch_dir=scratch_root,
            topology_pdb_path=topology_pdb,
            equil_state_path=equil_state,
            smoke_test=args.smoke_test,
            platform_name=platform_name,
        )
        logger.info("== Completed run: %s ==", run_tag)


if __name__ == "__main__":
    main()
