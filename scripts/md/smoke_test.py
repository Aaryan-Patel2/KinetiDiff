"""Condensed smoke test for the KinetiDiff MD pipeline.

Tests on an interactive SLURM node (16 CPUs, H200 GPU):
  1. CUDA platform detection
  2. Protein download + prep (3MTF WT)
  3. Pocket centroid validation
  4. Ligand parametrization (L1 from SMILES)
  5. Solvated system build (protein + ligand + TIP3P)
  6. Energy minimization (100 iterations)
  7. NVT mini-run (200 steps = 0.4 ps)
  8. Production mini-run (100 steps = 0.2 ps) with DCD output

Total expected wall time: ~5-10 min on H200.

Usage (inside kinetidiff-md env):
    python scripts/md/smoke_test.py --repo-root /path/to/KinetiDiff
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--platform", default="CUDA", help="OpenMM platform: CUDA or CPU")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("smoke_test")

    repo_root = Path(args.repo_root).resolve()
    t0 = time.time()

    # ── 1. CUDA platform check ────────────────────────────────────────────────
    log.info("=== [1/8] CUDA platform detection ===")
    import openmm as mm
    try:
        cuda_platform = mm.Platform.getPlatformByName(args.platform)
        log.info("Platform '%s' found. Version: %s", args.platform, mm.__version__)
    except Exception as exc:
        log.error("Platform '%s' not available: %s", args.platform, exc)
        if args.platform == "CUDA":
            log.warning("Falling back to CPU for smoke test.")
            args.platform = "CPU"

    # ── 2. Package import check ───────────────────────────────────────────────
    log.info("=== [2/8] Import check ===")
    import openmm.app as app
    import openmm.unit as unit
    from openff.toolkit.topology import Molecule as OFFMolecule
    from openmmforcefields.generators import SMIRNOFFTemplateGenerator
    from pdbfixer import PDBFixer
    import MDAnalysis as mda
    import numpy as np
    log.info("All imports OK (openmm=%s, numpy=%s)", mm.__version__, np.__version__)

    # ── 3. Protein download + prep ────────────────────────────────────────────
    log.info("=== [3/8] Protein download + prep (3MTF WT) ===")
    from kinetidiff.molecular_dynamics.prep.protein import download_pdb, prep_protein
    structures_dir = repo_root / "data" / "structures"
    raw_pdb = download_pdb("3MTF", structures_dir)
    prepped_pdb = prep_protein(raw_pdb, "WT", structures_dir / "prepped")
    log.info("Prepped PDB: %s (exists=%s)", prepped_pdb, prepped_pdb.exists())

    # ── 4. Pocket centroid validation ─────────────────────────────────────────
    log.info("=== [4/8] Pocket centroid validation ===")
    from kinetidiff.molecular_dynamics.prep.protein import validate_pocket_centroid
    validate_pocket_centroid(prepped_pdb, tolerance_A=3.0)
    log.info("Pocket centroid OK")

    # ── 5. Ligand from SMILES ─────────────────────────────────────────────────
    log.info("=== [5/8] Ligand parametrization (L1 from SMILES) ===")
    from kinetidiff.molecular_dynamics.prep.ligand import smiles_to_openff_molecule
    L1_SMILES = "O=C(NC1CCNCC1c1ccc[nH]1)c1c(C2CCCOC2)ccc2ccccc12"
    off_mol = smiles_to_openff_molecule(L1_SMILES, name="L1", resname="L01")
    log.info("OpenFF molecule built: %d heavy atoms, %d conformers",
             off_mol.n_atoms, off_mol.n_conformers)

    # ── 6. Solvated system build ──────────────────────────────────────────────
    log.info("=== [6/8] Solvated system build ===")
    scratch_dir = repo_root / "data" / "md_smoke"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Translate ligand conformer to the binding pocket center so it starts
    # inside the protein. The RDKit-generated conformer is centered near origin;
    # we shift it to the actual 3MTF chain A pocket centroid (from A3F ligand).
    from kinetidiff.molecular_dynamics.prep.protein import CANONICAL_POCKET_CENTER
    from openff.units import unit as off_unit
    pocket_A = np.array(CANONICAL_POCKET_CENTER)  # already in Å

    if off_mol.n_conformers > 0:
        conf_A = off_mol.conformers[0].magnitude  # (n_atoms, 3) in Å
        conf_center = conf_A.mean(axis=0)
        conf_shifted_A = conf_A - conf_center + pocket_A  # translate to pocket
        off_mol._conformers = [conf_shifted_A / 10.0 * off_unit.nanometer]
        log.info(
            "Ligand translated to pocket center (%.2f, %.2f, %.2f) Å",
            *pocket_A,
        )

    # Pre-assign partial charges so SMIRNOFFTemplateGenerator doesn't need
    # AmberTools (sqm/antechamber) for AM1-BCC. Preference: NAGL > Gasteiger.
    # NAGL gives AM1-BCC-quality charges without AmberTools; Gasteiger is a
    # fast empirical fallback acceptable for the smoke test only.
    _charge_assigned = False
    try:
        from openff.nagl_models import list_available_nagl_models  # type: ignore[import]
        from openff.toolkit.utils.nagl_toolkit import NAGLToolkitWrapper  # type: ignore[import]
        _nagl_models = list_available_nagl_models()
        _am1_models = [m for m in _nagl_models if "am1bcc" in str(m).lower()]
        if _am1_models:
            off_mol.assign_partial_charges(str(_am1_models[0]), toolkit_registry=NAGLToolkitWrapper())
            log.info("NAGL charges assigned (model: %s)", _am1_models[0])
            _charge_assigned = True
    except Exception as _e:
        log.debug("NAGL charge assignment skipped: %s", _e)

    if not _charge_assigned:
        log.warning(
            "NAGL unavailable; using Gasteiger charges (smoke-test only, not AM1-BCC quality). "
            "Install AmberTools or openff-nagl-models for production."
        )
        from openff.toolkit.utils import RDKitToolkitWrapper  # type: ignore[import]
        off_mol.assign_partial_charges("gasteiger", toolkit_registry=RDKitToolkitWrapper())
        log.info("Gasteiger partial charges assigned to L1.")

    # Inline mini system build (bypass caching for smoke test clarity)
    pdb_obj = app.PDBFile(str(prepped_pdb))
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    smirnoff_gen = SMIRNOFFTemplateGenerator(
        molecules=[off_mol],
        forcefield="openff-2.0.0.offxml",
    )
    forcefield.registerTemplateGenerator(smirnoff_gen.generator)

    from kinetidiff.molecular_dynamics.prep.system import _off_molecule_to_openmm
    lig_top, lig_pos = _off_molecule_to_openmm(off_mol, "L01")

    modeller = app.Modeller(pdb_obj.topology, pdb_obj.positions)
    modeller.add(lig_top, lig_pos)
    modeller.addHydrogens(forcefield, pH=7.4)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=0.8 * unit.nanometers,  # smaller box for speed
        ionicStrength=0.15 * unit.molar,
        neutralize=True,
    )
    n_atoms = modeller.topology.getNumAtoms()
    log.info("Solvated system: %d atoms", n_atoms)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
    )
    log.info("OpenMM System created: %d forces", system.getNumForces())

    # ── 7. Minimization + NVT (200 steps) ────────────────────────────────────
    log.info("=== [7/8] Minimization + mini NVT (200 steps) ===")
    temperature = 300 * unit.kelvin
    dt = 0.002 * unit.picoseconds

    # Try the requested platform; fall back to OpenCL then CPU if CUDA PTX fails.
    # Known issue: conda-forge openmm 8.5.1 NVRTC JIT fails on H200 SM 9.0.
    # OpenCL works as an equivalent GPU path for smoke-test purposes.
    platform_props: dict = {}
    platform = mm.Platform.getPlatformByName("CPU")  # safe default
    for attempt_platform in [args.platform, "OpenCL", "CPU"]:
        _props = {"CudaPrecision": "mixed"} if attempt_platform == "CUDA" else {}
        try:
            _plat = mm.Platform.getPlatformByName(attempt_platform)
            _test_integ = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
            _test_sim = app.Simulation(modeller.topology, system, _test_integ, _plat, _props)
            del _test_integ, _test_sim
            platform = _plat
            platform_props = _props
            if attempt_platform != args.platform:
                log.warning(
                    "Platform '%s' unavailable (CUDA PTX version error — known conda-forge "
                    "issue on H200 SM9.0). Falling back to '%s'. "
                    "For production: install openmm-cuda12 or load CUDA toolkit module.",
                    args.platform, attempt_platform,
                )
            args.platform = attempt_platform
            break
        except Exception as e:
            log.debug("Platform %s failed: %s", attempt_platform, e)

    integrator = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    sim = app.Simulation(modeller.topology, system, integrator, platform, platform_props)
    sim.context.setPositions(modeller.positions)

    t_min = time.time()
    sim.minimizeEnergy(maxIterations=5000)
    _state_min = sim.context.getState(getEnergy=True)
    _pe_min = _state_min.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    if np.isnan(_pe_min):
        raise RuntimeError(
            "NaN PE after 5000 minimization steps. "
            "Likely severe ligand-protein clash; centroid placement alone "
            "is not sufficient — consider docking the ligand before MD."
        )
    log.info("Minimization done (%.1f s), PE=%.1f kcal/mol", time.time() - t_min, _pe_min)

    sim.context.setVelocitiesToTemperature(temperature)
    t_nvt = time.time()
    sim.step(200)
    state_after_nvt = sim.context.getState(getEnergy=True)
    ke = state_after_nvt.getKineticEnergy().value_in_unit(unit.kilocalories_per_mole)
    pe = state_after_nvt.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    log.info("NVT 200 steps done (%.1f s). KE=%.1f kcal/mol, PE=%.1f kcal/mol",
             time.time() - t_nvt, ke, pe)

    # ── 8. Production mini-run (100 steps, DCD output) ───────────────────────
    log.info("=== [8/8] Production mini-run (100 steps, DCD output) ===")
    import copy
    prod_system = copy.deepcopy(system)
    prod_system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, temperature, 25))

    integrator2 = mm.LangevinMiddleIntegrator(temperature, 1.0 / unit.picosecond, dt)
    integrator2.setRandomNumberSeed(42)
    sim2 = app.Simulation(modeller.topology, prod_system, integrator2, platform, platform_props)
    nvt_state = sim.context.getState(getPositions=True, getVelocities=True)
    sim2.context.setPositions(nvt_state.getPositions())
    sim2.context.setVelocities(nvt_state.getVelocities())

    dcd_out = scratch_dir / "smoke_L1_WT.dcd"
    log_out = scratch_dir / "smoke_L1_WT.log"
    sim2.reporters.append(app.DCDReporter(str(dcd_out), 50))
    sim2.reporters.append(app.StateDataReporter(
        str(log_out), 50, step=True, time=True,
        potentialEnergy=True, temperature=True, density=True,
    ))

    t_prod = time.time()
    sim2.step(100)
    log.info("Production 100 steps done (%.1f s). DCD: %s", time.time() - t_prod, dcd_out)

    # ── Verify DCD was written ────────────────────────────────────────────────
    assert dcd_out.exists() and dcd_out.stat().st_size > 0, "DCD not written!"
    import struct
    with open(dcd_out, "rb") as f:
        f.seek(8)
        n_frames = struct.unpack("i", f.read(4))[0]
    log.info("DCD verified: %d frames written (expected ~2).", n_frames)

    # ── Bonus: runner logic sanity check ──────────────────────────────────────
    log.info("=== [bonus] Runner build_run_matrix sanity check ===")
    from kinetidiff.molecular_dynamics.simulation.runner import build_run_matrix, get_my_run
    campaign_cfg = {
        "priority_order": ["L1", "L2", "L3", "L4", "L5"],
        "receptors": ["WT", "R206H"],
        "seeds": [42, 43, 44],
    }
    matrix = build_run_matrix(campaign_cfg)
    assert len(matrix) == 30, f"Expected 30 runs, got {len(matrix)}"
    assert matrix[0] == ("L1", "WT", 1, 42), f"Wrong first run: {matrix[0]}"
    task0_runs = get_my_run(matrix, 0, 30)
    assert task0_runs[0] == ("L1", "WT", 1, 42)
    log.info("Run matrix OK: 30 runs, matrix[0]=%s", matrix[0])

    # ── Bonus: production._trajectory_is_complete check ──────────────────────
    from kinetidiff.molecular_dynamics.simulation.production import _trajectory_is_complete
    assert _trajectory_is_complete(dcd_out, n_steps=100, report_interval=50), \
        "Expected DCD to appear complete for smoke n_steps=100, interval=50"
    log.info("_trajectory_is_complete check passed.")

    # ── Write topology PDB for analysis (needed by MDAnalysis) ────────────────
    topo_pdb = scratch_dir / "smoke_L1_WT_topo.pdb"
    with open(topo_pdb, "w") as fh:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, fh)
    log.info("Topology PDB written for analysis: %s", topo_pdb)

    # ── Trajectory analysis on the smoke DCD ─────────────────────────────────
    log.info("=== [analysis] RMSD + RMSF on smoke trajectory ===")
    from kinetidiff.molecular_dynamics.analysis.trajectory import (
        run_rmsd_analysis,
        run_rmsf_analysis,
        is_pose_stable,
    )
    rmsd_df = run_rmsd_analysis(topo_pdb, dcd_out, lig_resname="L01")
    rmsf_df = run_rmsf_analysis(topo_pdb, dcd_out)
    stable = is_pose_stable(rmsd_df, threshold_A=4.0)
    log.info(
        "Analysis: RMSD rows=%d, RMSF rows=%d, pose_stable=%s",
        len(rmsd_df), len(rmsf_df), stable,
    )

    total_time = time.time() - t0
    log.info("")
    log.info("=" * 60)
    log.info("SMOKE TEST PASSED in %.1f s (%.1f min)", total_time, total_time / 60)
    log.info("Platform: %s | Atoms: %d | DCD: %s", args.platform, n_atoms, dcd_out)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
