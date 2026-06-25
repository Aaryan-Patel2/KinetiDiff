# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed
- `analysis/trajectory.py` — replaced all MDAnalysis frame loops with vectorized MDTraj calls; fixes all-NaN ACVR1 structural analysis (MDAnalysis couldn't parse resids from 138k-atom PDB hybrid36 atom serial), wrong RMSF residue labels, and failed H-bond analysis (no-charges error). ACVR1 canonical offset now auto-detected from DLG motif; αC-Glu corrected from canonical 282 (ILE in 3MTF) to 248, hinge Glu from 288 (MET) to 287. CLI `__main__` now loads the trajectory once and passes it to all 5 analysis functions via `_traj` parameter, reducing peak RSS from ~34 GB (5 serial loads accumulate in Python allocator) to ~20 GB; RMSD superposition uses `atom_slice` to avoid mutating the shared object.
- `analysis/mmgbsa.py` — removed OBC2 `GBSAOBCForce` that produced ΔG_bind of 10^7–10^11 kcal/mol (GB self-energies too large for double-precision cancellation); now computes vacuum protein–ligand interaction energy (ΔE = E_complex − E_protein − E_ligand, AMBER14+SMIRNOFF NoCutoff) giving numerically stable O(−100) kcal/mol values for relative ΔΔG comparisons.
- `CANONICAL_POCKET_CENTER` corrected from `(24.87, -12.54, 38.40)` to `(-17.66, -13.65, 38.41)` — prior x-coordinate was wrong by 42.53 Å (off by one structural alignment); pocket centroid validation now passes at 1.82 Å deviation.
- `pyproject.toml` `requires-python` widened from `<3.11` to `<3.13` — was blocking install on the cluster's Python 3.11.
- `scripts/md/submit_equil_array.sh` had `--smoke-test` hardcoded in the runner call; removed for production.
- `prep/system.py` `build_solvated_system` now pre-assigns NAGL or Gasteiger charges before `SMIRNOFFTemplateGenerator` so it does not silently fail when AmberTools/sqm is absent; hard error if neither NAGL nor `allow_gasteiger_charges=true` is set.
- `scripts/md/smoke_test.py` platform-fallback loop no longer passes `CudaPrecision` property to OpenCL when CUDA fails (was raising "Illegal property name").

### Added
- `scripts/md/smoke_test.py` — 8-stage + analysis smoke test verified on ORCD H200 interactive node (72 s, 128 939-atom system, OpenCL fallback, 2 DCD frames written, RMSD/RMSF analysis passes).
- `configs/simulation.yaml` `forcefield.allow_gasteiger_charges` guard flag (default false) — must be set true explicitly for smoke/dev runs, never for production.
- Full modular MD validation pipeline: `src/kinetidiff/molecular_dynamics/{prep,simulation,analysis,viz}/` refactored from monolithic Colab notebook.
- `configs/simulation.yaml` — single YAML source of truth for force-field stack, pocket geometry, campaign matrix (5 leads × WT + R206H × 3 replicas × 100 ns = 30 runs).
- `environment-md.yml` — pinned conda environment for the MD stack (OpenMM, OpenFF Sage 2.0, MDAnalysis, MDTraj, PDBFixer).
- `scripts/md/` — SLURM job scripts for equilibration array, production array (mit_preemptable + --requeue), analysis array, and Drive-bundle assembly.
- `notebooks/md_validation.ipynb` — thin orchestration notebook (no business logic in cells).
- `logs/md/` directory tracked for SLURM log output (contents gitignored).
- README section: "MD Validation on ORCD Engaging" with how-to-run guide.
