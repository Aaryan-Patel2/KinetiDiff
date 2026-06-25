# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

KinetiDiff is a drug discovery pipeline targeting ACVR1/ALK2 (FOP). The core contribution is **real-time Vina gradient injection into GCDM's E(n)-equivariant denoising loop** — guiding molecule generation toward better docking poses during diffusion. Architecture details live in `.claude/ARCH.md`.

---

## Environment Setup

Two separate conda environments; do not mix them.

**Diffusion / generation stack** (PyTorch 2.0.1 + CUDA 11.8):
```bash
conda env create -f environment.yml
conda activate kinetidiff
uv sync --frozen --extra dev
pip install -e .
```

**MD validation stack** (OpenMM, OpenFF, MDAnalysis):
```bash
conda env create -f environment-md.yml
conda activate kinetidiff-md
uv sync --frozen --extra md
```

`uv.lock` is committed — always use `uv sync --frozen`, not `uv sync`, to avoid re-resolving.

---

## Common Commands

```bash
# Lint
ruff check .
ruff format --check .
mypy src/kinetidiff/ --ignore-missing-imports   # advisory; CI continues-on-error

# Tests — unit only (no GPU, no Vina binary, no network)
pytest -m "not gpu and not vina and not slow"

# Single test file
pytest tests/unit/test_gradient_guidance.py -v

# Full test suite (requires Vina on PATH and GPU)
pytest -m "gpu"          # GPU tests
pytest -m "vina"         # Vina integration tests

# CLI entry points (after pip install -e .)
kinetidiff-generate       # → kinetidiff.generation.generate_with_vina_guidance:main
kinetidiff-generate-multi # → kinetidiff.generation.generate_multi_objective:main
kinetidiff-train          # → kinetidiff.train.train_affinity:main

# GCDM training — Hydra only, no argparse
python -m kinetidiff.gcdm.train --config-name bindingmoad_ca_cond_gcpnet \
    model.guidance_scale=1.5 model.guidance_start_timestep=300

# MD smoke test (10 ps, on ORCD Engaging)
sbatch --partition=mit_quicktest -c 2 --mem=8G -t 00:15:00 \
    --wrap="conda activate kinetidiff-md && \
    python -m kinetidiff.molecular_dynamics.simulation.runner \
    --config configs/simulation.yaml --repo-root \$(pwd) --smoke-test"

# MD full campaign (chained SLURM array jobs)
EQUIL=$(sbatch scripts/md/submit_equil_array.sh | awk '{print $NF}')
PROD=$(sbatch --dependency=afterok:${EQUIL} scripts/md/submit_prod_array.sh | awk '{print $NF}')
ANA=$(sbatch --dependency=afterok:${PROD} scripts/md/submit_analysis_array.sh | awk '{print $NF}')
```

---

## Architecture Overview

### Gradient Injection (Core Contribution)

Three files implement the guidance mechanism — read `.claude/ARCH.md` before modifying any of them:

| File | Role |
|------|------|
| `gcdm/equivariant_diffusion/conditional_model.py` | Injection site — `sample_given_pocket()` |
| `gcdm/equivariant_diffusion/vina_gradient_guidance.py` | `VinaGradientGuidance` + `GradientProcessor` |
| `guidance/vina_guidance.py` | Standalone `VinaGuidance` used by CLI generation scripts |

**Sign convention**: `compute_gradient()` returns ∇(Vina score). The loop *subtracts* it to minimize (better) Vina scores. Do not negate inside `compute_gradient`.

**Zero-mean constraint**: After applying the gradient, each sample must be independently re-centered. Skipping breaks E(n)-equivariance and causes generation to drift.

**Guidance is only active** when `t < guidance_start_timestep` (default 400/1000) and at `guidance_interval` cadence (default every 20 steps).

### Module Boundaries (enforced — do not cross)

| Module | May NOT import from |
|--------|---------------------|
| `gcdm/equivariant_diffusion/` | `guidance/`, `affinity_pred/`, `molecular_dynamics/` |
| `guidance/` | `affinity_pred/`, `molecular_dynamics/` |
| `generation/` | `molecular_dynamics/`, `train/` |
| `molecular_dynamics/` | `gcdm/`, `guidance/`, `generation/`, `affinity_pred/` |

### MD Pipeline

All business logic lives in `src/kinetidiff/molecular_dynamics/{prep,simulation,analysis,viz}/`. Notebooks (`notebooks/md_validation.ipynb`) are thin orchestration only — no computation in cells. All parameters come from `configs/simulation.yaml`; nothing hardcoded.

`simulation/production.py` detects existing `.chk` files and skips completed runs — never re-run without `--force`.

---

## Key Invariants

| Invariant | Where it lives |
|-----------|---------------|
| GCDM entrypoint is Hydra-only — no argparse | `gcdm/train.py` |
| `set_seed()` is the first call in every entrypoint | `train_affinity.py`, `gcdm/train.py` |
| `OmegaConf.to_container(..., resolve=True)` before merging DictConfigs | All Hydra config merges |
| No `print()` in `src/kinetidiff/` — use `_logging.get_logger(__name__)` | Package-wide |
| Force field locked: AMBER ff14SB + OpenFF Sage 2.0 + TIP3P | `environment-md.yml`, `configs/simulation.yaml` |
| Pocket centroid for 3MTF site A: `(24.87, -12.54, 38.40)` | Import from `prep/protein.py:CANONICAL_POCKET_CENTER` |
| GCDM atom map: C=0, N=1, O=2, F=3, P=4, S=5, Cl=6, Br=7, I=8 | `vina_gradient_guidance.py:GCDM_ATOM_MAP` |
| `torch_scatter` + `torch_geometric` must match PyTorch 2.0.1 + CUDA 11.8 exactly | `environment.yml` |
| Notebooks must have outputs cleared before commit (`nbstripout`) | Pre-commit |
| No absolute local paths in any tracked file | `OPEN-SOURCE.md` |

---

## Dependency Additions

- Pure Python: add to `[project.optional-dependencies]` in `pyproject.toml`, then `uv lock`.
- CUDA/system libs (openmm, rdkit, torch-scatter): add to `environment.yml` or `environment-md.yml` with a comment explaining why.
- Never add to `src/requirements.txt` (deprecated).
- Check license compatibility with `pip-licenses` — no GPL/LGPL.

---

## Test Markers

```
slow   — tests taking >5 s
gpu    — requires CUDA GPU
vina   — requires AutoDock Vina binary on PATH
```

Unit tests in `tests/unit/` must run with no GPU, no Vina, no network. Use mock PDBs from `tests/fixtures/`.

---

## Changelog

Every PR that changes behavior needs a one-line entry in `CHANGELOG.md` under `## [Unreleased]` (Keep a Changelog format).
