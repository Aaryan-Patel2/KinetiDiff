# Contributing to KinetiDiff

Thank you for your interest in contributing! This guide covers the setup, workflow, and standards for contributions.

## Environment Setup

KinetiDiff uses a hybrid conda + uv packaging model: conda manages the CUDA-locked ML stack, uv manages pure-Python deps.

```bash
# 1. Clone the repo
git clone https://github.com/Aaryan-Patel2/kinetidiff.git
cd kinetidiff

# 2. Install the CUDA/ML stack via conda
conda env create -f environment.yml
conda activate kinetidiff

# 3. Install the Python package + dev tools via uv
uv sync --frozen --extra dev
pip install -e .

# 4. Install pre-commit hooks
pre-commit install

# 5. Download public data
python scripts/download_data.py

# 6. Install AutoDock Vina binary (needed for guidance tests)
bash scripts/install_vina_binary.sh
```

For MD pipeline work, also run:
```bash
conda env create -f environment-md.yml
conda activate kinetidiff-md
uv sync --frozen --extra md
```

## Running Tests

```bash
# Fast unit tests (no GPU, no Vina binary required — runs in CI)
pytest -m "not gpu and not vina and not slow"

# Tests that call the Vina binary
pytest -m vina

# Full suite (GPU required)
pytest

# With coverage
pytest --cov=kinetidiff --cov-report=term-missing
```

## Code Quality

Pre-commit runs on every commit. You can also run manually:

```bash
ruff check .            # linter
ruff format .           # formatter
mypy src/kinetidiff/    # type checker
```

## What to Work On

- Check [open issues](https://github.com/Aaryan-Patel2/kinetidiff/issues) labelled `good first issue` or `help wanted`.
- Before starting non-trivial work, open an issue or comment on an existing one so we can discuss the approach.
- For paper-related changes (architecture decisions, reported results), open a discussion first.

## Branching Model

- `main` — always releasable. CI must pass.
- `feat/<description>` — feature branches. Open a PR to merge into main.
- `fix/<description>` — bug fix branches.

## Pull Request Checklist

The PR template will prompt you, but briefly:

1. No hardcoded absolute paths (`/home/...`).
2. No `print()` in `src/kinetidiff/` — use `logging`.
3. Type hints on all public API functions.
4. Tests for new behaviour.
5. `CHANGELOG.md` updated under `## [Unreleased]`.
6. `environment.yml` updated if you added a CUDA dependency.

## Reporting Bugs

Use the [bug report template](https://github.com/Aaryan-Patel2/kinetidiff/issues/new?template=bug_report.yml). Include your environment details and a minimal reproduction.

## Security Issues

Do **not** open public issues for security vulnerabilities. See [SECURITY.md](SECURITY.md).

## License

By submitting a PR, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
