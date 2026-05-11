## Summary

<!-- What does this PR do? One to three bullet points. -->

- 

## Motivation

<!-- Why is this change needed? Link to the relevant issue: Closes #N -->

## Changes

<!-- List of files/modules changed and what was changed. -->

## Testing

<!-- How did you test this? Check all that apply. -->

- [ ] `pytest -m "not gpu and not vina"` passes locally
- [ ] `ruff check . && ruff format --check .` passes
- [ ] Added/updated unit tests for new behaviour
- [ ] Tested end-to-end generation: `kinetidiff-generate --n-samples 4 ...`
- [ ] Tested on GPU (if GPU-touching code was changed)

## Checklist

- [ ] No hardcoded absolute file paths (no `/home/...`, `C:\...`)
- [ ] No new `print()` calls in `src/kinetidiff/` (use `logging`)
- [ ] Type hints added on public API functions
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`
- [ ] `environment.yml` updated if a new CUDA dependency was added

## Notes for reviewers

<!-- Anything tricky or that needs special attention? -->
