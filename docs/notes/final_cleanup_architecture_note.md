# Final Cleanup Architecture Note

**Date:** 2026-04-08

## Context

After Phase 5, the repo already had the intended package boundaries:

- `src/datasets/`
- `src/prompts/`
- `src/batching/`
- `src/backends/`
- `src/mechanisms/`
- `src/privacy/`
- `src/runtime/`
- `src/artifacts/`
- `src/evaluation/`

The remaining issues were not architectural gaps. They were cleanup issues:

- some legacy compatibility facades were still being used by internal scripts
- simple JSONL I/O still lived in `src/evaluate.py` instead of the artifact layer
- generation statistics still lived in `src/evaluate.py` instead of the runtime layer
- `pyproject.toml` still used an explicit package list and the legacy setuptools backend
- `docs/notes/` was ignored by `.gitignore`, so architecture and migration notes were not part of the visible repo surface
- README and notes still described the repo as if the package split were less complete
- test-count documentation had small drift in Phase 5 notes
- the role of `scripts/sweep_hyperparams.py` was unclear relative to the canonical workflow

This pass keeps the accepted Phase 1–5 architecture intact and only removes confusion around it.

## Cleanup decisions

### 1. Keep major compatibility facades, but stop using them internally where practical

Kept:

- `src/generate.py`
- `src/evaluate.py`
- `src/privacy_accounting.py`
- `src/config.compute_max_private_tokens(...)`

Reason:

- tests still protect these surfaces
- they remain plausible external entry points
- removing them now would create churn without improving the main workflow

Changed:

- internal scripts now prefer canonical package imports where the target package is already explicit
- compatibility facades remain for users, tests, and historical imports

This is the right balance for a cleanup pass: reduce internal reliance on wrappers without deleting public compatibility prematurely.

### 2. Move simple JSONL I/O to `src.artifacts`

Added canonical simple-artifact helpers:

- `src.artifacts.simple_jsonl.save_synthetic_data`
- `src.artifacts.simple_jsonl.load_synthetic_data`

Reason:

- these functions are artifact concerns, not evaluation concerns
- they already coexisted conceptually with checkpoint JSONL writing and resume logic
- moving them clarifies the distinction between:
  - crash-safe checkpoint output
  - simple one-shot JSONL output

Compatibility:

- `src.evaluate` still re-exports both names
- JSON schemas and load semantics did not change

### 3. Move generation statistics to `src.runtime`

Added:

- `src.runtime.stats.compute_generation_stats`

Reason:

- these are runtime diagnostics over generated examples
- they are not downstream evaluation metrics
- generation scripts should not need to import `src.evaluate` to summarize generation results

Compatibility:

- `src.evaluate.compute_generation_stats` remains importable as a re-export

### 4. Modernize packaging discovery

Changed `pyproject.toml` from:

- explicit package list
- legacy setuptools backend

to:

- automatic package discovery with `include = ["src", "src.*"]`
- `setuptools.build_meta`

Reason:

- the repo now has enough stable package structure that automatic discovery is clearer and less brittle
- the explicit package list would require manual edits every time a subpackage is added
- the include pattern remains conservative and does not sweep in `tests/`, `scripts/`, or notebooks

### 5. Keep `scripts/sweep_hyperparams.py`, but classify it correctly

Decision:

- kept, not deleted

Reason:

- README still referenced it
- it remains a useful research helper for optional sweeps
- it does not interfere with the canonical generation pipeline
- deleting it would be a larger workflow decision than this cleanup pass needs

What changed:

- its imports were cleaned up to use canonical package paths
- its docstring now explicitly says it is optional and non-canonical

### 6. Stop hiding `docs/notes/` from the repo surface

Changed:

- removed `docs/notes/` from `.gitignore`

Reason:

- the repo now uses architecture and migration notes as accepted reference material
- the final cleanup notes requested in this pass should not be hidden local artifacts
- leaving the directory ignored would keep documentation deliverables out of normal repo review
### 7. Preserve the canonical generation workflow exactly

The canonical workflow remains:

- `scripts/run_experiment.py` for generation
- `--resume` with the same command-line arguments to continue checkpointed runs

The accepted resume command format did not change in this pass.

## What intentionally remains as compatibility surface

Intentionally retained:

- `src/generate.py` as the generation facade and backward-compatible re-export hub
- `src/evaluate.py` as a thin compatibility facade
- `src/privacy_accounting.py` as the legacy privacy facade
- `src.config.compute_max_private_tokens(...)` with legacy positional argument order

Why they remain:

- they are still exercised by tests
- they still reflect practical public import paths
- removing them now would shift this cleanup pass into a breaking-surface phase

## Test-count documentation decision

No test-discovery bug was found in code. The discrepancy was documentation drift:

- `tests/test_evaluation.py` contains 20 tests
- `tests/test_imports.py` added 1 more evaluation-related smoke test in Phase 5
- together they explain the reported increase from 240 to 261 collected tests

The Phase 5 notes were corrected so that the per-file count and total are no longer inconsistent.
