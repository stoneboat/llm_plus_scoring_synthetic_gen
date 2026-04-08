# Final Cleanup Migration Report

**Date:** 2026-04-08

## Files Created

- `src/artifacts/simple_jsonl.py`
- `src/runtime/stats.py`
- `docs/notes/final_cleanup_architecture_note.md`
- `docs/notes/final_cleanup_migration_report.md`

## Files Modified

- `.gitignore`
- `src/artifacts/__init__.py`
- `src/runtime/__init__.py`
- `src/evaluation/data.py`
- `src/evaluate.py`
- `src/runtime/generation.py`
- `src/generate.py`
- `scripts/run_experiment.py`
- `scripts/sweep_hyperparams.py`
- `pyproject.toml`
- `README.md`
- `docs/notes/phase5_architecture_note.md`
- `docs/notes/phase5_migration_report.md`

## Files Deleted

None.

## Wrapper decisions

### Retained

- `src/generate.py`
- `src/evaluate.py`
- `src/privacy_accounting.py`
- `src.config.compute_max_private_tokens(...)`

Reason:

- still covered by tests
- still plausible public API surface
- removing them would create avoidable churn

### Reduced in internal usage

- `scripts/run_experiment.py` now uses canonical imports for:
  - privacy reporting
  - generation stats
  - evaluation helpers
- `scripts/sweep_hyperparams.py` now uses canonical imports for:
  - privacy helpers
  - simple artifact writing
  - generation stats
  - BERT evaluation

## Function-home cleanup

### Moved to canonical home

- `save_synthetic_data`:
  - old practical home: `src.evaluate`
  - new canonical home: `src.artifacts.simple_jsonl`
  - compatibility: still importable from `src.evaluate`

- `load_synthetic_data`:
  - old practical home: `src.evaluate`
  - new canonical home: `src.artifacts.simple_jsonl`
  - compatibility: still importable from `src.evaluate`

- `compute_generation_stats`:
  - old practical home: `src.evaluate`
  - new canonical home: `src.runtime.stats`
  - compatibility: still importable from `src.evaluate`

## Package discovery changes

`pyproject.toml` changed from:

- explicit package enumeration
- `setuptools.backends.legacy:build`

to:

- automatic discovery via:
  - `[tool.setuptools.packages.find]`
  - `include = ["src", "src.*"]`
- `setuptools.build_meta`

Rationale:

- avoids brittle manual package lists
- still keeps discovery restricted to the `src` package namespace

## Documentation visibility change

`.gitignore` was updated to stop ignoring `docs/notes/`.

Reason:

- the repo’s architecture and migration notes are now treated as part of the maintained documentation surface
- the new final-cleanup notes requested in this pass should be visible to normal repo tooling and review

## Test discovery and test-count audit

### Environment reality

`pytest` is not installed in the current shell, so live `pytest --collect-only`
could not be executed here. The command failed with:

```text
/bin/bash: line 1: pytest: command not found
```

This is an environment/tooling gap, not evidence of a repo discovery bug.

### Current test inventory

Static inventory of `tests/test_*.py` plus the single `pytest.mark.parametrize`
expansion in `tests/test_accounting.py` gives:

- 257 test functions/methods by name
- plus 4 extra collected cases from parametrization
- estimated collected total: 261

This matches the Phase 5 total.

### Actual inconsistency found

The inconsistency was in documentation, not discovery:

- `docs/notes/phase5_architecture_note.md` said `tests/test_evaluation.py` had 21 tests
- `docs/notes/phase5_migration_report.md` also said 21 tests there
- the file actually contains 20 tests
- the overall total of 261 is still consistent because Phase 5 also added 1 import smoke test in `tests/test_imports.py`

Both Phase 5 notes were corrected.

## Decision on `scripts/sweep_hyperparams.py`

Decision:

- kept

Reason:

- still referenced in the README and prior notes
- still useful as an optional research helper
- not part of the canonical checkpoint/resume workflow
- easy to keep in a low-confusion state by updating imports and documenting its non-canonical role

What changed:

- removed stale imports
- switched to canonical package imports
- clarified in the docstring that it is optional

## Behavior changes

No intended behavior changes to:

- private-prediction algorithm
- prompt strings
- batching
- privacy formulas
- privacy outputs
- checkpoint schema
- resume behavior

The main generation pipeline and `scripts/run_experiment.py --resume` workflow were preserved.

## Canonical current commands

### Main generation workflow

This remains the canonical command family:

```bash
python scripts/run_experiment.py \
  --dataset agnews \
  --epsilon 3.0 \
  --batch_size 255 \
  --clip_bound 10.0 \
  --temperature 2.0 \
  --top_k_vocab 1024 \
  --public_temperature 1.5 \
  --svt_threshold 0.5 \
  --svt_noise 0.2
```

### Resume workflow

The exact resume command style you provided is still valid. No replacement was required.

```bash
python scripts/run_experiment.py \
  --dataset agnews \
  --epsilon 3.0 \
  --delta 8.333333333333334e-06 \
  --batch_size 255 \
  --clip_bound 10.0 \
  --temperature 2.0 \
  --public_temperature 1.5 \
  --svt_threshold 0.5 \
  --svt_noise 0.2 \
  --top_k_vocab 1024 \
  --max_private_tokens 177 \
  --max_total_tokens 256 \
  --seed 42 \
  --micro_batch_size 32 \
  --output_path data/outputs/agnews_eps3.0_s255_20260404_194803.jsonl \
  --resume
```

## Low-priority items intentionally left unresolved

- `src/generate.py` remains a compatibility-heavy facade because tests still protect that surface
- `src/evaluate.py` remains as a compatibility facade for historical imports
- `src/privacy_accounting.py` remains as a compatibility facade for the legacy privacy surface
- `scripts/evaluate_downstream.py` still prepends the repo root to `sys.path`; changing script bootstrapping was out of scope for this pass
