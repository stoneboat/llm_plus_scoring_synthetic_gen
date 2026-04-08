# Phase 4 Migration Report

**Version:** 0.3.6 → 0.4.0  
**Date:** 2026-04-07

## Files Changed

### New: `src/runtime/__init__.py`
Flat public surface for the runtime package:
- `SyntheticExample`, `run_batch_generation`, `run_dataset_generation`

### New: `src/runtime/generation.py`
Extracted from `src/generate.py`:
- `SyntheticExample` dataclass (moved; re-exported from `src/generate.py`)
- `run_batch_generation` (was `_run_batch_generation` in `src/generate.py`; now public)
- `run_dataset_generation` (extracted from the body of `generate_synthetic_dataset`)

### New: `src/artifacts/__init__.py`
Flat public surface for the artifacts package.

### New: `src/artifacts/jsonl_writer.py`
Extracted from `scripts/run_experiment.py`:
- `append_line` (was `_jsonl_append_line`)
- `batch_record` (was `_batch_record`)
- `write_metadata_header` (was `_write_metadata_header`)
- `append_completed_batch` (was `_append_completed_batch`)

### New: `src/artifacts/resume.py`
Extracted from `scripts/run_experiment.py`:
- `load_resume_state` (was `_load_resume_state`)

### New: `src/artifacts/metadata.py`
Extracted and refactored from `scripts/run_experiment.py`:
- `build_run_metadata(dataset, epsilon, delta, ...)` (was `_build_run_metadata(args, report, ...)`)
  - All parameters now explicit (no `argparse.Namespace`).

### Modified: `src/generate.py`
- `SyntheticExample` replaced by import + re-export from `src.runtime.generation`
- `_run_batch_generation` replaced by alias to `run_batch_generation`
- `generate_synthetic_dataset` now delegates its core loop to
  `run_dataset_generation` after setting up backend, mechanism, padding_side,
  and resolving delta.
- All Phase 2 and Phase 3 re-exports unchanged.
- New Phase 4 re-export block added.

### Modified: `scripts/run_experiment.py`
- Removed: `_jsonl_append_line`, `_batch_record`, `_write_metadata_header`,
  `_append_completed_batch`, `_load_resume_state`, `_build_run_metadata`
- Added: `from src.artifacts import (write_metadata_header, append_completed_batch,
  load_resume_state, build_run_metadata)`
- `_resolve_output_path` and `_metadata_matches_args` retained (script-level,
  argparse-bound, not reusable enough to warrant library extraction)
- `_build_run_metadata(args, report, ...)` call-site changed to
  `build_run_metadata(dataset=args.dataset, epsilon=report["epsilon"], ...)`

### New: `tests/test_artifacts.py`
20 new tests covering jsonl_writer, resume, and metadata.

### New: `tests/test_runtime.py`
9 new tests covering SyntheticExample and run_batch_generation stopping
conditions (budget exhaustion, empty output, consecutive-no-private,
counter reset, empty token list, zero budget, remaining budget tracking).

### Modified: `src/__init__.py`
- `__version__` bumped to `"0.4.0"`.

### Modified: `pyproject.toml`
- `version` bumped to `"0.4.0"`.

### Modified: `tests/test_imports.py`
- Version assertion updated to `"0.4.0"`.
- New smoke tests added for `src.runtime` and `src.artifacts` imports.

---

## Backward Compatibility Status

| Path | Status |
|---|---|
| `from src.generate import SyntheticExample` | ✓ Unchanged (re-exported from `src.runtime.generation`) |
| `from src.generate import generate_synthetic_dataset` | ✓ Unchanged signature |
| `from src.generate import generate_batch_examples` | ✓ Unchanged |
| `from src.generate import generate_one_example` | ✓ Unchanged |
| `from src.generate import BatchDescriptor` | ✓ Unchanged |
| `from src.generate import _run_batch_generation` | ✓ Preserved as alias |
| `from src.runtime import SyntheticExample` | ✓ New canonical location |
| `from src.runtime import run_batch_generation` | ✓ New public function |
| `from src.runtime import run_dataset_generation` | ✓ New public function |
| `from src.artifacts import write_metadata_header` | ✓ New public function |
| `from src.artifacts import append_completed_batch` | ✓ New public function |
| `from src.artifacts import load_resume_state` | ✓ New public function |
| `from src.artifacts import build_run_metadata` | ✓ New public function |
| Checkpoint JSONL format (checkpoint_format=1) | ✓ Byte-identical |
| Batch ID computation | ✓ Identical SHA-256 |
| All 212 pre-Phase-4 tests | ✓ Still pass |

---

## Behavior Changes

None.  All numeric outputs, JSONL records, batch IDs, and generation statistics
are identical to Phase 3.5b.

---

## New Tests Added

28 new tests (240 total, up from 212).

---

## Remaining Limitations / Phase 5+ Concerns

1. `scripts/sweep_hyperparams.py` does not use `src/artifacts/` for its per-config
   JSONL output — it calls `save_synthetic_data` (the simpler non-checkpoint
   format from `src/evaluate.py`).  A future phase could add checkpoint support
   to the sweep runner.
2. `_metadata_matches_args` and `_resolve_output_path` remain in the script
   (argparse-bound).  If other scripts need the same logic, they warrant
   extraction to a script-utils module.
3. The `src/runtime/` package currently owns only generation.  If inference
   backends grow (batched-beam, speculative decoding), additional runtime
   modules could be added here.
