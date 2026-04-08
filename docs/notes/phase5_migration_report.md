# Phase 5 Migration Report

**Version:** 0.4.0 → 0.5.0  
**Date:** 2026-04-07

## Files Changed

### New: `src/evaluation/__init__.py`
Flat public surface: `load_test_set`, `load_real_train`, `load_synthetic_for_eval`,
`finetune_bert`, `build_icl_prompt`, `icl_evaluate`, `save_eval_results`.

### New: `src/evaluation/data.py`
Extracted from `scripts/evaluate_downstream.py`:
- `load_test_set` (was `load_test_set` in script)
- `load_real_train` (was `load_real_train` in script)
- `load_synthetic_for_eval` (new; wraps `load_synthetic_data` with deferred import)

### New: `src/evaluation/finetune.py`
Consolidated from two implementations:
- `finetune_bert` = merged `finetune_and_evaluate` (src/evaluate.py) + `finetune_bert` (script)
- Added `verbose=True` parameter; uses script-style print format.
- Same return schema: `{accuracy, macro_f1, weighted_f1, per_class}`.

### New: `src/evaluation/icl.py`
Extracted and reorganized from `scripts/evaluate_downstream.py`:
- `build_icl_prompt` (was anonymous inner function inside `icl_evaluate` in script)
- `icl_evaluate` (was `icl_evaluate` in script)

### New: `src/evaluation/results.py`
Extracted from `scripts/evaluate_downstream.py`:
- `save_eval_results` (was inlined `os.makedirs` + `json.dump` in script)

### Modified: `src/evaluate.py`
- `finetune_and_evaluate` is now a re-export alias:
  ```python
  from src.evaluation.finetune import finetune_bert as finetune_and_evaluate
  ```
- `save_synthetic_data`, `load_synthetic_data`, `compute_generation_stats`,
  `format_icl_prompt` are unchanged.
- Import of `SyntheticExample` updated: now from `src.runtime.generation`
  instead of `src.generate` (consistent with Phase 4 move).

### Modified: `scripts/evaluate_downstream.py`
Removed (replaced by `src/evaluation/` imports):
- `load_test_set`, `load_real_train`, `finetune_bert`, `icl_evaluate`
  (and the inner `build_icl_prompt`)

Added imports:
```python
from src.evaluation import (
    load_test_set, load_real_train, load_synthetic_for_eval,
    finetune_bert, icl_evaluate, save_eval_results,
)
```
Removed from `main()`: manual `os.makedirs` + `json.dump` for result saving;
replaced with `save_eval_results(results, args.output_path)`.

`_metadata_matching`, argparse, and final print formatting unchanged.

### New: `tests/test_evaluation.py`
20 new tests (see Architecture Note).

### Modified: `tests/test_imports.py`
- Version assertion updated to `"0.5.0"`.
- Added `test_import_evaluation()` smoke test.

### Modified: `src/__init__.py`
- `__version__` bumped to `"0.5.0"`.

### Modified: `pyproject.toml`
- `version` bumped to `"0.5.0"`.
- Added `"src.evaluation"` to `packages` list.

---

## Old-to-New Mapping

| Old | New canonical location |
|---|---|
| `scripts/evaluate_downstream.py::load_test_set` | `src.evaluation.data.load_test_set` |
| `scripts/evaluate_downstream.py::load_real_train` | `src.evaluation.data.load_real_train` |
| `scripts/evaluate_downstream.py::finetune_bert` | `src.evaluation.finetune.finetune_bert` |
| `scripts/evaluate_downstream.py::icl_evaluate` | `src.evaluation.icl.icl_evaluate` |
| `scripts/evaluate_downstream.py::build_icl_prompt` (anon) | `src.evaluation.icl.build_icl_prompt` |
| `src.evaluate.finetune_and_evaluate` | alias for `src.evaluation.finetune.finetune_bert` |
| (new) `load_synthetic_for_eval` | `src.evaluation.data.load_synthetic_for_eval` |
| (new) `save_eval_results` | `src.evaluation.results.save_eval_results` |

---

## Backward Compatibility Status

| Path | Status |
|---|---|
| `from src.evaluate import save_synthetic_data` | ✓ Unchanged |
| `from src.evaluate import load_synthetic_data` | ✓ Unchanged |
| `from src.evaluate import compute_generation_stats` | ✓ Unchanged |
| `from src.evaluate import format_icl_prompt` | ✓ Unchanged |
| `from src.evaluate import finetune_and_evaluate` | ✓ Now a re-export alias; same signature |
| `from src.evaluation import finetune_bert` | ✓ New canonical location |
| `from src.evaluation import icl_evaluate` | ✓ New canonical location |
| `from src.evaluation import build_icl_prompt` | ✓ New named function |
| `from src.evaluation import load_test_set` | ✓ New canonical location |
| `from src.evaluation import load_real_train` | ✓ New canonical location |
| `from src.evaluation import load_synthetic_for_eval` | ✓ New helper |
| `from src.evaluation import save_eval_results` | ✓ New helper |
| `scripts/evaluate_downstream.py` CLI arguments | ✓ Unchanged |
| `scripts/evaluate_downstream.py` result dict schema | ✓ Unchanged |
| All 240 pre-Phase-5 tests | ✓ Still pass |

---

## Behavior Changes

None.  The BERT fine-tuning print format changed slightly in the consolidated
version (``Epoch N/M`` capital-E, ``avg_loss`` as a local variable), but the
training algorithm, hyper-parameters, and return values are identical.  Callers
that capture stdout will see slightly different epoch-log formatting; callers
that use the returned dict are unaffected.

---

## New Tests Added

20 tests in `tests/test_evaluation.py` plus 1 new import smoke test in
`tests/test_imports.py`, for 21 newly collected tests total (261 total, up from 240).

---

## Remaining Technical Debt

1. `save_synthetic_data` (simple JSONL writer) lives in `src/evaluate.py` while
   the crash-safe checkpoint writer lives in `src/artifacts/`.  A future phase
   could merge or clearly document the two write paths.
2. `compute_generation_stats` is a generation diagnostic, not an evaluation
   metric.  It could move to `src/runtime/` in a future phase.
3. `scripts/sweep_hyperparams.py` still imports from `src.evaluate` directly.
   It could be updated to use `src.evaluation.finetune` and
   `src.evaluation.results` for consistency.
4. No support yet for tabular evaluation, RDP/PLD privacy evaluators, or
   retrieval-based metrics.
