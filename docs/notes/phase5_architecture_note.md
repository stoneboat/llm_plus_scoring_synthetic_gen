# Phase 5 Architecture Note

**Version:** 0.4.0 → 0.5.0  
**Date:** 2026-04-07

## Motivation

After Phase 4 the repo had explicit boundaries for generation, runtime,
artifacts, privacy, datasets, batching, prompts, backends, and mechanisms.
The remaining structural debt was the **evaluation layer**, which was split
inconsistently across two places:

| Old location | What it owned |
|---|---|
| `src/evaluate.py` | `finetune_and_evaluate` (BERT), `format_icl_prompt`, `save_synthetic_data`, `load_synthetic_data`, `compute_generation_stats` |
| `scripts/evaluate_downstream.py` | `finetune_bert` (BERT duplicate), `icl_evaluate`, `load_test_set`, `load_real_train` |

The BERT fine-tuning logic was **duplicated** — `src/evaluate.py::finetune_and_evaluate`
and `scripts/evaluate_downstream.py::finetune_bert` had identical training
loops, identical return schemas, and the same hyper-parameter defaults, with
only minor print-format differences.

The ICL evaluation logic (`icl_evaluate`, `build_icl_prompt`) lived entirely
in the script, making it unreusable and untested.

Phase 5 creates an explicit `src/evaluation/` package that consolidates all
evaluation logic and makes `scripts/evaluate_downstream.py` a thin argparse
wrapper.

---

## New Package Structure

```
src/evaluation/
├── __init__.py      # flat public surface
├── data.py          # evaluation-time data loading:
│                    #   load_test_set, load_real_train, load_synthetic_for_eval
├── finetune.py      # consolidated BERT fine-tuning evaluator:
│                    #   finetune_bert (merged from two implementations)
├── icl.py           # ICL evaluation + prompt building:
│                    #   build_icl_prompt, icl_evaluate
└── results.py       # result saving helper:
                     #   save_eval_results
```

---

## Responsibility Assignments

### `src/evaluation/data.py`

Owns evaluation-time data loading:

| Function | Purpose |
|---|---|
| `load_test_set(dataset_name, ...)` | Load real held-out test split via adapter |
| `load_real_train(dataset_name, ...)` | Load real train split for baseline comparison |
| `load_synthetic_for_eval(input_path)` | Load synthetic JSONL → `(texts, labels, label_names, metadata)` |

`load_synthetic_for_eval` delegates to `src.evaluate.load_synthetic_data`
(deferred import to avoid circular dependency), so batch-marker resume
semantics are preserved.

Dataset access always goes through `src.datasets.registry.get_adapter`.

### `src/evaluation/finetune.py`

Owns the BERT fine-tuning evaluator, consolidated from two duplicate
implementations:

| Consolidated from | Old location |
|---|---|
| `finetune_bert` | `scripts/evaluate_downstream.py` (script-private) |
| `finetune_and_evaluate` | `src/evaluate.py` (library function) |

Both had identical training loops and return schemas.  The consolidated
`finetune_bert` adds a `verbose=True` parameter (from `src/evaluate.py`)
and uses the clearer epoch-format prints (from the script version).

`src/evaluate.py::finetune_and_evaluate` is now a direct alias:
```python
from src.evaluation.finetune import finetune_bert as finetune_and_evaluate
```

### `src/evaluation/icl.py`

Owns ICL evaluation:

| Function | Purpose |
|---|---|
| `build_icl_prompt(synthetic_texts, synthetic_labels, dataset_name, test_text, num_shots)` | Build a balanced few-shot ICL prompt |
| `icl_evaluate(...)` | Run full ICL evaluation loop with a local LM |

`build_icl_prompt` was previously an anonymous inner function inside
`icl_evaluate` in the script.  It is now a named, importable, testable
function with the same logic.

**Note on prompt-format distinction:**  Two ICL formatters coexist by design:
- `src/evaluate.py::format_icl_prompt` — simple format, "Answer:" marker
- `src/evaluation/icl.py::build_icl_prompt` — balanced per-label format, "Category:" marker

Both are preserved.  They are not duplicates — they produce structurally
different prompts and serve different call sites.

### `src/evaluation/results.py`

Owns evaluation result persistence:

| Function | Purpose |
|---|---|
| `save_eval_results(results, output_path)` | Write evaluation dict to JSON with directory creation |

Callers no longer need to inline `os.makedirs` + `json.dump`.

---

## What Stays in `src/evaluate.py`

| Function | Reason kept |
|---|---|
| `save_synthetic_data` | Tested in `test_jsonl_schema.py`; used by sweep script; sweep-specific JSONL format (no batch markers, no fsync) |
| `load_synthetic_data` | Tested in `test_jsonl_schema.py`; imported by `test_artifacts.py` fixture writer; canonical load path for both simple and checkpoint formats |
| `compute_generation_stats` | Generation diagnostic (token counts), not downstream evaluation metric; used by post-generation scripts |
| `format_icl_prompt` | Tested in `test_imports.py`; different prompt format from `build_icl_prompt`; kept for backward compat |
| `finetune_and_evaluate` | Tested in `test_imports.py`; now a re-export alias of `finetune_bert` |

---

## Interaction with Other Packages

```
scripts/evaluate_downstream.py
  └── src.evaluation (load_test_set, load_real_train, load_synthetic_for_eval,
                       finetune_bert, icl_evaluate, save_eval_results)

src.evaluation.data
  ├── src.datasets.registry (get_adapter)
  └── src.evaluate.load_synthetic_data (deferred import)

src.evaluation.icl
  └── src.prompts.PROMPT_TEMPLATES (label-name lookup)

src.evaluate
  └── src.evaluation.finetune.finetune_bert (re-exported as finetune_and_evaluate)

scripts/run_experiment.py
  └── src.evaluate (compute_generation_stats, finetune_and_evaluate)

scripts/sweep_hyperparams.py
  └── src.evaluate (save_synthetic_data, compute_generation_stats,
                     finetune_and_evaluate)
```

No circular dependencies (the `data.py` → `src.evaluate` import is deferred
to function body to break the initialization-time cycle).

---

## What Did Not Change

1. **Algorithm behavior**: unchanged.
2. **JSONL schemas**: unchanged.
3. **Batch IDs**: unchanged.
4. **`src/evaluate.py` public API**: all five previously-importable names
   (`save_synthetic_data`, `load_synthetic_data`, `compute_generation_stats`,
   `format_icl_prompt`, `finetune_and_evaluate`) remain importable from
   `src.evaluate`.
5. **Script behavior**: `scripts/evaluate_downstream.py` produces identical
   output, results dict, and JSON schema.  Argument names unchanged.
6. **All 240 pre-Phase-5 tests**: still pass.

---

## New Tests

| File | Count | Coverage |
|---|---|---|
| `tests/test_evaluation.py` | 20 | imports (3), data.load_synthetic_for_eval (4), icl.build_icl_prompt (5), prompt-format distinction (2), results.save_eval_results (3), compute_generation_stats regression (3) |
| `tests/test_imports.py` | +1 | evaluation package smoke test |

Total: **261 tests** (up from 240).

---

## Deferred for Future Phases

1. `save_synthetic_data` could eventually move to `src/artifacts/` — it is
   a simpler (non-checkpoint) JSONL writer.  Held back because it is widely
   tested and there is no pressing need to move it.
2. Sweep-script evaluation (`scripts/sweep_hyperparams.py`) still calls
   `finetune_and_evaluate` directly from `src.evaluate`; it could be updated
   to use the new evaluation package if a sweep-level evaluation abstraction
   becomes warranted.
3. No support yet for: tabular evaluation, RDP/PLD privacy evaluators,
   multi-model ICL ensemble, retrieval-based metrics.
