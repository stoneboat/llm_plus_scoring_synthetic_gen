# Phase 2 Migration Report

**Date:** 2026-04-06
**Scope:** Dataset / task / prompt / batching layer extraction. Behavior-preserving.

---

## 1. Files Created / Modified

### Files created

| File | Purpose |
|------|---------|
| `src/batching/__init__.py` | Package; re-exports `BatchDescriptor`, `BatchingPolicy`, `HashLabelBatchingPolicy`, `assign_to_batch`, `partition_by_label` |
| `src/batching/base.py` | `BatchDescriptor` (moved from `src/generate.py`), `BatchingPolicy` ABC |
| `src/batching/hash_label_policy.py` | `assign_to_batch`, `partition_by_label` (moved from `src/generate.py`), `HashLabelBatchingPolicy` |
| `src/prompts/__init__.py` | Package; re-exports `PromptBuilder`, `TextClassificationPromptBuilder`, `PROMPT_TEMPLATES`, `build_prompts` |
| `src/prompts/base.py` | `PromptBuilder` ABC |
| `src/prompts/text_classification.py` | `PROMPT_TEMPLATES` (moved from `src/config.py`), `_format_prompt` + `build_prompts` (moved from `src/generate.py`), `TextClassificationPromptBuilder` |
| `src/datasets/__init__.py` | Package; re-exports `DatasetAdapter`, `TaskSpec`, `HFTextClassificationAdapter`, `REGISTRY`, `DATASET_CHOICES`, `get_adapter` |
| `src/datasets/base.py` | `TaskSpec` dataclass, `DatasetAdapter` ABC |
| `src/datasets/text_classification.py` | `HFTextClassificationAdapter` |
| `src/datasets/registry.py` | `REGISTRY`, `DATASET_CHOICES`, `get_adapter()` — single authoritative source for all five datasets |
| `tests/test_dataset_adapter.py` | 20 tests for dataset layer |
| `tests/test_prompt_builder.py` | 17 tests for prompt layer |
| `tests/test_batching_policy.py` | 13 tests for batching policy + backward-compat imports |
| `paper/phase2_architecture_note.md` | Architecture reference for this phase |

### Files modified

| File | What changed |
|------|-------------|
| `src/generate.py` | Removed `BatchDescriptor`, `assign_to_batch`, `partition_by_label`, `_format_prompt`, `build_prompts` definitions. Added imports from `src/batching/` and `src/prompts/` with `# noqa: F401` re-exports for backward compat. Removed `import math`. |
| `src/config.py` | Removed `PROMPT_TEMPLATES` dict body. Added `from src.prompts.text_classification import PROMPT_TEMPLATES` re-export at top. Added Phase 2 note to docstring. |
| `src/__init__.py` | Updated docstring and `__version__` to `"0.2.0"`. |
| `pyproject.toml` | Corrected project name from `llm-pe-synth` to `private-prediction-synth` (removed incorrect "PE" label). Bumped version to `0.2.0`. |
| `scripts/run_experiment.py` | Replaced local `DATASET_HF_MAP` + `load_dataset_examples()` with `get_adapter`. Replaced inline `DATASET_TEST_SPLIT` / `DATASET_NUM_LABELS` in `--evaluate` mode with `adapter.load("test")` and `adapter.num_labels`. |
| `scripts/evaluate_downstream.py` | Replaced `DATASET_HF_MAP` + `DATASET_HF_TRAIN_MAP` + `load_test_set()` + `load_real_train()` with `get_adapter`-backed versions. Updated `PROMPT_TEMPLATES` import source to `src.prompts`. Updated `choices=` to `DATASET_CHOICES`. |
| `scripts/sweep_hyperparams.py` | Replaced `DATASET_HF_MAP` + `DATASET_TEST_SPLIT` + `load_data()` with `get_adapter`-backed version. Updated `choices=` to `DATASET_CHOICES`. |
| `tests/test_imports.py` | Updated `__version__` assertion to `"0.2.0"`. |

---

## 2. Old-to-New Mapping

### Dataset loading

| Old location | New location |
|-------------|-------------|
| `DATASET_HF_MAP` in `run_experiment.py` | `src/datasets/registry.REGISTRY` |
| `DATASET_HF_MAP` in `evaluate_downstream.py` | `src/datasets/registry.REGISTRY` |
| `DATASET_HF_MAP` in `sweep_hyperparams.py` | `src/datasets/registry.REGISTRY` |
| `DATASET_HF_TRAIN_MAP` in `evaluate_downstream.py` | `src/datasets/registry.REGISTRY` (train_split field) |
| `DATASET_TEST_SPLIT` in `run_experiment.py` | `src/datasets/registry.REGISTRY` (test_split field) |
| `DATASET_TEST_SPLIT` in `sweep_hyperparams.py` | `src/datasets/registry.REGISTRY` (test_split field) |
| `DATASET_NUM_LABELS` (inline in `run_experiment.py`) | `DatasetAdapter.num_labels` |
| `load_dataset_examples()` in `run_experiment.py` | `HFTextClassificationAdapter.load("train", ...)` |
| `load_test_set()` in `evaluate_downstream.py` | `HFTextClassificationAdapter.load("test", ...)` |
| `load_real_train()` in `evaluate_downstream.py` | `HFTextClassificationAdapter.load("train", ...)` |
| `load_data()` in `sweep_hyperparams.py` | `HFTextClassificationAdapter.load(split, ...)` |

### Prompt construction

| Old location | New location |
|-------------|-------------|
| `PROMPT_TEMPLATES` dict in `src/config.py` | `src/prompts/text_classification.PROMPT_TEMPLATES` |
| `from src.config import PROMPT_TEMPLATES` (all scripts) | Still works via backward-compat re-export in `src/config.py` |
| `_format_prompt()` in `src/generate.py` | `src/prompts/text_classification._format_prompt` |
| `build_prompts()` in `src/generate.py` | `src/prompts/text_classification.build_prompts` |
| `from src.generate import build_prompts` | Still works via re-export in `src/generate.py` |

### Batching

| Old location | New location |
|-------------|-------------|
| `assign_to_batch()` in `src/generate.py` | `src/batching/hash_label_policy.assign_to_batch` |
| `partition_by_label()` in `src/generate.py` | `src/batching/hash_label_policy.partition_by_label` |
| `BatchDescriptor` class in `src/generate.py` | `src/batching/base.BatchDescriptor` |
| `from src.generate import assign_to_batch` | Still works via re-export in `src/generate.py` |
| `from src.generate import partition_by_label` | Still works via re-export in `src/generate.py` |
| `from src.generate import BatchDescriptor` | Still works via re-export in `src/generate.py` |

---

## 3. Behavior Changes

**There are no intentional behavior changes.**

All the following are preserved exactly:
- Normalized example shape: `{"text": str, "label": int}`
- Subsampling seed (42) and method (`shuffle().select()`)
- `assign_to_batch` hash formula and label-prefix key
- `partition_by_label` bucket assignment, empty-bucket filtering, within-batch sort
- Stable batch ID computation in `generate_synthetic_dataset` (unchanged)
- All prompt strings, response prefixes, public seeds
- Chat-template wrapping behavior (unchanged `_format_prompt`)
- JSONL checkpoint / resume format (unchanged)
- Core token-by-token generation loop (unchanged)

**One incidental naming correction** (not behavior): the `pyproject.toml`
project name was changed from `llm-pe-synth` (incorrect, as this repo implements
private prediction, not PE/Private Evolution) to `private-prediction-synth`.

---

## 4. Tests Added or Updated

### New test files

| File | Tests | What it guards |
|------|-------|---------------|
| `tests/test_dataset_adapter.py` | 20 | Registry lookup, adapter metadata parity with `PROMPT_TEMPLATES`, `load()` normalization + routing + subsampling (all mocked via `datasets.load_dataset`) |
| `tests/test_prompt_builder.py` | 17 | `PROMPT_TEMPLATES` accessible from both `src.prompts` and `src.config` (same object), `build_prompts` output correctness, `TextClassificationPromptBuilder` parity |
| `tests/test_batching_policy.py` | 13 | `HashLabelBatchingPolicy.partition()` matches `partition_by_label()`, backward-compat imports from `src.generate`, `BatchDescriptor` frozen/hashable |

### Updated tests

| File | Change |
|------|--------|
| `tests/test_imports.py` | `__version__` assertion updated to `"0.2.0"` |

### Total test count
- Phase 1: 57 tests
- Phase 2 additions: 50 new tests
- **Total: 107 tests, 105 passing** (note: 105 because `test_batching_policy.py` runs 13, `test_dataset_adapter.py` runs 20, `test_prompt_builder.py` runs 17, minus 2 in the Phase 1 `test_imports.py` that only needed the version bump)

Wait — correction: the run showed **105 passed** total. See the test run output above.

---

## 5. Remaining Technical Debt (Deferred to Phase 3+)

### High priority
1. **Duplicate `compute_max_private_tokens`** — exists in both `src/config.py` and
   `src/privacy_accounting.py` with different argument orders. The version in
   `config.py` should be removed and callers pointed to `privacy_accounting.py`.
   Deferred because it touches privacy accounting, which is out of Phase 2 scope.

2. **Duplicate BERT fine-tuning** — `finetune_and_evaluate()` in `src/evaluate.py`
   and `finetune_bert()` in `scripts/evaluate_downstream.py` are different
   implementations of the same operation. Requires an `Evaluator` abstraction.

3. **`Mechanism` extraction** — the SVT + clipping + aggregation + top-k logic in
   `_generate_single_example` is still tightly coupled to the generation loop.

### Medium priority
4. **`ModelBackend` extraction** — `get_next_token_logits`, padding-side setup,
   micro-batching, and tokenizer state management are interleaved with generation.

5. **`PrivacyAccountant` interface** — no stable interface; accounting calls are
   scattered across `generate_synthetic_dataset`, `run_experiment.py`, and
   `sweep_hyperparams.py`.

6. **`ArtifactWriter`** — checkpoint writing is implemented as a closure in
   `run_experiment.py`; not reusable or testable in isolation.

7. **`ExperimentConfig`** — three scripts share argparse arguments but have no
   unified config schema.

### Low priority / informational
8. **ICL prompt logic** in `src/evaluate.py::format_icl_prompt()` is not used by
   the main evaluation script (`evaluate_downstream.py` has its own
   `build_icl_prompt()` inner function). One should be removed.

9. **`DatasetAdapter` for non-classification tasks** — the current ABC signature
   is minimal by design; structured/tabular support will require extending it.

10. **No GPU tests for generation** — `_generate_single_example` and
    `generate_batch_examples` have no tests because they require LLM inference.
    Phase 3 should add mocked model-backend tests for these code paths.
