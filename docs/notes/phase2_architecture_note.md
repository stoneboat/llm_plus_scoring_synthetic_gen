# Phase 2 Architecture Note

**Date:** 2026-04-06
**Scope:** Dataset / task / prompt / batching layer extraction.

---

## 1. New Module Structure

```
src/
  batching/
    __init__.py              # public re-exports
    base.py                  # BatchDescriptor, BatchingPolicy (ABC)
    hash_label_policy.py     # assign_to_batch, partition_by_label, HashLabelBatchingPolicy

  prompts/
    __init__.py              # public re-exports
    base.py                  # PromptBuilder (ABC)
    text_classification.py   # PROMPT_TEMPLATES, _format_prompt, build_prompts,
                             #   TextClassificationPromptBuilder

  datasets/
    __init__.py              # public re-exports
    base.py                  # TaskSpec, DatasetAdapter (ABC)
    text_classification.py   # HFTextClassificationAdapter
    registry.py              # REGISTRY, DATASET_CHOICES, get_adapter()
```

### Relationship diagram

```
scripts/run_experiment.py
scripts/evaluate_downstream.py  ──► src/datasets/registry.py
scripts/sweep_hyperparams.py           │
                                       ▼
                              HFTextClassificationAdapter
                                       │ (load, num_labels, label_names)
                                       ▼
                              datasets (HuggingFace library)

src/generate.py ──► src/batching/hash_label_policy.py
                │       (assign_to_batch, partition_by_label)
                │
                └──► src/prompts/text_classification.py
                         (PROMPT_TEMPLATES, build_prompts, _format_prompt)

src/config.py ──re-exports──► src/prompts/text_classification.PROMPT_TEMPLATES
```

---

## 2. Interfaces Introduced

### `TaskSpec` (`src/datasets/base.py`)
A plain dataclass holding `num_labels: int` and `label_names: Dict[int, str]`.

**Why:** These are task-level constants that were previously scattered between
`PROMPT_TEMPLATES["labels"]` and per-script `DATASET_NUM_LABELS` dicts.
Centralizing them in `TaskSpec` gives evaluators and downstream code a single
source of truth.

### `DatasetAdapter` (`src/datasets/base.py`)
An abstract base class with one required method: `load(split, num_examples, cache_dir)
→ List[{"text": str, "label": int}]` and convenience properties `num_labels` and
`label_names` delegated from `self.task`.

**Why:** The three scripts each had their own `DATASET_HF_MAP` dict and `load_*`
helper, all with slightly different schemas (some included `num_labels` in the tuple,
one used different column semantics for the test split). A single adapter eliminates
this inconsistency and makes adding a new dataset a one-place change.

### `HFTextClassificationAdapter` (`src/datasets/text_classification.py`)
A dataclass implementing `DatasetAdapter` that wraps a HuggingFace dataset:
`hf_name`, `train_split`, `test_split`, `hf_text_column`, `hf_label_column`, `task`.

The `load()` method routes `"train"`/`"test"` to the right HF split name, handles
subsampling (seed=42 for reproducibility), and normalizes rows to `{"text", "label"}`.

**Why:** This is the only concrete adapter needed for the current five datasets.
Future datasets (tabular, multi-lingual, etc.) would add new concrete adapter classes;
the registry and scripts would not change.

### `REGISTRY` + `get_adapter()` (`src/datasets/registry.py`)
A dict mapping canonical names → adapter instances, and a `get_adapter(name)` lookup
that raises `ValueError` for unknown names.

**Why:** The single source of truth for which datasets are supported. The three scripts
previously duplicated this list in three `DATASET_HF_MAP` variables.

### `PromptBuilder` (`src/prompts/base.py`)
An ABC with `build_prompts(examples, text_column, label, tokenizer) → (List[str], str)`.

**Why:** Prompt construction was inline in `generate.py` mixed with generation logic.
The ABC creates a clean boundary: the generation loop calls `build_prompts` without
knowing how the templates are rendered.

### `TextClassificationPromptBuilder` (`src/prompts/text_classification.py`)
A concrete `PromptBuilder` that wraps the standalone `build_prompts()` function.
Exposes `label_names` and `response_prefix` as properties.

**Why:** Allows callers that want the class interface (e.g., for dependency injection
in tests or future mechanism variants) to use it, while preserving the unchanged
standalone function for code that already calls it directly.

### `BatchingPolicy` (`src/batching/base.py`)
An ABC with `partition(examples, label_column, text_column, batch_size) → Dict[...]`.

**Why:** The hash-based batching logic was inline in `generate.py`. The ABC makes the
contract explicit. Phase 3 could swap in a different policy without touching
`generate_synthetic_dataset`.

### `HashLabelBatchingPolicy` (`src/batching/hash_label_policy.py`)
The concrete `BatchingPolicy` that wraps the existing `partition_by_label()` function.

**Why:** Same as above — wraps the unchanged function so it can be used either
directly (function call) or via the policy interface (class call).

---

## 3. Current Behavior Deliberately Preserved

| Contract | Where it lives now |
|----------|--------------------|
| `assign_to_batch`: SHA-256 mod N, label-prefixed key | `src/batching/hash_label_policy.py` |
| `partition_by_label`: hash-based, label-grouped, within-batch sorted | `src/batching/hash_label_policy.py` |
| Batch ID computation in `generate_synthetic_dataset` | `src/generate.py` (unchanged) |
| `BatchDescriptor` frozen dataclass (stable batch IDs) | `src/batching/base.py` |
| `build_prompts` signature + output | `src/prompts/text_classification.py` |
| `_format_prompt` chat-template wrapping | `src/prompts/text_classification.py` |
| `PROMPT_TEMPLATES` dict content (all five datasets, all keys) | `src/prompts/text_classification.py` |
| `from src.config import PROMPT_TEMPLATES` (backward compat) | `src/config.py` re-exports |
| `from src.generate import assign_to_batch / partition_by_label / BatchDescriptor / build_prompts` | `src/generate.py` re-exports |
| Five dataset HF names, splits, column names, num_labels, label_names | `src/datasets/registry.py` |
| Normalized example shape: `{"text": str, "label": int}` | `HFTextClassificationAdapter.load()` |
| Subsampling: `shuffle(seed=42).select(range(n))` | `HFTextClassificationAdapter.load()` |
| JSONL checkpoint / resume format | `src/evaluate.py`, `scripts/run_experiment.py` — **unchanged** |
| Core generation loop (all of `_generate_single_example`, `generate_batch_examples`) | `src/generate.py` — **unchanged** |

---

## 4. Intentionally Left for Later Phases

### Phase 3 targets (in rough priority order)

1. **`Mechanism` abstraction** — extract the SVT + clipping + aggregation +
   top-k logic from `_generate_single_example`.  This is the highest-risk
   extraction and needs its own phase.

2. **`ModelBackend` abstraction** — separate `get_next_token_logits`,
   tokenizer setup, padding-side management from the generation loop.

3. **`PrivacyAccountant` cleanup** — `compute_max_private_tokens` is still
   duplicated between `src/config.py` and `src/privacy_accounting.py`.
   The version in `config.py` should eventually delegate to or be removed
   in favor of the one in `privacy_accounting.py`.

4. **`Evaluator` / `DownstreamTask`** — the BERT fine-tuning code is still
   duplicated between `src/evaluate.py` and `scripts/evaluate_downstream.py`.

5. **`ArtifactWriter`** — the JSONL checkpoint writing logic in
   `scripts/run_experiment.py` could be extracted.

6. **`ExperimentConfig`** — unified dataclass-based config to replace per-script
   argparse definitions.

7. **New datasets** — once `DatasetAdapter` is stable, non-classification
   datasets (tabular, structured) can be added by implementing the ABC.

---

## 5. How Phase 2 Prepares for Mechanism Extraction

The extraction of `build_prompts` into `src/prompts/` is a prerequisite for
`Mechanism` extraction: the mechanism will need to call a `PromptBuilder` to get
its inputs.  With the ABC in place, that dependency is now expressible without
reaching back into `generate.py`.

Similarly, `BatchingPolicy` is a prerequisite for a future `GenerationRuntime`
that orchestrates multiple mechanisms: the runtime needs to call a policy to get
batches, then hand them to a mechanism.  The ABC makes that composition possible
without coupling the runtime to the hash-label-specific implementation.

The `DatasetAdapter.label_names` property is used by both prompt construction and
evaluation, avoiding the need for either component to import `PROMPT_TEMPLATES`
directly.  This decoupling is necessary for future non-classification tasks where
prompt templates may not exist.
