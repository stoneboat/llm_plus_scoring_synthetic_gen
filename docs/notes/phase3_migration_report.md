# Phase 3 Migration Report

**Version:** 0.2.0 → 0.3.0  
**Date:** 2026-04-06

## Summary

Phase 3 extracted two explicit boundaries from `src/generate.py` into
dedicated sub-packages, and eliminated a duplicate function in `src/config.py`.
No algorithm was changed; all external behavior is identical.

---

## Files Created

### `src/backends/base.py`
`ModelBackend` abstract base class.  Defines the interface between the
generation orchestrator and the underlying LLM framework.

### `src/backends/huggingface_causal_lm.py`
`HuggingFaceCausalLM` — concrete implementation wrapping a HuggingFace
`AutoModelForCausalLM` + `AutoTokenizer` pair.  Migrated from the
`get_next_token_logits()` free function that was defined inline in
`src/generate.py`.

Key responsibilities now owned by this class:
- Left-padding enforcement (`tokenizer.padding_side = "left"`)
- Micro-batch splitting to avoid GPU OOM
- Appending already-generated token IDs to each prompt input
- Decoding token ID lists back to strings

### `src/backends/__init__.py`
Re-exports `ModelBackend`, `HuggingFaceCausalLM`.

### `src/mechanisms/base.py`
`Mechanism` abstract base class.  Defines `generate_example(...)` interface.

### `src/mechanisms/private_prediction.py`
`PrivatePredictionMechanism` — implements Algorithm 1's inner loop (Amin et
al. 2024).  Migrated from `_generate_single_example()` in `src/generate.py`.

Also contains `_apply_top_k_filter()` (moved from `src/generate.py`).

### `src/mechanisms/__init__.py`
Re-exports `Mechanism`, `PrivatePredictionMechanism`.

### `tests/test_backends.py` (12 tests)
Guards the backend layer contract.

### `tests/test_mechanism.py` (14 tests)
Guards the mechanism layer contract.

### `tests/test_accounting.py` (13 tests)
Guards parity between `src/config.compute_max_private_tokens` (wrapper) and
`src/privacy_accounting.compute_max_private_tokens` (authoritative).

---

## Files Modified

### `src/generate.py`
- Added imports from `src/backends/` and `src/mechanisms/`.
- `get_next_token_logits()` → backward-compat wrapper (creates a
  `HuggingFaceCausalLM` and delegates).
- `_generate_single_example()` → backward-compat wrapper (creates backend +
  mechanism and delegates).
- `generate_batch_examples()` → creates backend + mechanism, calls new
  `_run_batch_generation()` helper.
- `generate_synthetic_dataset()` → creates `HuggingFaceCausalLM` +
  `PrivatePredictionMechanism` once at the top; padding-side management
  delegated to `backend.padding_side`; decoding delegated to `backend.decode`.
- Added `_run_batch_generation()` internal helper (the old outer loop from
  `generate_batch_examples`, now mechanism-agnostic).

### `src/config.py`
- Replaced the duplicate `compute_max_private_tokens` body with a delegation
  wrapper to `src/privacy_accounting.compute_max_private_tokens`.
- Removed `import math` (no longer needed after body removal).
- The original argument order `(target_epsilon, delta, batch_size, clip_bound,
  temperature, svt_noise)` is preserved for backward compatibility.

### `src/__init__.py`
- `__version__` bumped to `"0.3.0"`.
- Docstring updated to mention Phase 3.

### `pyproject.toml`
- `version` bumped to `"0.3.0"`.

### `tests/test_imports.py`
- `test_src_package_version`: updated assertion to `"0.3.0"`.

---

## Backward Compatibility

All symbols importable in 0.2.0 remain importable in 0.3.0:

```python
from src.generate import (
    SyntheticExample, BatchDescriptor, partition_by_label,
    assign_to_batch, build_prompts, _format_prompt, _apply_top_k_filter,
    generate_synthetic_dataset, generate_batch_examples,
    generate_one_example, get_next_token_logits, _generate_single_example,
)
from src.config import (
    PrivacyConfig, GenerationConfig, ModelConfig, DatasetConfig,
    PROMPT_TEMPLATES, HYPERPARAM_GRID, SVT_SETTINGS,
    compute_max_private_tokens,
)
```

---

## Test Results

```
143 passed in 2.6s
```
