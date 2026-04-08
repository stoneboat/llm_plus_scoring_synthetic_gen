# Phase 3 Architecture Note

**Version:** 0.3.0  
**Date:** 2026-04-06

## Goal

Extract the model-inference and per-token decision logic from the monolithic
`src/generate.py` into two new sub-packages:

- `src/backends/` — isolate HuggingFace tokenizer/model assumptions
- `src/mechanisms/` — own the clipped-logit aggregation, SVT gating, and
  per-token accounting

Additionally, remove the duplicate `compute_max_private_tokens` implementation
from `src/config.py` (it was a copy of the authoritative version in
`src/privacy_accounting.py` with a different argument order).

All changes are behavior-preserving.  Every external import path that worked in
0.2.0 continues to work in 0.3.0.

---

## New Sub-packages

### `src/backends/`

```
src/backends/
├── __init__.py                  # re-exports ModelBackend, HuggingFaceCausalLM
├── base.py                      # ModelBackend ABC
└── huggingface_causal_lm.py     # HuggingFaceCausalLM (concrete)
```

**`ModelBackend` ABC** (`src/backends/base.py`)

```python
class ModelBackend(ABC):
    @property @abstractmethod
    def eos_token_id(self) -> Optional[int]: ...

    @property @abstractmethod
    def padding_side(self) -> str: ...
    @padding_side.setter @abstractmethod
    def padding_side(self, side: str) -> None: ...

    @abstractmethod
    def get_next_token_logits(
        self, prompts: List[str], generated_tokens: List[int]
    ) -> Tensor: ...

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str: ...
```

**`HuggingFaceCausalLM`** (`src/backends/huggingface_causal_lm.py`)

Wraps `(model, tokenizer, device, micro_batch_size)`.  Contains all
left-padding, micro-batch inference, and token-appending logic that was
previously embedded in `get_next_token_logits()` in `src/generate.py`.
Exposes a `tokenizer` property so the prompt builder can still apply the
model's chat template.

---

### `src/mechanisms/`

```
src/mechanisms/
├── __init__.py                  # re-exports Mechanism, PrivatePredictionMechanism
├── base.py                      # Mechanism ABC
└── private_prediction.py        # PrivatePredictionMechanism + _apply_top_k_filter
```

**`Mechanism` ABC** (`src/mechanisms/base.py`)

```python
class Mechanism(ABC):
    @abstractmethod
    def generate_example(
        self,
        private_prompts: List[str],
        public_prompt: str,
        backend: ModelBackend,
        remaining_budget: int,
        max_total_tokens: int,
    ) -> Tuple[List[int], int, int]: ...
```

**`PrivatePredictionMechanism`** (`src/mechanisms/private_prediction.py`)

Implements Algorithm 1's inner loop (Amin et al. 2024):

1. Get private next-token logits (batch of prompts → `backend.get_next_token_logits`)
2. Optionally get public logits and apply SVT gate
3. Clip-and-aggregate (`src/clip_utils.clip_and_aggregate`)
4. Optionally restrict to top-k of public logits (`_apply_top_k_filter`)
5. Sample via `torch.softmax` + `torch.multinomial` (exponential mechanism)
6. Repeat until EOS, `remaining_budget` private tokens, or `max_total_tokens`

`_apply_top_k_filter` is also exported here (moved from `src/generate.py`).

---

## Changes to `src/generate.py`

`src/generate.py` is now a thin orchestrator and backward-compat re-export hub:

| Symbol | Previously | Now |
|---|---|---|
| `get_next_token_logits` | defined here | backward-compat wrapper → `HuggingFaceCausalLM` |
| `_apply_top_k_filter` | defined here | re-exported from `src/mechanisms/private_prediction` |
| `_generate_single_example` | defined here | backward-compat wrapper → `PrivatePredictionMechanism` |
| `generate_batch_examples` | calls `_generate_single_example` inline | creates backend+mechanism, calls `_run_batch_generation` |
| `generate_synthetic_dataset` | creates model/tokenizer directly | creates `HuggingFaceCausalLM` + `PrivatePredictionMechanism` once at top |
| `_run_batch_generation` | **new** internal helper | the old outer-loop from `generate_batch_examples`, now mechanism-agnostic |

All public symbols (`SyntheticExample`, `generate_synthetic_dataset`,
`generate_batch_examples`, `generate_one_example`, `BatchDescriptor`, …)
remain importable from `src.generate`.

---

## `src/config.py` — Duplicate Removed

`compute_max_private_tokens` in `src/config.py` was an independent copy of
the same function in `src/privacy_accounting.py`, with a different argument
order:

| Location | Argument order |
|---|---|
| `src/config.py` (old) | `(target_epsilon, delta, **batch_size**, **clip_bound**, temperature, svt_noise)` |
| `src/privacy_accounting.py` (authoritative) | `(target_epsilon, delta, **clip_bound**, **batch_size**, temperature, svt_noise)` |

The `src/config.py` version is now a wrapper that re-maps its positional
arguments to the authoritative signature:

```python
def compute_max_private_tokens(
    target_epsilon, delta,
    batch_size,    # NOTE: config.py order preserved
    clip_bound, temperature, svt_noise=None,
):
    from src.privacy_accounting import compute_max_private_tokens as _auth
    return _auth(target_epsilon, delta, clip_bound, batch_size, temperature, svt_noise)
```

Existing call sites using the config.py argument order continue to work.

---

## Test Coverage (Phase 3)

| File | Tests | What's covered |
|---|---|---|
| `tests/test_backends.py` | 12 | ModelBackend subclass, property delegation, decode, logits shape, micro-batching, token appending, backward-compat import |
| `tests/test_mechanism.py` | 14 | Mechanism subclass, generate_example return types, token count identity, SVT-disabled → all private, budget exhaustion, EOS stop, max_total_tokens cap, top-k filter, backward-compat imports |
| `tests/test_accounting.py` | 13 | config wrapper parity, arg-order preservation, monotonicity, round-trip epsilon ≤ target, rho consistency, zCDP conversion |

Total tests: **143** (up from 105 in Phase 2).

---

## Invariants Preserved

1. No algorithmic change — the token-sampling math is identical to Phase 2.
2. All Phase 1 + Phase 2 import paths continue to work.
3. Checkpoint/JSONL format unchanged.
4. `generate_synthetic_dataset` public signature unchanged.
5. `PrivacyConfig`, `GenerationConfig`, `ModelConfig`, `DatasetConfig`
   unchanged.
