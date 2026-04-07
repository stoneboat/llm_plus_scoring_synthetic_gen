"""
Tests for the mechanism layer: Mechanism ABC, PrivatePredictionMechanism,
and backward-compat imports (Phase 3 guard).

Contracts protected here:
- PrivatePredictionMechanism is a Mechanism subclass.
- generate_example() returns (token_ids, n_priv, n_pub) with correct types.
- n_priv + n_pub == len(token_ids) (every token is classified).
- When SVT is disabled (svt_threshold=-inf), all tokens are private.
- When remaining_budget=0, generate_example returns an empty token list.
- When EOS token is emitted, generation stops immediately.
- max_total_tokens hard cap is respected.
- _apply_top_k_filter is importable from both src.mechanisms and src.generate.
- PrivatePredictionMechanism importable from src.mechanisms.
- _generate_single_example backward-compat wrapper in src.generate works.
"""

import pytest
import torch
from unittest.mock import MagicMock

from src.mechanisms.base import Mechanism
from src.mechanisms.private_prediction import (
    PrivatePredictionMechanism,
    _apply_top_k_filter,
)
from src.backends.base import ModelBackend
from src.config import PrivacyConfig, GenerationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 16


def _make_privacy_config(svt_enabled=False, clip_bound=1.0, temperature=1.0):
    if svt_enabled:
        return PrivacyConfig(
            clip_bound=clip_bound,
            temperature=temperature,
            svt_threshold=0.5,
            svt_noise=0.2,
        )
    return PrivacyConfig(
        clip_bound=clip_bound,
        temperature=temperature,
        svt_threshold=float("-inf"),
    )


def _make_gen_config(max_private=10, max_total=20, batch_size=4, eos_id=None, top_k=0):
    return GenerationConfig(
        batch_size=batch_size,
        max_private_tokens=max_private,
        max_total_tokens=max_total,
        eos_token_id=eos_id,
        top_k_vocab=top_k,
    )


def _make_backend(logits_fn=None, eos_id=1):
    """Return a mock ModelBackend whose get_next_token_logits calls logits_fn."""
    backend = MagicMock(spec=ModelBackend)
    backend.eos_token_id = eos_id

    if logits_fn is None:
        def logits_fn(prompts, generated):
            return torch.zeros(len(prompts), VOCAB_SIZE)

    backend.get_next_token_logits = logits_fn
    return backend


def _uniform_backend(eos_id=None):
    """Backend that always returns uniform logits (no preferred token)."""
    def logits_fn(prompts, generated):
        return torch.zeros(len(prompts), VOCAB_SIZE)

    backend = MagicMock(spec=ModelBackend)
    backend.eos_token_id = eos_id
    backend.get_next_token_logits = logits_fn
    return backend


# ---------------------------------------------------------------------------
# Type / subclass checks
# ---------------------------------------------------------------------------

def test_private_prediction_is_mechanism():
    pc = _make_privacy_config()
    gc = _make_gen_config()
    m = PrivatePredictionMechanism(pc, gc)
    assert isinstance(m, Mechanism)


# ---------------------------------------------------------------------------
# generate_example basic contracts
# ---------------------------------------------------------------------------

def test_generate_example_returns_tuple():
    pc = _make_privacy_config()
    gc = _make_gen_config(max_private=5, max_total=10)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    result = m.generate_example(["prompt1", "prompt2"], "public", backend,
                                remaining_budget=5, max_total_tokens=10)
    assert isinstance(result, tuple) and len(result) == 3


def test_generate_example_token_ids_are_ints():
    pc = _make_privacy_config()
    gc = _make_gen_config(max_private=5, max_total=5)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    token_ids, n_priv, n_pub = m.generate_example(
        ["p"], "pub", backend, remaining_budget=5, max_total_tokens=5
    )
    assert all(isinstance(t, int) for t in token_ids)


def test_token_counts_sum_to_length():
    """n_priv + n_pub must equal len(token_ids)."""
    pc = _make_privacy_config(svt_enabled=False)
    gc = _make_gen_config(max_private=8, max_total=8)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    token_ids, n_priv, n_pub = m.generate_example(
        ["p"] * 3, "pub", backend, remaining_budget=8, max_total_tokens=8
    )
    assert n_priv + n_pub == len(token_ids), (
        f"n_priv={n_priv} + n_pub={n_pub} != len(token_ids)={len(token_ids)}"
    )


# ---------------------------------------------------------------------------
# SVT disabled → all tokens private
# ---------------------------------------------------------------------------

def test_all_private_when_svt_disabled():
    """With svt_threshold=-inf, every token must be private."""
    pc = _make_privacy_config(svt_enabled=False)
    gc = _make_gen_config(max_private=6, max_total=6)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    token_ids, n_priv, n_pub = m.generate_example(
        ["p"] * 3, "pub", backend, remaining_budget=6, max_total_tokens=6
    )
    assert n_pub == 0, f"Expected 0 public tokens, got {n_pub}"
    assert n_priv == len(token_ids)


# ---------------------------------------------------------------------------
# Budget exhaustion
# ---------------------------------------------------------------------------

def test_remaining_budget_zero_returns_empty():
    """With remaining_budget=0 and SVT disabled, no tokens should be produced."""
    pc = _make_privacy_config(svt_enabled=False)
    gc = _make_gen_config(max_private=10, max_total=10)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    token_ids, n_priv, n_pub = m.generate_example(
        ["p"], "pub", backend, remaining_budget=0, max_total_tokens=10
    )
    assert token_ids == []
    assert n_priv == 0


def test_private_tokens_capped_by_budget():
    """n_priv must not exceed remaining_budget."""
    budget = 3
    pc = _make_privacy_config(svt_enabled=False)
    gc = _make_gen_config(max_private=budget, max_total=20)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    token_ids, n_priv, n_pub = m.generate_example(
        ["p"] * 4, "pub", backend, remaining_budget=budget, max_total_tokens=20
    )
    assert n_priv <= budget


# ---------------------------------------------------------------------------
# EOS stops generation
# ---------------------------------------------------------------------------

def test_eos_stops_generation():
    """After EOS is emitted, no more tokens should be produced."""
    eos_id = 5
    call_count = {"n": 0}

    def logits_fn(prompts, generated):
        # Always return a distribution peaked on eos_id
        logits = torch.full((len(prompts), VOCAB_SIZE), -100.0)
        logits[:, eos_id] = 100.0
        return logits

    pc = _make_privacy_config(svt_enabled=False)
    gc = _make_gen_config(max_private=10, max_total=10, eos_id=eos_id)
    m = PrivatePredictionMechanism(pc, gc)

    backend = MagicMock(spec=ModelBackend)
    backend.eos_token_id = eos_id
    backend.get_next_token_logits = logits_fn

    token_ids, n_priv, n_pub = m.generate_example(
        ["p"] * 4, "pub", backend, remaining_budget=10, max_total_tokens=10
    )
    # EOS should be the only token and generation stops
    assert token_ids == [eos_id], f"Expected [eos_id], got {token_ids}"
    assert len(token_ids) == 1


# ---------------------------------------------------------------------------
# max_total_tokens cap
# ---------------------------------------------------------------------------

def test_max_total_tokens_respected():
    """Generated sequence must not exceed max_total_tokens."""
    max_total = 4
    pc = _make_privacy_config(svt_enabled=False)
    gc = _make_gen_config(max_private=100, max_total=max_total)
    m = PrivatePredictionMechanism(pc, gc)
    backend = _uniform_backend()

    token_ids, _, _ = m.generate_example(
        ["p"] * 3, "pub", backend, remaining_budget=100, max_total_tokens=max_total
    )
    assert len(token_ids) <= max_total


# ---------------------------------------------------------------------------
# _apply_top_k_filter
# ---------------------------------------------------------------------------

def test_top_k_filter_masks_low_tokens():
    """Only top-k tokens should remain unmasked after filtering."""
    k = 3
    logits = torch.zeros(VOCAB_SIZE)
    public_logits = torch.arange(VOCAB_SIZE, dtype=torch.float)  # token 15 is top

    filtered = _apply_top_k_filter(logits, public_logits, k)

    top_indices = public_logits.topk(k).indices.tolist()
    for i in range(VOCAB_SIZE):
        if i in top_indices:
            assert filtered[i] == 0.0, f"Token {i} should be unmasked"
        else:
            assert filtered[i] == float("-inf"), f"Token {i} should be masked"


def test_top_k_filter_importable_from_generate():
    from src.generate import _apply_top_k_filter as f
    assert callable(f)


# ---------------------------------------------------------------------------
# Backward-compat: src.generate._generate_single_example works
# ---------------------------------------------------------------------------

def test_generate_single_example_compat():
    """_generate_single_example in src.generate is callable with model+tokenizer."""
    from src.generate import _generate_single_example

    vocab_size = 8
    seq_len = 4

    class FakeEncoding(dict):
        def to(self, device):
            return self

    tok = MagicMock()
    tok.eos_token_id = None

    def fake_tokenize(texts, **kwargs):
        n = len(texts)
        return FakeEncoding({
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        })

    tok.side_effect = fake_tokenize

    model = MagicMock()

    def fake_forward(**inputs):
        bsz = inputs["input_ids"].shape[0]
        seq = inputs["input_ids"].shape[1]
        out = MagicMock()
        out.logits = torch.zeros(bsz, seq, vocab_size)
        return out

    model.side_effect = fake_forward

    pc = PrivacyConfig(
        clip_bound=1.0, temperature=1.0, svt_threshold=float("-inf")
    )
    gc = GenerationConfig(
        batch_size=2, max_private_tokens=3, max_total_tokens=3, eos_token_id=None
    )

    result = _generate_single_example(
        model, tok, ["p1", "p2"], "pub", pc, gc,
        remaining_private_budget=3, device="cpu", micro_batch_size=8,
    )
    assert isinstance(result, tuple) and len(result) == 3


# ---------------------------------------------------------------------------
# Import checks
# ---------------------------------------------------------------------------

def test_private_prediction_mechanism_importable_from_mechanisms():
    from src.mechanisms import PrivatePredictionMechanism as M
    assert M is PrivatePredictionMechanism
