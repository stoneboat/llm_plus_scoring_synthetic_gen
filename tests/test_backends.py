"""
Tests for the model backend layer: ModelBackend ABC, HuggingFaceCausalLM
(Phase 3 guard).

Contracts protected here:
- HuggingFaceCausalLM is a ModelBackend subclass.
- padding_side property reads from and writes to the underlying tokenizer.
- eos_token_id is delegated to the tokenizer.
- decode() delegates to the tokenizer.
- get_next_token_logits() returns shape (len(prompts), vocab_size) and
  uses micro-batching correctly.
- generated_tokens are appended correctly to each prompt.
- Backward-compat: get_next_token_logits() in src.generate still callable.
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, call

from src.backends.base import ModelBackend
from src.backends.huggingface_causal_lm import HuggingFaceCausalLM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeEncoding(dict):
    """Dict subclass that supports `.to(device)` like a HF BatchEncoding."""
    def to(self, device):
        return self


def _make_tokenizer(vocab_size=32, eos_id=1, padding_side="left"):
    tok = MagicMock()
    tok.eos_token_id = eos_id
    tok.padding_side = padding_side
    return tok


def _fake_tokenize(seq_len):
    """Return a tokenize side_effect that yields FakeEncoding dicts."""
    def _fn(texts, **kwargs):
        n = len(texts)
        return FakeEncoding({
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        })
    return _fn


def _fake_model_forward(vocab_size):
    """Return a model side_effect that yields plausible logits."""
    def _fn(**inputs):
        bsz = inputs["input_ids"].shape[0]
        seq = inputs["input_ids"].shape[1]
        out = MagicMock()
        out.logits = torch.zeros(bsz, seq, vocab_size)
        return out
    return _fn


# ---------------------------------------------------------------------------
# Type / subclass checks
# ---------------------------------------------------------------------------

def test_hf_backend_is_model_backend():
    tok = _make_tokenizer()
    model = MagicMock()
    backend = HuggingFaceCausalLM(model, tok)
    assert isinstance(backend, ModelBackend)


# ---------------------------------------------------------------------------
# Property delegation
# ---------------------------------------------------------------------------

def test_eos_token_id_from_tokenizer():
    tok = _make_tokenizer(eos_id=7)
    backend = HuggingFaceCausalLM(MagicMock(), tok)
    assert backend.eos_token_id == 7


def test_padding_side_read():
    tok = _make_tokenizer(padding_side="right")
    backend = HuggingFaceCausalLM(MagicMock(), tok)
    assert backend.padding_side == "right"


def test_padding_side_write():
    tok = _make_tokenizer(padding_side="right")
    backend = HuggingFaceCausalLM(MagicMock(), tok)
    backend.padding_side = "left"
    assert tok.padding_side == "left"


def test_tokenizer_property_exposed():
    tok = _make_tokenizer()
    backend = HuggingFaceCausalLM(MagicMock(), tok)
    assert backend.tokenizer is tok


# ---------------------------------------------------------------------------
# decode()
# ---------------------------------------------------------------------------

def test_decode_delegates_to_tokenizer():
    tok = _make_tokenizer()
    tok.decode = MagicMock(return_value="hello world")
    backend = HuggingFaceCausalLM(MagicMock(), tok)
    result = backend.decode([1, 2, 3], skip_special_tokens=True)
    tok.decode.assert_called_once_with([1, 2, 3], skip_special_tokens=True)
    assert result == "hello world"


# ---------------------------------------------------------------------------
# get_next_token_logits() shape and micro-batching
# ---------------------------------------------------------------------------

def _build_backend_with_fake_tokenizer(vocab_size, num_prompts, seq_len, device="cpu"):
    """Build a HuggingFaceCausalLM whose tokenizer returns predictable tensors."""
    tok = _make_tokenizer()
    tok.side_effect = _fake_tokenize(seq_len)

    model = MagicMock()
    model.side_effect = _fake_model_forward(vocab_size)

    return HuggingFaceCausalLM(model, tok, device=device, micro_batch_size=2)


def test_logits_shape_single_prompt():
    vocab_size = 16
    backend = _build_backend_with_fake_tokenizer(vocab_size, num_prompts=1, seq_len=5)
    logits = backend.get_next_token_logits(["hello"], [])
    assert logits.shape == (1, vocab_size)


def test_logits_shape_multiple_prompts():
    vocab_size = 16
    backend = _build_backend_with_fake_tokenizer(vocab_size, num_prompts=5, seq_len=5)
    logits = backend.get_next_token_logits(["p"] * 5, [])
    assert logits.shape == (5, vocab_size)


def test_micro_batch_splits_correctly():
    """With micro_batch_size=2 and 5 prompts, the model is called 3 times."""
    vocab_size = 16
    tok = _make_tokenizer()
    call_batches = []

    def fake_tokenize(texts, **kwargs):
        n = len(texts)
        call_batches.append(n)
        return FakeEncoding({
            "input_ids": torch.zeros(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        })

    tok.side_effect = fake_tokenize

    model = MagicMock()
    model.side_effect = _fake_model_forward(vocab_size)

    backend = HuggingFaceCausalLM(model, tok, device="cpu", micro_batch_size=2)
    backend.get_next_token_logits(["p"] * 5, [])

    assert call_batches == [2, 2, 1], (
        f"Expected micro-batch splits [2,2,1], got {call_batches}"
    )


def test_generated_tokens_appended():
    """When generated_tokens is non-empty, input_ids are extended."""
    vocab_size = 8
    seq_len = 3
    gen_tokens = [10, 20]
    tok = _make_tokenizer()
    tok.side_effect = _fake_tokenize(seq_len)

    seen_input_ids = []

    def fake_forward(**inputs):
        seen_input_ids.append(inputs["input_ids"].shape[1])
        bsz = inputs["input_ids"].shape[0]
        seq = inputs["input_ids"].shape[1]
        out = MagicMock()
        out.logits = torch.zeros(bsz, seq, vocab_size)
        return out

    model = MagicMock()
    model.side_effect = fake_forward

    backend = HuggingFaceCausalLM(model, tok, device="cpu", micro_batch_size=8)
    backend.get_next_token_logits(["hello"], gen_tokens)

    # seq_len (3) + len(gen_tokens) (2) = 5
    assert seen_input_ids == [seq_len + len(gen_tokens)], (
        f"Expected sequence length {seq_len + len(gen_tokens)}, got {seen_input_ids}"
    )


# ---------------------------------------------------------------------------
# Backward-compat: src.generate still exposes get_next_token_logits
# ---------------------------------------------------------------------------

def test_get_next_token_logits_importable_from_generate():
    from src.generate import get_next_token_logits as f
    assert callable(f)
