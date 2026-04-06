"""
Tests for SVT public/private token gate behavior (Phase 1 guard).

Contracts protected here:
- compute_distribution_distance returns 0 for identical distributions
  and a large value (~2) for near-antipodal ones.
- L1 distance between probability distributions is bounded in [0, 2].
- sample_noisy_threshold is reproducible under the same manual seed.
- should_use_private_token returns True when noisy_threshold is very
  low (high-distance case) and False when it is very high (low-distance).
- Return type is (bool, float).

Note: random noise in should_use_private_token is controlled by setting
torch.manual_seed before the call; noise_scale is set to 0.001 to keep
noise negligible while still exercising the Laplace sampling code path.
"""

import torch
import pytest
from src.sparse_vector import (
    compute_distribution_distance,
    sample_noisy_threshold,
    should_use_private_token,
)


VOCAB = 20
BATCH = 5


# ---------------------------------------------------------------------------
# Fixture logit tensors
# ---------------------------------------------------------------------------

def _identical_logits(vocab=VOCAB, batch=BATCH):
    """Private and public both strongly prefer token 0 — distance should be ~0."""
    logits = torch.zeros(vocab)
    logits[0] = 10.0
    return logits.unsqueeze(0).expand(batch, -1).clone(), logits.clone()


def _divergent_logits(vocab=VOCAB, batch=BATCH):
    """Private prefers token 0; public prefers last token — distance should be ~2."""
    private = torch.full((batch, vocab), -10.0)
    private[:, 0] = 10.0
    public = torch.full((vocab,), -10.0)
    public[-1] = 10.0
    return private, public


# ---------------------------------------------------------------------------
# compute_distribution_distance
# ---------------------------------------------------------------------------

def test_distribution_distance_identical_is_near_zero():
    """L1 distance between identical distributions is effectively 0."""
    private_logits, public_logits = _identical_logits()
    dist = compute_distribution_distance(private_logits, public_logits, BATCH)
    assert dist < 1e-4, f"Expected ~0 for identical distributions, got {dist}"


def test_distribution_distance_divergent_is_large():
    """L1 distance is large (~2) when private and public strongly disagree."""
    private_logits, public_logits = _divergent_logits()
    dist = compute_distribution_distance(private_logits, public_logits, BATCH)
    # Two near-one-hot distributions differ by ~2 in L1 norm
    assert dist > 1.5, f"Expected large distance (>1.5) for divergent logits, got {dist}"


def test_distribution_distance_bounded_in_0_2():
    """L1 distance between probability distributions is always in [0, 2]."""
    torch.manual_seed(42)
    for _ in range(20):
        private = torch.randn(BATCH, VOCAB)
        public = torch.randn(VOCAB)
        dist = compute_distribution_distance(private, public, BATCH)
        assert 0.0 <= dist <= 2.0 + 1e-5, (
            f"L1 distance {dist} outside [0, 2]"
        )


def test_distribution_distance_non_negative():
    """Distance is always non-negative (L1 norm property)."""
    torch.manual_seed(7)
    for _ in range(10):
        private = torch.randn(BATCH, VOCAB)
        public = torch.randn(VOCAB)
        dist = compute_distribution_distance(private, public, BATCH)
        assert dist >= 0.0, f"Distance {dist} is negative"


# ---------------------------------------------------------------------------
# sample_noisy_threshold
# ---------------------------------------------------------------------------

def test_sample_noisy_threshold_reproducible():
    """Same torch seed yields identical threshold samples."""
    torch.manual_seed(99)
    t1 = sample_noisy_threshold(0.5, 0.1)
    torch.manual_seed(99)
    t2 = sample_noisy_threshold(0.5, 0.1)
    assert abs(t1 - t2) < 1e-8, "Threshold samples differ under same seed"


def test_sample_noisy_threshold_near_base():
    """With very small noise, sample is close to the base threshold."""
    torch.manual_seed(0)
    base = 0.5
    sigma = 1e-6
    samples = [sample_noisy_threshold(base, sigma) for _ in range(20)]
    for s in samples:
        assert abs(s - base) < 0.01, (
            f"Noisy threshold {s} far from base {base} at sigma={sigma}"
        )


# ---------------------------------------------------------------------------
# should_use_private_token
# ---------------------------------------------------------------------------

def test_should_use_private_token_very_low_threshold():
    """With a threshold of -999, even low-distance inputs should use private."""
    private, public = _identical_logits()
    torch.manual_seed(0)
    use_private, _ = should_use_private_token(
        private, public, BATCH,
        noisy_threshold=-999.0,
        noise_scale=0.001,
    )
    assert use_private, (
        "should_use_private_token should return True when noisy threshold is -999"
    )


def test_should_use_private_token_very_high_threshold():
    """With a threshold of +999, even high-distance inputs should use public."""
    private, public = _divergent_logits()
    torch.manual_seed(0)
    use_private, _ = should_use_private_token(
        private, public, BATCH,
        noisy_threshold=999.0,
        noise_scale=0.001,
    )
    assert not use_private, (
        "should_use_private_token should return False when noisy threshold is +999"
    )


def test_should_use_private_token_high_distance_uses_private():
    """High-distance case with moderate threshold -> private token expected."""
    private, public = _divergent_logits()
    torch.manual_seed(0)
    use_private, noisy_dist = should_use_private_token(
        private, public, BATCH,
        noisy_threshold=0.0,   # distance ~2 >> threshold 0
        noise_scale=0.001,
    )
    assert use_private, (
        "High-distance case should use private token (noisy_dist={noisy_dist:.4f})"
    )


def test_should_use_private_token_low_distance_uses_public():
    """Low-distance case with moderate positive threshold -> public token expected."""
    private, public = _identical_logits()
    torch.manual_seed(0)
    use_private, noisy_dist = should_use_private_token(
        private, public, BATCH,
        noisy_threshold=1.0,   # distance ~0 << threshold 1
        noise_scale=0.001,
    )
    assert not use_private, (
        f"Low-distance case should use public token (noisy_dist={noisy_dist:.4f})"
    )


def test_should_use_private_token_return_type():
    """Return value is a (bool, float) tuple."""
    private, public = _identical_logits()
    result = should_use_private_token(private, public, BATCH, 0.5, 0.1)
    assert isinstance(result, tuple) and len(result) == 2, (
        "Expected a 2-tuple"
    )
    use_private, noisy_dist = result
    assert isinstance(use_private, bool), f"First element must be bool, got {type(use_private)}"
    assert isinstance(noisy_dist, float), f"Second element must be float, got {type(noisy_dist)}"
