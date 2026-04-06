"""
Tests for logit clipping invariants (Phase 1 guard).

Contracts protected here:
- clip_logits maps every component to [-c, c] with the maximum at exactly c.
- clip_logits is lossless (preserves softmax) when input range <= 2c.
- clip_and_aggregate divides by expected_batch_size (not actual batch size),
  which is a deliberate privacy-accounting requirement.
- Aggregated result is bounded by [-c, c] when actual == expected batch size.
"""

import torch
import pytest
from src.clip_utils import clip_logits, clip_and_aggregate


# ---------------------------------------------------------------------------
# clip_logits
# ---------------------------------------------------------------------------

def test_clip_logits_max_equals_c():
    """After clipping, the maximum component is exactly c (recentering)."""
    logits = torch.tensor([3.0, -10.0, 7.0, 1.0])
    c = 5.0
    clipped = clip_logits(logits, c)
    assert abs(clipped.max().item() - c) < 1e-5, (
        f"Expected max={c}, got {clipped.max().item()}"
    )


def test_clip_logits_min_ge_neg_c():
    """After clipping, all components are >= -c."""
    logits = torch.tensor([100.0, -200.0, 0.0, 50.0])
    c = 4.0
    clipped = clip_logits(logits, c)
    assert clipped.min().item() >= -c - 1e-5, (
        f"Expected min >= {-c}, got {clipped.min().item()}"
    )


def test_clip_logits_all_within_range():
    """All clipped logits lie in the closed interval [-c, c]."""
    torch.manual_seed(0)
    logits = torch.randn(200) * 20.0
    c = 7.0
    clipped = clip_logits(logits, c)
    assert (clipped >= -c - 1e-5).all(), "Some values below -c after clipping"
    assert (clipped <= c + 1e-5).all(), "Some values above c after clipping"


def test_clip_logits_preserves_softmax_when_lossless():
    """When input range <= 2c, clipping is lossless: softmax is unchanged."""
    logits = torch.tensor([2.0, 1.0, 0.5, -0.3])  # range ~2.3, < 2*c=20
    c = 10.0
    clipped = clip_logits(logits, c)
    assert torch.allclose(
        torch.softmax(logits, dim=-1),
        torch.softmax(clipped, dim=-1),
        atol=1e-5,
    ), "softmax should be invariant when input range <= 2c"


def test_clip_logits_2d_batch():
    """clip_logits handles 2-D batched input (batch, vocab_size)."""
    torch.manual_seed(1)
    logits = torch.randn(4, 50) * 15.0
    c = 6.0
    clipped = clip_logits(logits, c)
    assert clipped.shape == logits.shape
    # Each row should have max == c
    row_maxes = clipped.max(dim=-1).values
    assert torch.allclose(row_maxes, torch.full_like(row_maxes, c), atol=1e-5)
    assert (clipped >= -c - 1e-5).all()


def test_clip_logits_idempotent():
    """Applying clip_logits twice with the same c yields the same result."""
    torch.manual_seed(2)
    logits = torch.randn(30) * 10.0
    c = 5.0
    once = clip_logits(logits, c)
    twice = clip_logits(once, c)
    assert torch.allclose(once, twice, atol=1e-5), (
        "clip_logits should be idempotent"
    )


# ---------------------------------------------------------------------------
# clip_and_aggregate
# ---------------------------------------------------------------------------

def test_clip_and_aggregate_output_shape():
    """Output has shape (vocab_size,) regardless of batch size."""
    batch = torch.randn(5, 100)
    result = clip_and_aggregate(batch, c=2.0, expected_batch_size=5)
    assert result.shape == (100,)


def test_clip_and_aggregate_divides_by_expected_not_actual():
    """
    Division is by expected_batch_size, not actual batch size.

    This is the critical privacy-accounting invariant: sensitivity = c/s
    regardless of how many examples actually land in a batch.

    Setup: actual_size=3, expected_size=5.
    Each row clips to max=c. Token 0 wins in every row.
    Aggregated token-0 value = sum_of_clipped[0] / expected_size
                              = 3 * c / 5.
    """
    vocab = 20
    c = 2.0
    actual_size = 3
    expected_size = 5
    # All logits very negative except token 0, ensuring clip max maps to c
    logits = torch.full((actual_size, vocab), -100.0)
    logits[:, 0] = 100.0
    result = clip_and_aggregate(logits, c, expected_batch_size=expected_size)
    expected_token0 = actual_size * c / expected_size
    assert abs(result[0].item() - expected_token0) < 1e-4, (
        f"Expected result[0]={expected_token0:.4f}, got {result[0].item():.4f}. "
        "Ensure division is by expected_batch_size, not actual."
    )


def test_clip_and_aggregate_bounded_when_sizes_equal():
    """When actual == expected batch size, output is bounded by [-c, c]."""
    torch.manual_seed(3)
    batch = torch.randn(10, 200) * 50.0
    c = 3.0
    result = clip_and_aggregate(batch, c, expected_batch_size=10)
    assert result.max().item() <= c + 1e-4, "Aggregated logits exceed +c"
    assert result.min().item() >= -c - 1e-4, "Aggregated logits go below -c"


def test_clip_and_aggregate_single_row():
    """Aggregation of a single-row batch equals that row's clipped logits / expected."""
    logits = torch.tensor([[5.0, -3.0, 10.0, 2.0]])
    c = 4.0
    expected = 3  # intentionally != actual (1) to test the division
    result = clip_and_aggregate(logits, c, expected_batch_size=expected)
    expected_result = clip_logits(logits, c).squeeze(0) / expected
    assert torch.allclose(result, expected_result, atol=1e-5)
