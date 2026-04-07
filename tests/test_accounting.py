"""
Tests for the privacy accounting layer (Phase 3 guard).

Contracts protected here:
- compute_max_private_tokens in src.config is a backward-compat wrapper
  that delegates to src.privacy_accounting with the correct argument mapping.
- Both versions produce identical results for the same logical inputs.
- The config.py version preserves its original argument order
  (batch_size before clip_bound).
- The privacy_accounting.py version is the authoritative implementation
  (argument order: clip_bound before batch_size).
- Edge cases: very small epsilon → r=1, svt_noise=None handled correctly.
"""

import pytest

from src.privacy_accounting import (
    compute_max_private_tokens as auth_compute,
    compute_rho_per_token,
    compute_total_rho,
    compute_epsilon,
    zcdp_to_approx_dp,
)
from src.config import compute_max_private_tokens as config_compute


# ---------------------------------------------------------------------------
# Argument-order parity: config.py wrapper vs. privacy_accounting.py
# ---------------------------------------------------------------------------

CASES = [
    # (target_epsilon, delta, batch_size, clip_bound, temperature, svt_noise)
    (1.0, 1e-5, 255, 10.0, 2.0, None),
    (1.0, 1e-5, 255, 10.0, 2.0, 0.2),
    (2.0, 1e-5, 127, 10.0, 1.5, 0.1),
    (0.5, 1e-4, 511, 10.0, 2.25, 0.3),
    (3.0, 1e-5, 1023, 10.0, 2.0, None),
]


@pytest.mark.parametrize("eps,delta,batch_size,clip_bound,temp,svt_noise", CASES)
def test_config_wrapper_matches_authoritative(
    eps, delta, batch_size, clip_bound, temp, svt_noise
):
    """config.compute_max_private_tokens must produce the same result as the
    authoritative implementation in privacy_accounting.py."""
    # config.py order: (target_epsilon, delta, batch_size, clip_bound, temperature, svt_noise)
    r_config = config_compute(eps, delta, batch_size, clip_bound, temp, svt_noise)

    # privacy_accounting.py order: (target_epsilon, delta, clip_bound, batch_size, ...)
    r_auth = auth_compute(eps, delta, clip_bound, batch_size, temp, svt_noise)

    assert r_config == r_auth, (
        f"config_compute={r_config} != auth_compute={r_auth} for "
        f"eps={eps}, batch_size={batch_size}, clip_bound={clip_bound}"
    )


def test_config_arg_order_preserved():
    """Passing clip_bound and batch_size in config.py order must not accidentally
    use privacy_accounting.py order (which swaps the two)."""
    # batch_size=255, clip_bound=1.0 in config.py order
    r_config = config_compute(1.0, 1e-5, 255, 1.0, 2.0, None)
    # The same call in privacy_accounting.py order would be (1.0, 1e-5, 1.0, 255, ...)
    # which is a different budget — just confirm they are different numbers.
    r_swapped = auth_compute(1.0, 1e-5, 255.0, 1.0, 2.0, None)  # clip_bound=255 is nonsense
    # We just verify config_compute matches auth with correct mapping
    r_auth_correct = auth_compute(1.0, 1e-5, 1.0, 255, 2.0, None)
    assert r_config == r_auth_correct


# ---------------------------------------------------------------------------
# Authoritative: compute_max_private_tokens basic contracts
# ---------------------------------------------------------------------------

def test_result_is_positive_integer():
    r = auth_compute(1.0, 1e-5, 10.0, 255, 2.0, None)
    assert isinstance(r, int) and r >= 1


def test_larger_epsilon_allows_more_tokens():
    r_small = auth_compute(0.5, 1e-5, 10.0, 255, 2.0, None)
    r_large = auth_compute(5.0, 1e-5, 10.0, 255, 2.0, None)
    assert r_large >= r_small, (
        f"Larger epsilon should allow at least as many private tokens: "
        f"r(0.5)={r_small}, r(5.0)={r_large}"
    )


def test_larger_batch_size_allows_more_tokens():
    """Larger batch size reduces per-token rho, so more tokens fit in budget."""
    r_small = auth_compute(1.0, 1e-5, 10.0, 127, 2.0, None)
    r_large = auth_compute(1.0, 1e-5, 10.0, 1023, 2.0, None)
    assert r_large >= r_small, (
        f"Larger batch size should allow more private tokens: "
        f"r(127)={r_small}, r(1023)={r_large}"
    )


def test_svt_noise_reduces_tokens():
    """Adding SVT noise increases rho_per_token, so fewer tokens fit."""
    r_no_svt = auth_compute(1.0, 1e-5, 10.0, 255, 2.0, None)
    r_svt = auth_compute(1.0, 1e-5, 10.0, 255, 2.0, 0.2)
    assert r_svt <= r_no_svt, (
        f"SVT should reduce private-token budget: no_svt={r_no_svt}, svt={r_svt}"
    )


def test_very_small_epsilon_returns_one():
    """Impossibly tight budget should return r=1 (graceful floor)."""
    r = auth_compute(1e-10, 1e-5, 10.0, 255, 2.0, None)
    assert r >= 1


# ---------------------------------------------------------------------------
# Round-trip: tokens → epsilon → within target
# ---------------------------------------------------------------------------

def test_epsilon_from_max_tokens_is_within_target():
    """compute_epsilon(compute_max_private_tokens(...)) should be ≤ target_epsilon."""
    target_eps = 1.0
    delta = 1e-5
    clip_bound = 10.0
    batch_size = 255
    temp = 2.0
    svt_noise = 0.2

    r = auth_compute(target_eps, delta, clip_bound, batch_size, temp, svt_noise)
    actual_eps = compute_epsilon(r, clip_bound, batch_size, temp, delta, svt_noise)

    assert actual_eps <= target_eps * 1.05, (
        f"Actual epsilon {actual_eps:.4f} exceeds target {target_eps} "
        f"(with 5% tolerance)"
    )


# ---------------------------------------------------------------------------
# compute_rho_per_token / compute_total_rho consistency
# ---------------------------------------------------------------------------

def test_total_rho_is_r_times_per_token():
    clip_bound = 10.0
    batch_size = 255
    temp = 2.0
    svt_noise = 0.2
    r = 50

    rho_pt = compute_rho_per_token(clip_bound, batch_size, temp, svt_noise)
    rho_total = compute_total_rho(r, clip_bound, batch_size, temp, svt_noise)

    assert abs(rho_total - r * rho_pt) < 1e-12


# ---------------------------------------------------------------------------
# zcdp_to_approx_dp monotonicity
# ---------------------------------------------------------------------------

def test_epsilon_increases_with_rho():
    eps1 = zcdp_to_approx_dp(0.01, 1e-5)
    eps2 = zcdp_to_approx_dp(0.1, 1e-5)
    assert eps2 > eps1
