"""
Operational privacy planning helpers.

Planning answers the question: "given a privacy budget (epsilon, delta)
and mechanism parameters, how many private tokens can we generate?"

These functions are distinct from mechanism-specific cost formulas
(src/privacy/analyses/) and generic conversions (src/privacy/conversions.py):
they answer operational, pre-generation questions.

Current helper signatures preserve the argument order used by
src/privacy_accounting.py (authoritative order: clip_bound before batch_size).
See src/config.py for the backward-compat wrapper with swapped argument order.
"""

import math
from typing import Optional

from src.privacy.analyses.private_prediction import rho_per_token, total_rho
from src.privacy.conversions import zcdp_to_approx_dp


def compute_epsilon(
    num_private_tokens: int,
    clip_bound: float,
    batch_size: int,
    temperature: float,
    delta: float,
    svt_noise: Optional[float] = None,
) -> float:
    """Compute the (epsilon, delta)-DP guarantee for the full algorithm.

    Computes the realized epsilon for a given number of private tokens.

    Args:
        num_private_tokens: r, total private tokens (worst-case batch).
        clip_bound: c.
        batch_size: s.
        temperature: tau.
        delta: target failure probability.
        svt_noise: sigma for SVT.  None if SVT disabled.

    Returns:
        Realized epsilon (float).
    """
    rho = total_rho(num_private_tokens, clip_bound, batch_size, temperature, svt_noise)
    return zcdp_to_approx_dp(rho, delta)


def compute_max_private_tokens(
    target_epsilon: float,
    delta: float,
    clip_bound: float,
    batch_size: int,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> int:
    """Compute the maximum number of private tokens within a privacy budget.

    Inverts the epsilon formula:
        eps = rho + sqrt(4 * rho * log(1/delta))
    where rho = r * rho_per_token.

    Solving for rho (treating rho as the unknown):
        eps^2 = rho^2 + 8*rho*L + 4*rho*L  →  quadratic in rho,
    then r = floor(rho / rho_per_token).

    IMPORTANT argument order: clip_bound before batch_size.
    For the legacy argument order (batch_size before clip_bound), use
    src.config.compute_max_private_tokens or src.privacy_accounting.compute_max_private_tokens
    which wraps this function.

    Args:
        target_epsilon: desired epsilon budget.
        delta: failure probability.
        clip_bound: c.
        batch_size: s.
        temperature: tau.
        svt_noise: sigma for SVT.  None if SVT disabled.

    Returns:
        Maximum number of private tokens r (int >= 1).
    """
    rho_pt = rho_per_token(clip_bound, batch_size, temperature, svt_noise)
    if rho_pt <= 0:
        return 10000

    log_inv_delta = math.log(1.0 / delta)
    # Solve: eps = x + sqrt(4*x*L) where x = r * rho_pt
    a = 1.0
    b = -(2.0 * target_epsilon + 4.0 * log_inv_delta)
    c_coeff = target_epsilon ** 2
    disc = b ** 2 - 4.0 * a * c_coeff
    if disc < 0:
        return 1
    total_rho_val = (-b - math.sqrt(disc)) / (2.0 * a)
    if total_rho_val <= 0:
        return 1

    return max(1, int(total_rho_val / rho_pt))
