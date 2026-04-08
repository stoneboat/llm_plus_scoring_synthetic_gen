"""
Generic zCDP → approximate-DP conversion utilities.

These functions are independent of any specific mechanism and operate on
bound objects (ZCDPBound, ApproxDPBound) as well as raw floats for
compatibility with older call sites.

Reference: Bun & Steinke (2016), "Concentrated Differential Privacy:
Simplifications, Extensions, and Lower Bounds."
"""

import math

from src.privacy.bounds import ZCDPBound, ApproxDPBound


def zcdp_to_approx_dp(rho: float, delta: float) -> float:
    """Convert a zCDP parameter rho to epsilon for (epsilon, delta)-DP.

    Uses the analytical formula from Bun & Steinke (2016):
        epsilon = rho + sqrt(4 * rho * log(1/delta))

    Args:
        rho: the zCDP parameter (>= 0).
        delta: the failure probability (in (0, 1]).

    Returns:
        epsilon such that the mechanism is (epsilon, delta)-DP.
    """
    if rho <= 0:
        return 0.0
    return rho + math.sqrt(4.0 * rho * math.log(1.0 / delta))


def zcdp_to_dp_tight(rho: float, delta: float) -> float:
    """Convert zCDP to (epsilon, delta)-DP using the tight bound.

    Uses Google's dp-accounting library when available, falling back to the
    analytical formula.  Currently the library path is stubbed and always
    falls back.

    Reference: Canonne, Kamath & Steinke (2020).
    """
    try:
        from dp_accounting.rdp import rdp_privacy_accountant  # noqa: F401
        raise ImportError("Using analytical bound")
    except (ImportError, Exception):
        pass
    return zcdp_to_approx_dp(rho, delta)


def bound_to_approx_dp(bound: ZCDPBound, delta: float) -> ApproxDPBound:
    """Convert a ZCDPBound to an ApproxDPBound at the given delta.

    Args:
        bound: a ZCDPBound object.
        delta: the failure probability.

    Returns:
        An ApproxDPBound representing the converted guarantee.
    """
    epsilon = zcdp_to_approx_dp(bound.rho, delta)
    return ApproxDPBound(epsilon=epsilon, delta=delta)
