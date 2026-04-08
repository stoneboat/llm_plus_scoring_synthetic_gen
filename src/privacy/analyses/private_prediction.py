"""
Mechanism-specific privacy analysis for the private-prediction algorithm.

This module owns the theorem-specific privacy cost model from Amin et al.
(2024).  It answers: "what is the privacy cost of one private token under
Algorithm 1?"

Theorem 1 (Amin et al. 2024):
    rho = r * (1/2 * (c / (s*tau))^2 + 2 / (s*sigma)^2)

where:
    r   = number of private tokens
    c   = logit clipping bound
    s   = expected batch size
    tau = sampling temperature
    sigma = SVT noise level (omit the SVT term when SVT is disabled)

Important adjacency note:
    The exponential-mechanism term uses score sensitivity Delta_q = c/s,
    which matches the paper's add/remove-style neighboring relation on
    clipped-and-aggregated logits.  Under replace-one adjacency the
    sensitivity would be 2c/s.

Public API:
    rho_per_token(...)          -- per-private-token zCDP cost (float)
    total_rho(...)              -- total zCDP cost for r private tokens (float)
    token_bound(...)            -- ZCDPBound for one private token
    private_token_event(...)    -- PrivacyEvent(bound=ZCDPBound(...)) for one private token
    public_token_event()        -- zero-cost PrivacyEvent(bound=ZCDPBound(rho=0.0))

Phase 3.5b note:
    private_token_event() and public_token_event() now construct PrivacyEvent
    with bound=ZCDPBound(...) rather than rho=...  This is consistent with the
    updated PrivacyEvent interface that carries a PrivacyBound object.
"""

import math
from typing import Optional

from src.privacy.bounds import ZCDPBound
from src.privacy.events import PrivacyEvent


# ---------------------------------------------------------------------------
# Core cost formulas (mechanism-specific, Theorem 1)
# ---------------------------------------------------------------------------

def rho_per_token(
    clip_bound: float,
    batch_size: int,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> float:
    """Compute the zCDP cost (rho) per private token.

    From Theorem 1, the per-token rho decomposes into:
        rho_exp = 1/2 * (c / (s * tau))^2     (exponential mechanism)
        rho_svt = 2 / (s * sigma)^2            (sparse vector technique)

    The SVT term is zero when SVT is disabled (svt_noise is None).

    Args:
        clip_bound: c, the logit clipping range [-c, c].
        batch_size: s, expected number of prompts per batch.
        temperature: tau, sampling temperature.
        svt_noise: sigma, SVT Laplace noise scale.  None if SVT disabled.

    Returns:
        rho per private token (a non-negative float).
    """
    rho_exp = 0.5 * (clip_bound / (batch_size * temperature)) ** 2
    rho_svt = 0.0
    if svt_noise is not None and svt_noise > 0:
        rho_svt = 2.0 / (batch_size * svt_noise) ** 2
    return rho_exp + rho_svt


def total_rho(
    num_private_tokens: int,
    clip_bound: float,
    batch_size: int,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> float:
    """Compute total zCDP parameter for r private tokens.

    By sequential composition: total_rho = r * rho_per_token.
    By parallel composition across independent batches, this is also the
    total rho for the entire dataset (each batch is an independent draw).

    Args:
        num_private_tokens: r, total number of private tokens.
        clip_bound: c.
        batch_size: s.
        temperature: tau.
        svt_noise: sigma.  None if SVT disabled.

    Returns:
        Total rho (non-negative float).
    """
    return num_private_tokens * rho_per_token(
        clip_bound, batch_size, temperature, svt_noise
    )


def token_bound(
    clip_bound: float,
    batch_size: int,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> ZCDPBound:
    """Return the ZCDPBound for one private token.

    Args:
        clip_bound: c.
        batch_size: s.
        temperature: tau.
        svt_noise: sigma.  None if SVT disabled.

    Returns:
        ZCDPBound for a single private token event.
    """
    return ZCDPBound(rho=rho_per_token(clip_bound, batch_size, temperature, svt_noise))


# ---------------------------------------------------------------------------
# Privacy event constructors
# ---------------------------------------------------------------------------

def private_token_event(
    clip_bound: float,
    batch_size: int,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> PrivacyEvent:
    """Return a PrivacyEvent representing one private token.

    Constructs the event with bound=ZCDPBound(rho=...) so that the event
    carries its privacy cost as a native bound object rather than a raw scalar.
    The ZCDPAccountant will extract rho from event.bound explicitly.

    Args:
        clip_bound: c.
        batch_size: s.
        temperature: tau.
        svt_noise: sigma.  None if SVT disabled.

    Returns:
        PrivacyEvent with bound=ZCDPBound(rho=rho_per_token(...)).
    """
    bound = token_bound(clip_bound, batch_size, temperature, svt_noise)
    return PrivacyEvent(bound=bound, label="private_token")


def public_token_event() -> PrivacyEvent:
    """Return a zero-cost PrivacyEvent representing one public token.

    A public token is sampled from the public (non-sensitive) distribution
    via the SVT gate and incurs no additional privacy cost.  The event's
    bound is ZCDPBound(rho=0.0), which is trivial.

    Returns:
        PrivacyEvent with bound=ZCDPBound(rho=0.0).
    """
    return PrivacyEvent(bound=ZCDPBound(rho=0.0), label="public_token")
