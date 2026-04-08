"""
Privacy reporting and metadata helpers.

Reporting is distinct from formula evaluation and planning.  Its job is to
package privacy parameters and guarantees into human-readable summaries and
machine-readable metadata payloads.

Current outputs:
- privacy_report(...)        -- the existing dict used by src/generate.py and scripts
- privacy_metadata(...)      -- structured run-level metadata for JSONL headers
"""

from typing import Optional

from src.privacy.analyses.private_prediction import rho_per_token
from src.privacy.conversions import zcdp_to_approx_dp


def privacy_report(
    num_private_tokens: int,
    clip_bound: float,
    batch_size: int,
    temperature: float,
    delta: float,
    svt_noise: Optional[float] = None,
) -> dict:
    """Return a summary dict of privacy parameters and the realized guarantee.

    Used by src/generate.py (pre-generation print) and scripts for logging.
    The dict schema is stable and consumed by downstream tooling.

    Args:
        num_private_tokens: r, planned or realized private token count.
        clip_bound: c.
        batch_size: s.
        temperature: tau.
        delta: failure probability.
        svt_noise: sigma for SVT.  None if SVT disabled.

    Returns:
        Dict with keys: rho_per_token, total_rho, epsilon, delta,
        num_private_tokens, clip_bound, batch_size, temperature, svt_noise.
    """
    rho_pt = rho_per_token(clip_bound, batch_size, temperature, svt_noise)
    total_rho = num_private_tokens * rho_pt
    epsilon = zcdp_to_approx_dp(total_rho, delta)

    return {
        "rho_per_token": rho_pt,
        "total_rho": total_rho,
        "epsilon": epsilon,
        "delta": delta,
        "num_private_tokens": num_private_tokens,
        "clip_bound": clip_bound,
        "batch_size": batch_size,
        "temperature": temperature,
        "svt_noise": svt_noise,
    }


def privacy_metadata(
    epsilon: float,
    delta: float,
    batch_size: int,
    clip_bound: float,
    temperature: float,
    public_temperature: float,
    svt_threshold: float,
    svt_noise: Optional[float],
    top_k_vocab: int,
    max_private_tokens: int,
) -> dict:
    """Return the privacy-metadata payload written to JSONL run headers.

    This mirrors the ``_metadata`` dict currently assembled in
    ``scripts/run_experiment.py``, giving it a stable, named home.

    Args:
        epsilon: planned epsilon.
        delta: planned delta.
        batch_size: s.
        clip_bound: c.
        temperature: tau (private tokens).
        public_temperature: tau_pub (public tokens).
        svt_threshold: theta (SVT distance threshold; -inf if disabled).
        svt_noise: sigma (SVT noise; None if disabled).
        top_k_vocab: k for top-k vocabulary filter (0 = off).
        max_private_tokens: planned r.

    Returns:
        Dict suitable for the ``_metadata`` record in a JSONL checkpoint.
    """
    return {
        "epsilon": epsilon,
        "delta": delta,
        "batch_size": batch_size,
        "clip_bound": clip_bound,
        "temperature": temperature,
        "public_temperature": public_temperature,
        "svt_threshold": svt_threshold,
        "svt_noise": svt_noise,
        "top_k_vocab": top_k_vocab,
        "max_private_tokens": max_private_tokens,
    }
