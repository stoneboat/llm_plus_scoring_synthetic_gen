"""
Logit clipping with recentering for bounded-sensitivity private token selection.

Implements Eq. 1 from Amin et al. (2024):
    clip_c(z)_i = max{-c, z_i - max_j{z_j} + c}

This re-centers logits so the maximum maps to c, then clips from below at -c.
The key property: softmax(z) is invariant to constant shifts, so if the logits
already span a range <= 2c, clipping with recentering is lossless.
"""

import torch
from torch import Tensor


def clip_logits(logits: Tensor, c: float) -> Tensor:
    """Clip and recenter a logit vector to the range [-c, c].

    For each logit vector z, compute:
        clip_c(z)_i = max(-c, z_i - max(z) + c)

    This ensures all components lie in [-c, c] while preserving the
    softmax distribution when the original range is <= 2c.

    Args:
        logits: shape (..., vocab_size). Raw logit vectors.
        c: clipping bound. Each component is mapped into [-c, c].

    Returns:
        Clipped logits with the same shape as input.
    """
    max_logits = logits.max(dim=-1, keepdim=True).values
    recentered = logits - max_logits + c
    return recentered.clamp(min=-c)


def clip_and_aggregate(logit_batch: Tensor, c: float, expected_batch_size: int) -> Tensor:
    """Clip each logit vector in a batch, then average them.

    Implements the aggregation step from Algorithm 1, line 16:
        z_bar = (1/s) * sum_{z in Z} clip_c(z)

    The division by expected_batch_size (s) instead of the actual batch size
    is intentional and required for the privacy analysis (sensitivity = c/s
    regardless of actual batch size).

    Args:
        logit_batch: shape (batch, vocab_size). Logit vectors from the LLM
            for each sensitive prompt in the batch.
        c: clipping bound.
        expected_batch_size: s, the expected batch size used in privacy accounting.

    Returns:
        Aggregated logit vector, shape (vocab_size,).
    """
    clipped = clip_logits(logit_batch, c)
    return clipped.sum(dim=0) / expected_batch_size
