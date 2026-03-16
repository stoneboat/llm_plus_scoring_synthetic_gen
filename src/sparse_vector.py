"""
Sparse Vector Technique (SVT) for public/private token switching.

When the public LLM prediction is close to the private (sensitive-data-informed)
prediction, we sample from the public distribution at zero privacy cost. Only
when they diverge do we pay privacy budget for a private token.

Reference: Algorithm 1 lines 13-21, and Eq. 2 from Amin et al. (2024).
"""

import torch
from torch import Tensor
from typing import Tuple


def compute_distribution_distance(
    private_logits_batch: Tensor,
    public_logits: Tensor,
    expected_batch_size: int,
) -> float:
    """Compute L1 distance between mean private and public token distributions.

    Implements Eq. 2:
        d(Z, z_public) = || (1/s) sum_{z in Z} softmax(z) - softmax(z_public) ||_1

    Args:
        private_logits_batch: shape (batch, vocab_size). Raw logits for each
            sensitive prompt.
        public_logits: shape (vocab_size,). Raw logits from the public prompt.
        expected_batch_size: s, the expected batch size (used for normalization).

    Returns:
        Scalar L1 distance.
    """
    private_probs = torch.softmax(private_logits_batch, dim=-1)
    mean_private_probs = private_probs.sum(dim=0) / expected_batch_size
    public_probs = torch.softmax(public_logits, dim=-1)
    return (mean_private_probs - public_probs).abs().sum().item()


def sample_noisy_threshold(threshold: float, noise_scale: float) -> float:
    """Sample a noisy threshold: theta_hat = theta + Laplace(sigma).

    Args:
        threshold: theta, the base threshold.
        noise_scale: sigma, the Laplace noise scale.

    Returns:
        Noisy threshold value.
    """
    noise = torch.distributions.Laplace(0.0, noise_scale).sample().item()
    return threshold + noise


def should_use_private_token(
    private_logits_batch: Tensor,
    public_logits: Tensor,
    expected_batch_size: int,
    noisy_threshold: float,
    noise_scale: float,
) -> Tuple[bool, float]:
    """Decide whether to use a private or public token via SVT.

    Computes the noisy distance and compares to the noisy threshold.
    If d_hat >= theta_hat, use a private token (costs privacy budget).
    Otherwise, use the public token (free).

    Args:
        private_logits_batch: shape (batch, vocab_size).
        public_logits: shape (vocab_size,).
        expected_batch_size: s.
        noisy_threshold: theta_hat (pre-sampled noisy threshold).
        noise_scale: sigma, for the query noise Laplace(2*sigma).

    Returns:
        Tuple of (use_private: bool, noisy_distance: float).
    """
    distance = compute_distribution_distance(
        private_logits_batch, public_logits, expected_batch_size
    )
    query_noise = torch.distributions.Laplace(0.0, 2.0 * noise_scale).sample().item()
    noisy_distance = distance + query_noise
    return noisy_distance >= noisy_threshold, noisy_distance
