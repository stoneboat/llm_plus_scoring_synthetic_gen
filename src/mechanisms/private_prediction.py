"""
Private-prediction mechanism (Algorithm 1, Amin et al. 2024).

Implements clipped-logit aggregation (exponential mechanism) with an optional
Sparse Vector Technique gate to skip the privacy cost for predictable tokens.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from src.backends.base import ModelBackend
from src.clip_utils import clip_and_aggregate
from src.sparse_vector import sample_noisy_threshold, should_use_private_token
from src.config import PrivacyConfig, GenerationConfig
from src.mechanisms.base import Mechanism


def _apply_top_k_filter(
    logits: Tensor,
    public_logits: Tensor,
    top_k: int,
) -> Tensor:
    """Restrict *logits* to the top-k tokens from the public prediction.

    Large-vocabulary models (e.g. Gemma-2 with 256K tokens) combined with
    logit clipping raise the probability floor of nonsense tokens.  Filtering
    to the public prediction's top-k tokens avoids sampling from that long
    tail.  The mask is a deterministic function of the public (non-sensitive)
    logits, so it does not affect the privacy guarantee.

    Reference: Appendix F.1 of Amin et al. (2024) — used at τ ≥ 2.25.
    """
    _, top_indices = public_logits.topk(top_k, dim=-1)
    mask = torch.full_like(logits, float("-inf"))
    mask[top_indices] = 0.0
    return logits + mask


class PrivatePredictionMechanism(Mechanism):
    """Algorithm 1 inner loop: clipped-logit aggregation + optional SVT gate.

    Args:
        privacy_config: privacy parameters (clip bound, temperature, SVT
            threshold/noise, delta).
        gen_config: generation parameters (batch size, top-k, EOS token).
    """

    def __init__(
        self,
        privacy_config: PrivacyConfig,
        gen_config: GenerationConfig,
    ) -> None:
        self._privacy_config = privacy_config
        self._gen_config = gen_config

    def generate_example(
        self,
        private_prompts: List[str],
        public_prompt: str,
        backend: ModelBackend,
        remaining_budget: int,
        max_total_tokens: int,
    ) -> Tuple[List[int], int, int]:
        """Generate one synthetic example (Algorithm 1, inner loop).

        Produces tokens until EOS, ``max_total_tokens``, or
        ``remaining_budget`` private tokens are exhausted — whichever comes
        first.

        Returns:
            ``(generated_token_ids, num_private_tokens, num_public_tokens)``
        """
        c = self._privacy_config.clip_bound
        tau = self._privacy_config.temperature
        tau_pub = self._privacy_config.public_temperature
        s = self._gen_config.batch_size
        eos_id = self._gen_config.eos_token_id
        top_k = self._gen_config.top_k_vocab

        svt_enabled = self._privacy_config.svt_enabled
        theta = self._privacy_config.svt_threshold
        sigma = self._privacy_config.svt_noise

        generated_tokens: List[int] = []
        private_token_count = 0
        public_token_count = 0

        noisy_thresh: Optional[float] = None
        if svt_enabled:
            noisy_thresh = sample_noisy_threshold(theta, sigma)

        while (
            private_token_count < remaining_budget
            and len(generated_tokens) < max_total_tokens
        ):
            private_logits = backend.get_next_token_logits(
                private_prompts, generated_tokens
            )

            if svt_enabled:
                public_logits = backend.get_next_token_logits(
                    [public_prompt], generated_tokens
                )[0]  # (vocab_size,)

                use_private, _ = should_use_private_token(
                    private_logits, public_logits, s, noisy_thresh, sigma
                )

                if use_private:
                    z_bar = clip_and_aggregate(private_logits, c, s)
                    if top_k > 0:
                        z_bar = _apply_top_k_filter(z_bar, public_logits, top_k)
                    probs = torch.softmax(z_bar / tau, dim=-1)
                    token_id = torch.multinomial(probs, num_samples=1).item()
                    private_token_count += 1
                    noisy_thresh = sample_noisy_threshold(theta, sigma)
                else:
                    pub_logits = public_logits
                    if top_k > 0:
                        pub_logits = _apply_top_k_filter(
                            pub_logits, public_logits, top_k
                        )
                    probs = torch.softmax(pub_logits / tau_pub, dim=-1)
                    token_id = torch.multinomial(probs, num_samples=1).item()
                    public_token_count += 1
            else:
                z_bar = clip_and_aggregate(private_logits, c, s)
                if top_k > 0:
                    public_logits = backend.get_next_token_logits(
                        [public_prompt], generated_tokens
                    )[0]
                    z_bar = _apply_top_k_filter(z_bar, public_logits, top_k)
                probs = torch.softmax(z_bar / tau, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                private_token_count += 1

            generated_tokens.append(token_id)

            if eos_id is not None and token_id == eos_id:
                break

        return generated_tokens, private_token_count, public_token_count
