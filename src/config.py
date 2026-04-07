"""
Hyperparameters and configuration for private prediction synthetic text generation.

Reference: Table 7 in Amin et al. (2024), "Private prediction for large-scale
synthetic text generation", Findings of EMNLP 2024.

Phase 2 note: PROMPT_TEMPLATES was moved to src/prompts/text_classification.py.
It is re-exported here so that existing ``from src.config import PROMPT_TEMPLATES``
imports continue to work unchanged.

Phase 3 note: compute_max_private_tokens now delegates to the authoritative
implementation in src/privacy_accounting.  The argument order here differs from
that module (batch_size before clip_bound) — the wrapper preserves the original
call-site contract.
"""

from dataclasses import dataclass, field
from typing import Optional

# Backward-compatible re-export: PROMPT_TEMPLATES now lives in src/prompts/.
from src.prompts.text_classification import PROMPT_TEMPLATES  # noqa: F401


@dataclass
class PrivacyConfig:
    """Privacy budget and mechanism parameters."""

    target_epsilon: float = 1.0
    delta: Optional[float] = None  # defaults to 1/n at runtime
    clip_bound: float = 10.0       # c: logit clipping range [-c, c]
    temperature: float = 2.0       # tau: sampling temperature for private tokens
    public_temperature: float = 1.5  # tau_public: temperature for public tokens

    # Sparse vector technique (SVT) parameters
    svt_threshold: float = 0.5   # theta: distance threshold for pub/priv switching
    svt_noise: float = 0.2       # sigma: Laplace noise scale for SVT

    # Set svt_threshold = -inf to disable SVT (all tokens are private)
    @property
    def svt_enabled(self) -> bool:
        return self.svt_threshold != float("-inf")


@dataclass
class GenerationConfig:
    """Parameters controlling the generation process."""

    batch_size: int = 255           # s: number of sensitive prompts per batch
    max_private_tokens: int = 100   # r: max private tokens per synthetic example
    max_total_tokens: int = 256     # absolute cap on output length (private + public)
    eos_token_id: Optional[int] = None  # set from tokenizer at runtime
    top_k_vocab: int = 0            # restrict sampling to top-k of public logits (0 = off)


@dataclass
class ModelConfig:
    """LLM model configuration."""

    model_name_or_path: str = "data/models/gemma-2-2b-it"
    hf_model_id: str = "google/gemma-2-2b-it"
    torch_dtype: str = "auto"
    device: str = "cuda"


@dataclass
class DatasetConfig:
    """Dataset and prompt configuration."""

    dataset_name: str = "agnews"
    cache_dir: str = "data/datasets"
    num_examples: Optional[int] = None  # None = use full dataset
    label_column: str = "label"
    text_column: str = "text"


# Pre-defined hyperparameter sweep grid from Table 7
HYPERPARAM_GRID = {
    "batch_size": [127, 255, 511, 1023, 1535, 2047],
    "clip_bound": [10],
    "temperature": [1.5, 2.0, 2.25],
    "svt_threshold": [float("-inf"), 0.3, 0.5, 0.7],
    "svt_noise": [None, 0.1, 0.2, 0.3],  # None pairs with -inf threshold
    "public_temperature": [1.5],
}

# Paired SVT settings: (threshold, noise)
SVT_SETTINGS = [
    (float("-inf"), None),   # no SVT
    (0.3, 0.1),
    (0.5, 0.2),
    (0.7, 0.3),
]


def compute_max_private_tokens(
    target_epsilon: float,
    delta: float,
    batch_size: int,       # NOTE: order differs from privacy_accounting version
    clip_bound: float,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> int:
    """Backward-compat wrapper around src/privacy_accounting.compute_max_private_tokens.

    The argument order here (batch_size before clip_bound) is preserved for
    existing call sites.  The authoritative implementation is in
    :func:`src.privacy_accounting.compute_max_private_tokens`.
    """
    from src.privacy_accounting import (
        compute_max_private_tokens as _authoritative,
    )
    # privacy_accounting signature: (target_epsilon, delta, clip_bound, batch_size, ...)
    return _authoritative(target_epsilon, delta, clip_bound, batch_size,
                          temperature, svt_noise)
