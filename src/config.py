"""
Hyperparameters and configuration for private prediction synthetic text generation.

Reference: Table 7 in Amin et al. (2024), "Private prediction for large-scale
synthetic text generation", Findings of EMNLP 2024.
"""

from dataclasses import dataclass, field
from typing import Optional
import math


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


# Prompt templates from Appendix F of the paper.
#
# Each dataset has:
#   user_message  – the instruction shown to the LLM (with {label}/{example})
#   response_prefix – the first tokens of the model's response that the
#                     generated text should continue from (e.g. "Text:")
#   labels        – mapping from integer label to human-readable name
#
# build_prompts() in generate.py wraps these in the model's native chat
# template via tokenizer.apply_chat_template so that IT models get proper
# role markers (<start_of_turn>, etc.) instead of raw "# [User]" text.
PROMPT_TEMPLATES = {
    "agnews": {
        "user_message": (
            "Here are texts with News Type: {label}.\n\n"
            "Text: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Text:",
        "labels": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
    },
    "trec": {
        "user_message": (
            "Here are questions with Answer Type: {label}.\n\n"
            "```\nText: {example}\n```\n\n"
            "Please give me another one."
        ),
        "response_prefix": "```\nQuestion:",
        "labels": {
            0: "Abbreviation", 1: "Entity", 2: "Description",
            3: "Human", 4: "Location", 5: "Number",
        },
    },
    "dbpedia": {
        "user_message": (
            "Here are entries of Category: {label}.\n\n"
            "Entry: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Entry:",
        "labels": {
            0: "Company", 1: "School", 2: "Artist", 3: "Athlete",
            4: "Politician", 5: "Transportation", 6: "Building",
            7: "Nature", 8: "Village", 9: "Animal", 10: "Plant",
            11: "Album", 12: "Film", 13: "Book",
        },
    },
    "imdb": {
        "user_message": (
            "Here are texts with Sentiment: {label}.\n\n"
            "Text: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Text:",
        "labels": {0: "Negative", 1: "Positive"},
    },
    "yelp": {
        "user_message": (
            "Here are texts with Sentiment: {label}.\n\n"
            "Text: {example}\n\n"
            "Please give me another one."
        ),
        "response_prefix": "Text:",
        "labels": {0: "Negative", 1: "Positive"},
    },
}


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
    batch_size: int,
    clip_bound: float,
    temperature: float,
    svt_noise: Optional[float] = None,
) -> int:
    """Estimate maximum private tokens r for a given privacy budget.

    Uses the simplified formula from Theorem 1:
        epsilon = rho + sqrt(4 * rho * log(1/delta))
    where rho = r * rho_per_token.
    """
    rho_exp = 0.5 * (clip_bound / (batch_size * temperature)) ** 2
    rho_svt = 0.0
    if svt_noise is not None and svt_noise > 0:
        rho_svt = 2.0 / (batch_size * svt_noise) ** 2
    rho_per_token = rho_exp + rho_svt

    if rho_per_token <= 0:
        return 10000

    # Solve: eps = r*rho_per_token + sqrt(4 * r*rho_per_token * log(1/delta))
    # Let x = r * rho_per_token (total rho). Solve eps = x + sqrt(4*x*log(1/delta))
    # Rearranging: (eps - x)^2 = 4*x*log(1/delta) => x^2 - 2*eps*x + eps^2 = 4*x*L
    # => x^2 - (2*eps + 4*L)*x + eps^2 = 0
    log_inv_delta = math.log(1.0 / delta)
    a = 1.0
    b = -(2.0 * target_epsilon + 4.0 * log_inv_delta)
    c_coeff = target_epsilon ** 2
    discriminant = b ** 2 - 4.0 * a * c_coeff
    if discriminant < 0:
        return 1
    total_rho = (-b - math.sqrt(discriminant)) / (2.0 * a)
    if total_rho <= 0:
        return 1

    r = int(total_rho / rho_per_token)
    return max(1, r)
