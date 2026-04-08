"""
Core Algorithm 1: Private synthetic text generation via private prediction.

Generates differentially private synthetic text by:
1. Partitioning sensitive prompts into disjoint batches (by label)
2. Running LLM inference on each batch in parallel
3. Clipping and aggregating logits across the batch
4. Sampling tokens via softmax (= exponential mechanism)
5. Using the sparse vector technique to skip privacy cost for predictable tokens

Reference: Algorithm 1 in Amin et al. (2024), "Private prediction for large-scale
synthetic text generation", Findings of EMNLP 2024.

Phase 2 note: BatchDescriptor, assign_to_batch, partition_by_label, build_prompts,
and _format_prompt now live in src/batching/ and src/prompts/ respectively.
They are re-exported here so that existing call sites continue to work unchanged.

Phase 3 note: get_next_token_logits, _apply_top_k_filter, and
_generate_single_example now live in src/backends/ and src/mechanisms/
respectively.  They are re-exported here for backward compatibility.
generate_synthetic_dataset and generate_batch_examples now delegate to
HuggingFaceCausalLM + PrivatePredictionMechanism internally.

Phase 4 note: SyntheticExample, _run_batch_generation, and the dataset
orchestration loop now live in src/runtime/generation.py.  They are
re-exported here for backward compatibility.  generate_synthetic_dataset
delegates to run_dataset_generation after constructing the backend and
mechanism.
"""

from typing import Callable, List, Optional, Set, Tuple

import torch
from torch import Tensor

from src.privacy_accounting import compute_max_private_tokens
from src.config import PrivacyConfig, GenerationConfig, ModelConfig

# ---------------------------------------------------------------------------
# Phase 2 imports: batching and prompt layers
# ---------------------------------------------------------------------------
from src.batching.base import BatchDescriptor, BatchingPolicy          # noqa: F401
from src.batching.hash_label_policy import (                           # noqa: F401
    assign_to_batch,
    partition_by_label,
    HashLabelBatchingPolicy,
)
from src.prompts.text_classification import (                          # noqa: F401
    PROMPT_TEMPLATES,
    build_prompts,
    _format_prompt,
)

# ---------------------------------------------------------------------------
# Phase 3 imports: backend and mechanism layers
# ---------------------------------------------------------------------------
from src.backends.huggingface_causal_lm import HuggingFaceCausalLM    # noqa: F401
from src.mechanisms.private_prediction import (                        # noqa: F401
    PrivatePredictionMechanism,
    _apply_top_k_filter,
)

# ---------------------------------------------------------------------------
# Phase 4 imports: runtime layer
# ---------------------------------------------------------------------------
from src.runtime.generation import (                                   # noqa: F401
    SyntheticExample,
    run_batch_generation,
    run_dataset_generation,
)

# Backward-compat alias used internally and re-exported.
_run_batch_generation = run_batch_generation


# ---------------------------------------------------------------------------
# Backward-compat wrappers for functions moved to Phase 3 modules
# ---------------------------------------------------------------------------

@torch.no_grad()
def get_next_token_logits(
    model,
    tokenizer,
    prompts: List[str],
    generated_tokens: List[int],
    device: str = "cuda",
    micro_batch_size: int = 32,
) -> Tensor:
    """Backward-compat wrapper.  See HuggingFaceCausalLM.get_next_token_logits."""
    backend = HuggingFaceCausalLM(model, tokenizer, device=device,
                                  micro_batch_size=micro_batch_size)
    return backend.get_next_token_logits(prompts, generated_tokens)


@torch.no_grad()
def _generate_single_example(
    model,
    tokenizer,
    private_prompts: List[str],
    public_prompt: str,
    privacy_config: PrivacyConfig,
    gen_config: GenerationConfig,
    remaining_private_budget: int,
    device: str = "cuda",
    micro_batch_size: int = 32,
) -> Tuple[List[int], int, int]:
    """Backward-compat wrapper.  See PrivatePredictionMechanism.generate_example."""
    backend = HuggingFaceCausalLM(model, tokenizer, device=device,
                                  micro_batch_size=micro_batch_size)
    mechanism = PrivatePredictionMechanism(privacy_config, gen_config)
    return mechanism.generate_example(
        private_prompts, public_prompt, backend,
        remaining_budget=remaining_private_budget,
        max_total_tokens=gen_config.max_total_tokens,
    )


@torch.no_grad()
def generate_batch_examples(
    model,
    tokenizer,
    private_prompts: List[str],
    public_prompt: str,
    privacy_config: PrivacyConfig,
    gen_config: GenerationConfig,
    device: str = "cuda",
    micro_batch_size: int = 32,
) -> List[Tuple[List[int], int, int]]:
    """Generate synthetic examples from one batch (Algorithm 1, outer loop).

    The per-batch privacy budget *r* (``gen_config.max_private_tokens``)
    is shared across all examples produced from this batch.  Each example
    starts from a clean prompt context, generates until EOS or the
    per-example length cap, and then the next example begins with the
    remaining budget.

    Returns:
        List of (token_ids, num_private, num_public) tuples — one per
        synthetic example produced from this batch.
    """
    backend = HuggingFaceCausalLM(model, tokenizer, device=device,
                                  micro_batch_size=micro_batch_size)
    mechanism = PrivatePredictionMechanism(privacy_config, gen_config)
    return run_batch_generation(mechanism, backend, private_prompts, public_prompt,
                                gen_config)


def generate_one_example(
    model,
    tokenizer,
    private_prompts: List[str],
    public_prompt: str,
    privacy_config: PrivacyConfig,
    gen_config: GenerationConfig,
    device: str = "cuda",
    micro_batch_size: int = 32,
) -> Tuple[List[int], int, int]:
    """Convenience wrapper: generate a single example using the full budget.

    Kept for backward compatibility.  For the correct Algorithm 1
    behaviour (multiple examples per batch), use
    :func:`generate_batch_examples` instead.
    """
    return _generate_single_example(
        model, tokenizer,
        private_prompts, public_prompt,
        privacy_config, gen_config,
        remaining_private_budget=gen_config.max_private_tokens,
        device=device,
        micro_batch_size=micro_batch_size,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    model,
    tokenizer,
    examples: List[dict],
    dataset_name: str,
    privacy_config: PrivacyConfig,
    gen_config: GenerationConfig,
    text_column: str = "text",
    label_column: str = "label",
    device: str = "cuda",
    verbose: bool = True,
    micro_batch_size: int = 32,
    completed_batch_ids: Optional[Set[str]] = None,
    batch_callback: Optional[Callable[[BatchDescriptor, List[SyntheticExample]], None]] = None,
) -> List[SyntheticExample]:
    """Generate a full synthetic dataset using Algorithm 1.

    Args:
        model: HuggingFace causal LM.
        tokenizer: tokenizer.
        examples: list of dicts with text and label columns.
        dataset_name: name matching a key in PROMPT_TEMPLATES.
        privacy_config: privacy parameters.
        gen_config: generation parameters.
        text_column: column name for text in examples.
        label_column: column name for label in examples.
        device: torch device.
        verbose: print progress.
        completed_batch_ids: optional set of stable batch IDs to skip.
        batch_callback: optional callback invoked after each completed batch.

    Returns:
        List of SyntheticExample objects.
    """
    # Set EOS token before constructing the mechanism (which reads gen_config).
    if gen_config.eos_token_id is None:
        gen_config.eos_token_id = tokenizer.eos_token_id

    # Build backend and mechanism once; shared across all batches.
    backend = HuggingFaceCausalLM(model, tokenizer, device=device,
                                  micro_batch_size=micro_batch_size)
    mechanism = PrivatePredictionMechanism(privacy_config, gen_config)

    # Causal LMs require left-padding so that position -1 is always the
    # last real token for every sequence in the batch.
    orig_padding_side = backend.padding_side
    backend.padding_side = "left"

    # Resolve delta (mutates privacy_config for downstream callers, preserved
    # from pre-Phase-4 behaviour).
    delta = privacy_config.delta
    if delta is None:
        delta = 1.0 / len(examples)
        privacy_config.delta = delta

    result = run_dataset_generation(
        backend=backend,
        mechanism=mechanism,
        examples=examples,
        dataset_name=dataset_name,
        gen_config=gen_config,
        privacy_config=privacy_config,
        delta=delta,
        text_column=text_column,
        label_column=label_column,
        completed_batch_ids=completed_batch_ids,
        batch_callback=batch_callback,
        verbose=verbose,
    )

    backend.padding_side = orig_padding_side
    return result
