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
"""

import math
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
from torch import Tensor

from src.clip_utils import clip_logits, clip_and_aggregate
from src.sparse_vector import (
    sample_noisy_threshold,
    should_use_private_token,
)
from src.privacy_accounting import (
    compute_max_private_tokens,
    compute_epsilon,
    privacy_report,
)
from src.config import PrivacyConfig, GenerationConfig, ModelConfig, PROMPT_TEMPLATES


@dataclass
class SyntheticExample:
    """A single generated synthetic example."""
    text: str
    label: int
    label_name: str
    num_private_tokens: int
    num_public_tokens: int
    num_total_tokens: int


def assign_to_batch(prompt_text: str, num_batches: int) -> int:
    """Assign a prompt to a batch using hashing (satisfies Assumption 1).

    The batch assignment depends only on the prompt itself, not on other prompts.
    """
    h = int(hashlib.sha256(prompt_text.encode()).hexdigest(), 16)
    return h % num_batches


def partition_by_label(
    examples: List[dict],
    label_column: str,
    batch_size: int,
) -> Dict[int, List[List[dict]]]:
    """Partition examples into batches, grouped by label.

    Returns a dict mapping label -> list of batches, where each batch
    is a list of examples of size approximately batch_size.
    """
    by_label: Dict[int, List[dict]] = {}
    for ex in examples:
        label = ex[label_column]
        by_label.setdefault(label, []).append(ex)

    batches_by_label: Dict[int, List[List[dict]]] = {}
    for label, label_examples in by_label.items():
        batches = []
        for i in range(0, len(label_examples), batch_size):
            batch = label_examples[i : i + batch_size]
            batches.append(batch)
        batches_by_label[label] = batches

    return batches_by_label


def build_prompts(
    examples: List[dict],
    dataset_name: str,
    text_column: str,
    label: int,
) -> Tuple[List[str], str]:
    """Build private prompts for a batch plus the corresponding public prompt.

    Returns:
        (private_prompts, public_prompt)
    """
    templates = PROMPT_TEMPLATES[dataset_name]
    label_name = templates["labels"][label]

    private_prompts = []
    for ex in examples:
        prompt = templates["private"].format(
            label=label_name, example=ex[text_column]
        )
        private_prompts.append(prompt)

    public_prompt = templates["public"].format(label=label_name, example="")

    return private_prompts, public_prompt


@torch.no_grad()
def get_next_token_logits(
    model,
    tokenizer,
    prompts: List[str],
    generated_tokens: List[int],
    device: str = "cuda",
) -> Tensor:
    """Run LLM inference to get next-token logits for a set of prompts.

    Each prompt is concatenated with the generated tokens so far, then
    we extract the logits for the next position.

    Args:
        model: HuggingFace causal LM.
        tokenizer: corresponding tokenizer.
        prompts: list of prompt strings.
        generated_tokens: token IDs generated so far (shared across all prompts).
        device: torch device.

    Returns:
        Logits tensor of shape (len(prompts), vocab_size).
    """
    if generated_tokens:
        suffix = tokenizer.decode(generated_tokens)
        full_texts = [p + suffix for p in prompts]
    else:
        full_texts = prompts

    inputs = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    outputs = model(**inputs)
    # Extract logits at the last position for each sequence
    last_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
    return last_logits


@torch.no_grad()
def generate_one_example(
    model,
    tokenizer,
    private_prompts: List[str],
    public_prompt: str,
    privacy_config: PrivacyConfig,
    gen_config: GenerationConfig,
    device: str = "cuda",
) -> Tuple[List[int], int, int]:
    """Generate one synthetic example from a batch of sensitive prompts.

    Implements the inner loop of Algorithm 1.

    Args:
        model: HuggingFace causal LM.
        tokenizer: tokenizer.
        private_prompts: prompts containing sensitive data.
        public_prompt: prompt without sensitive data.
        privacy_config: privacy parameters (c, tau, theta, sigma).
        gen_config: generation parameters (batch_size, max tokens).
        device: torch device.

    Returns:
        (generated_token_ids, num_private_tokens, num_public_tokens)
    """
    c = privacy_config.clip_bound
    tau = privacy_config.temperature
    tau_pub = privacy_config.public_temperature
    s = gen_config.batch_size
    r = gen_config.max_private_tokens
    eos_id = gen_config.eos_token_id

    svt_enabled = privacy_config.svt_enabled
    theta = privacy_config.svt_threshold
    sigma = privacy_config.svt_noise

    generated_tokens: List[int] = []
    private_token_count = 0
    public_token_count = 0

    # Initialize noisy threshold for SVT
    noisy_thresh = None
    if svt_enabled:
        noisy_thresh = sample_noisy_threshold(theta, sigma)

    while private_token_count < r and len(generated_tokens) < gen_config.max_total_tokens:
        # Get logits from all private prompts
        private_logits = get_next_token_logits(
            model, tokenizer, private_prompts, generated_tokens, device
        )

        if svt_enabled:
            # Get logits from public prompt
            public_logits = get_next_token_logits(
                model, tokenizer, [public_prompt], generated_tokens, device
            )[0]  # (vocab_size,)

            # SVT: decide private vs public
            use_private, _ = should_use_private_token(
                private_logits, public_logits, s, noisy_thresh, sigma
            )

            if use_private:
                # Private token: clip, aggregate, sample
                z_bar = clip_and_aggregate(private_logits, c, s)
                probs = torch.softmax(z_bar / tau, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                private_token_count += 1
                # Re-sample noisy threshold for next SVT query
                noisy_thresh = sample_noisy_threshold(theta, sigma)
            else:
                # Public token: sample from public distribution (free)
                probs = torch.softmax(public_logits / tau_pub, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                public_token_count += 1
        else:
            # No SVT: every token is private
            z_bar = clip_and_aggregate(private_logits, c, s)
            probs = torch.softmax(z_bar / tau, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).item()
            private_token_count += 1

        generated_tokens.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break

    return generated_tokens, private_token_count, public_token_count


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

    Returns:
        List of SyntheticExample objects.
    """
    templates = PROMPT_TEMPLATES[dataset_name]

    # Set EOS token
    if gen_config.eos_token_id is None:
        gen_config.eos_token_id = tokenizer.eos_token_id

    # Compute privacy budget
    delta = privacy_config.delta
    if delta is None:
        delta = 1.0 / len(examples)
        privacy_config.delta = delta

    # Partition into batches by label
    batches_by_label = partition_by_label(examples, label_column, gen_config.batch_size)

    total_batches = sum(len(bs) for bs in batches_by_label.values())
    if verbose:
        report = privacy_report(
            gen_config.max_private_tokens,
            privacy_config.clip_bound,
            gen_config.batch_size,
            privacy_config.temperature,
            delta,
            privacy_config.svt_noise if privacy_config.svt_enabled else None,
        )
        print(f"Privacy report: epsilon={report['epsilon']:.4f}, delta={delta:.2e}")
        print(f"  rho/token={report['rho_per_token']:.6f}, "
              f"total_rho={report['total_rho']:.6f}")
        print(f"  {total_batches} batches, batch_size={gen_config.batch_size}")
        print()

    synthetic_data: List[SyntheticExample] = []
    batch_idx = 0

    for label, batches in batches_by_label.items():
        label_name = templates["labels"][label]
        for batch in batches:
            batch_idx += 1
            if verbose:
                print(f"  Batch {batch_idx}/{total_batches} "
                      f"(label={label_name}, size={len(batch)})")

            private_prompts, public_prompt = build_prompts(
                batch, dataset_name, text_column, label
            )

            token_ids, n_priv, n_pub = generate_one_example(
                model, tokenizer,
                private_prompts, public_prompt,
                privacy_config, gen_config,
                device=device,
            )

            text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

            synthetic_data.append(SyntheticExample(
                text=text,
                label=label,
                label_name=label_name,
                num_private_tokens=n_priv,
                num_public_tokens=n_pub,
                num_total_tokens=len(token_ids),
            ))

            if verbose:
                pub_frac = n_pub / max(1, n_priv + n_pub) * 100
                print(f"    -> {len(token_ids)} tokens "
                      f"({n_priv} private, {n_pub} public = {pub_frac:.0f}% free)")

    if verbose:
        total_priv = sum(e.num_private_tokens for e in synthetic_data)
        total_pub = sum(e.num_public_tokens for e in synthetic_data)
        print(f"\nGenerated {len(synthetic_data)} synthetic examples")
        print(f"Total tokens: {total_priv} private, {total_pub} public")
        actual_eps = compute_epsilon(
            max(e.num_private_tokens for e in synthetic_data),
            privacy_config.clip_bound,
            gen_config.batch_size,
            privacy_config.temperature,
            delta,
            privacy_config.svt_noise if privacy_config.svt_enabled else None,
        )
        print(f"Actual epsilon (worst-case batch): {actual_eps:.4f}")

    return synthetic_data
