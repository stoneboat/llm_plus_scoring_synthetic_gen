"""
Runtime / orchestration for Algorithm 1: Private synthetic text generation.

Phase 4: extracted from src/generate.py.

Owns:
- SyntheticExample dataclass (the per-example generation result)
- run_batch_generation — Algorithm 1 outer loop for one batch:
      repeatedly calls mechanism.generate_example until the token budget
      is exhausted or the stopping condition fires.
- run_dataset_generation — dataset-level orchestration:
      partitions examples into batches, skips completed batches,
      builds prompts, calls run_batch_generation, decodes tokens,
      invokes the batch callback, and tracks aggregated statistics.

No changes to algorithm behavior, prompt strings, JSONL schema, or batch IDs
relative to Phase 3/3.5b.  This is a pure reorganisation.
"""

import hashlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch

from src.config import GenerationConfig, PrivacyConfig
from src.privacy_accounting import compute_epsilon, privacy_report
from src.batching.base import BatchDescriptor
from src.batching.hash_label_policy import partition_by_label
from src.prompts.text_classification import PROMPT_TEMPLATES, build_prompts
from src.backends.huggingface_causal_lm import HuggingFaceCausalLM
from src.mechanisms.private_prediction import PrivatePredictionMechanism


@dataclass
class SyntheticExample:
    """A single generated synthetic example."""
    text: str
    label: int
    label_name: str
    num_private_tokens: int
    num_public_tokens: int
    num_total_tokens: int


def run_batch_generation(
    mechanism: PrivatePredictionMechanism,
    backend: HuggingFaceCausalLM,
    private_prompts: List[str],
    public_prompt: str,
    gen_config: GenerationConfig,
) -> List[Tuple[List[int], int, int]]:
    """Run the Algorithm 1 outer loop for one batch.

    Repeatedly calls ``mechanism.generate_example`` until the batch's
    private-token budget *r* is exhausted or the example cap is reached.

    Stopping conditions (in priority order):
    1. ``private_tokens_used >= r`` — budget exhausted.
    2. ``len(results) >= max_examples`` — example cap.
    3. ``n_priv == 0 and n_pub == 0`` — mechanism returned empty output.
    4. ``consecutive_no_private >= 3`` — three consecutive public-only
       examples; generation has stalled.

    Args:
        mechanism: configured PrivatePredictionMechanism.
        backend: configured HuggingFaceCausalLM.
        private_prompts: per-batch private context strings.
        public_prompt: shared public context string.
        gen_config: generation parameters (max_private_tokens, max_total_tokens).

    Returns:
        List of (token_ids, num_private, num_public) tuples — one per
        synthetic example produced from this batch.
    """
    r = gen_config.max_private_tokens
    max_examples = max(r, 10)
    results: List[Tuple[List[int], int, int]] = []
    private_tokens_used = 0
    consecutive_no_private = 0

    while private_tokens_used < r and len(results) < max_examples:
        remaining = r - private_tokens_used

        token_ids, n_priv, n_pub = mechanism.generate_example(
            private_prompts, public_prompt, backend,
            remaining_budget=remaining,
            max_total_tokens=gen_config.max_total_tokens,
        )

        private_tokens_used += n_priv

        if token_ids:
            results.append((token_ids, n_priv, n_pub))

        if n_priv == 0 and n_pub == 0:
            break

        if n_priv == 0:
            consecutive_no_private += 1
            if consecutive_no_private >= 3:
                break
        else:
            consecutive_no_private = 0

    return results


def run_dataset_generation(
    backend: HuggingFaceCausalLM,
    mechanism: PrivatePredictionMechanism,
    examples: List[dict],
    dataset_name: str,
    gen_config: GenerationConfig,
    privacy_config: PrivacyConfig,
    delta: float,
    text_column: str = "text",
    label_column: str = "label",
    completed_batch_ids: Optional[Set[str]] = None,
    batch_callback: Optional[Callable[
        [BatchDescriptor, List[SyntheticExample]], None
    ]] = None,
    verbose: bool = True,
) -> List[SyntheticExample]:
    """Dataset-level orchestration loop for Algorithm 1.

    Partitions *examples* into fixed disjoint batches by label, skips any
    batch whose stable ID is already in *completed_batch_ids*, builds prompts,
    runs ``run_batch_generation``, decodes tokens into ``SyntheticExample``
    objects, and invokes *batch_callback* after each completed batch.

    The stable batch ID is the SHA-256 hex digest of
    ``"<dataset_name>\\n<label>\\n<newline-joined batch texts>"``,
    exactly matching the computation used before Phase 4.

    Args:
        backend: configured HuggingFaceCausalLM (padding_side already set).
        mechanism: configured PrivatePredictionMechanism.
        examples: source examples (dicts with text and label columns).
        dataset_name: key into PROMPT_TEMPLATES.
        gen_config: generation parameters.
        privacy_config: privacy parameters.
        delta: resolved delta (must not be None).
        text_column: column name for text in examples.
        label_column: column name for label in examples.
        completed_batch_ids: stable batch IDs to skip (resume support).
        batch_callback: called with (descriptor, batch_examples) after each
            completed batch; used by callers for checkpointing.
        verbose: print per-batch progress.

    Returns:
        List of SyntheticExample objects (only from newly generated batches;
        caller is responsible for merging with resumed examples).
    """
    templates = PROMPT_TEMPLATES[dataset_name]
    completed_batch_ids = completed_batch_ids or set()

    batches_by_label = partition_by_label(
        examples, label_column, text_column, gen_config.batch_size
    )
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
    max_batch_private_tokens = 0

    for label, batches in batches_by_label.items():
        label_name = templates["labels"][label]
        for batch in batches:
            batch_idx += 1
            batch_key = "\n".join(ex[text_column] for ex in batch)
            batch_id = hashlib.sha256(
                f"{dataset_name}\n{label}\n{batch_key}".encode("utf-8")
            ).hexdigest()
            descriptor = BatchDescriptor(
                batch_id=batch_id,
                batch_index=batch_idx,
                total_batches=total_batches,
                label=label,
                label_name=label_name,
                batch_size=len(batch),
            )

            if batch_id in completed_batch_ids:
                if verbose:
                    print(f"  Batch {batch_idx}/{total_batches} "
                          f"(label={label_name}, size={len(batch)}) [resume: skipped]")
                continue

            if verbose:
                print(f"  Batch {batch_idx}/{total_batches} "
                      f"(label={label_name}, size={len(batch)})")

            private_prompts, public_prompt = build_prompts(
                batch, dataset_name, text_column, label,
                tokenizer=backend.tokenizer,
            )

            batch_results = run_batch_generation(
                mechanism, backend,
                private_prompts, public_prompt,
                gen_config,
            )

            batch_examples: List[SyntheticExample] = []
            batch_priv_total = 0
            batch_pub_total = 0

            for token_ids, n_priv, n_pub in batch_results:
                text = backend.decode(token_ids, skip_special_tokens=True).strip()
                example = SyntheticExample(
                    text=text,
                    label=label,
                    label_name=label_name,
                    num_private_tokens=n_priv,
                    num_public_tokens=n_pub,
                    num_total_tokens=len(token_ids),
                )
                synthetic_data.append(example)
                batch_examples.append(example)
                batch_priv_total += n_priv
                batch_pub_total += n_pub

            max_batch_private_tokens = max(
                max_batch_private_tokens, batch_priv_total,
            )

            if batch_callback is not None:
                batch_callback(descriptor, batch_examples)

            if verbose:
                n_ex = len(batch_results)
                pub_frac = (batch_pub_total
                            / max(1, batch_priv_total + batch_pub_total) * 100)
                print(f"    -> {n_ex} example(s), "
                      f"{batch_priv_total} private + "
                      f"{batch_pub_total} public tokens "
                      f"({pub_frac:.0f}% free)")

    if verbose:
        total_priv = sum(e.num_private_tokens for e in synthetic_data)
        total_pub = sum(e.num_public_tokens for e in synthetic_data)
        print(f"\nGenerated {len(synthetic_data)} synthetic examples "
              f"from {total_batches} batches")
        print(f"Total tokens: {total_priv} private, {total_pub} public")
        actual_eps = compute_epsilon(
            max_batch_private_tokens,
            privacy_config.clip_bound,
            gen_config.batch_size,
            privacy_config.temperature,
            delta,
            privacy_config.svt_noise if privacy_config.svt_enabled else None,
        )
        print(f"Actual epsilon (worst-case batch, "
              f"{max_batch_private_tokens} private tokens): {actual_eps:.4f}")

    return synthetic_data
