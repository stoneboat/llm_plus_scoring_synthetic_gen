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
    text_column: str,
    batch_size: int,
) -> Dict[int, List[List[dict]]]:
    """Partition examples into stable, disjoint batches grouped by label.

    Returns a dict mapping label -> list of batches, where each batch
    is a list of examples of size approximately batch_size.

    The assignment is hash-based rather than slice-based so a record's batch
    membership depends only on that record. This is the fixed-assignment
    property needed for the paper's parallel-composition argument: adding or
    removing one example affects only the batch containing that example, rather
    than shifting many later examples between batches.
    """
    by_label: Dict[int, List[dict]] = {}
    for ex in examples:
        label = ex[label_column]
        by_label.setdefault(label, []).append(ex)

    batches_by_label: Dict[int, List[List[dict]]] = {}
    for label, label_examples in by_label.items():
        num_batches = max(1, math.ceil(len(label_examples) / batch_size))
        buckets: List[List[dict]] = [[] for _ in range(num_batches)]

        for ex in label_examples:
            # Include the label in the hash key so identical texts from
            # different labels cannot collide into the same label-local bucket.
            key = f"{label}\n{ex[text_column]}"
            batch_idx = assign_to_batch(key, num_batches)
            buckets[batch_idx].append(ex)

        # Drop empty buckets and sort deterministically for reproducibility.
        non_empty_batches = []
        for bucket in buckets:
            if not bucket:
                continue
            bucket.sort(key=lambda ex: ex[text_column])
            non_empty_batches.append(bucket)

        batches_by_label[label] = non_empty_batches

    return batches_by_label


def _format_prompt(tokenizer, user_content: str, response_prefix: str) -> str:
    """Wrap a user message in the model's native chat template.

    Uses tokenizer.apply_chat_template so that instruction-tuned models
    receive proper role markers (e.g. <start_of_turn>user / model for
    Gemma, [INST] for Llama, etc.) instead of raw text markers.

    The leading <bos> emitted by some templates is stripped because the
    tokenizer already adds one during tokenization.
    """
    messages = [{"role": "user", "content": user_content}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if formatted.startswith("<bos>"):
        formatted = formatted[len("<bos>"):]
    return formatted + response_prefix


def build_prompts(
    examples: List[dict],
    dataset_name: str,
    text_column: str,
    label: int,
    tokenizer=None,
) -> Tuple[List[str], str]:
    """Build private prompts for a batch plus the corresponding public prompt.

    When *tokenizer* is provided the prompts are wrapped in the model's
    chat template.  This is strongly recommended for instruction-tuned
    models (Gemma IT, Llama Instruct, …) so that the model properly
    enters response mode instead of treating role markers as plain text.

    Returns:
        (private_prompts, public_prompt)
    """
    templates = PROMPT_TEMPLATES[dataset_name]
    label_name = templates["labels"][label]
    user_msg_tpl = templates["user_message"]
    response_prefix = templates["response_prefix"]

    private_prompts = []
    for ex in examples:
        user_content = user_msg_tpl.format(
            label=label_name, example=ex[text_column],
        )
        if tokenizer is not None:
            prompt = _format_prompt(tokenizer, user_content, response_prefix)
        else:
            prompt = f"# [User]\n{user_content}\n\n# [Assistant]\n{response_prefix}"
        private_prompts.append(prompt)

    public_user = user_msg_tpl.format(label=label_name, example="")
    if tokenizer is not None:
        public_prompt = _format_prompt(tokenizer, public_user, response_prefix)
    else:
        public_prompt = f"# [User]\n{public_user}\n\n# [Assistant]\n{response_prefix}"

    return private_prompts, public_prompt


def _apply_top_k_filter(
    logits: Tensor,
    public_logits: Tensor,
    top_k: int,
) -> Tensor:
    """Restrict logits to the top-k tokens from the public prediction.

    Large-vocabulary models (e.g. Gemma with 256K tokens) combined with
    logit clipping raise the probability floor of nonsense tokens.
    Filtering to the public prediction's top-k tokens avoids sampling
    from that long tail.  The mask is a deterministic function of the
    public (non-sensitive) logits, so it does not affect the privacy
    guarantee.

    Reference: Appendix F.1 of Amin et al. (2024) — used at τ ≥ 2.25.
    """
    _, top_indices = public_logits.topk(top_k, dim=-1)
    mask = torch.full_like(logits, float("-inf"))
    mask[top_indices] = 0.0
    return logits + mask


@torch.no_grad()
def get_next_token_logits(
    model,
    tokenizer,
    prompts: List[str],
    generated_tokens: List[int],
    device: str = "cuda",
    micro_batch_size: int = 32,
) -> Tensor:
    """Run LLM inference to get next-token logits for a set of prompts.

    Each prompt is concatenated with the generated tokens so far, then
    we extract the logits for the next position.

    IMPORTANT: The caller must set ``tokenizer.padding_side = "left"``
    before invoking this function.  Causal LMs require left-padding so
    that position -1 always corresponds to the last real token for every
    sequence in the batch.

    Processes prompts in micro-batches to avoid GPU OOM when batch_size
    is large (e.g. 255) and the model has a large vocabulary (e.g. Gemma 2
    with 256k tokens), since the logits tensor would be
    (batch × seq_len × vocab_size) which can exceed GPU memory.

    Args:
        model: HuggingFace causal LM.
        tokenizer: corresponding tokenizer (padding_side must be "left").
        prompts: list of prompt strings.
        generated_tokens: token IDs generated so far (shared across all prompts).
        device: torch device.
        micro_batch_size: number of prompts to forward at once. Reduce if OOM.

    Returns:
        Logits tensor of shape (len(prompts), vocab_size).
    """
    all_last_logits: List[Tensor] = []

    for i in range(0, len(prompts), micro_batch_size):
        chunk = prompts[i : i + micro_batch_size]
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Preserve exact continuation tokens by appending token IDs directly.
        # Decoding then re-tokenizing can alter whitespace/subword boundaries.
        if generated_tokens:
            batch_len = inputs["input_ids"].shape[0]
            gen = torch.tensor(
                generated_tokens, dtype=inputs["input_ids"].dtype, device=device
            ).unsqueeze(0).expand(batch_len, -1)
            gen_attn = torch.ones(
                (batch_len, gen.shape[1]),
                dtype=inputs["attention_mask"].dtype,
                device=device,
            )
            inputs = {
                "input_ids": torch.cat([inputs["input_ids"], gen], dim=1),
                "attention_mask": torch.cat([inputs["attention_mask"], gen_attn], dim=1),
            }

        outputs = model(**inputs)
        last_logits = outputs.logits[:, -1, :].cpu()
        all_last_logits.append(last_logits)

    return torch.cat(all_last_logits, dim=0).to(device)


@torch.no_grad()
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
    top_k = gen_config.top_k_vocab

    svt_enabled = privacy_config.svt_enabled
    theta = privacy_config.svt_threshold
    sigma = privacy_config.svt_noise

    generated_tokens: List[int] = []
    private_token_count = 0
    public_token_count = 0

    noisy_thresh = None
    if svt_enabled:
        noisy_thresh = sample_noisy_threshold(theta, sigma)

    while private_token_count < r and len(generated_tokens) < gen_config.max_total_tokens:
        private_logits = get_next_token_logits(
            model, tokenizer, private_prompts, generated_tokens, device,
            micro_batch_size=micro_batch_size,
        )

        if svt_enabled:
            public_logits = get_next_token_logits(
                model, tokenizer, [public_prompt], generated_tokens, device,
                micro_batch_size=micro_batch_size,
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
                    pub_logits = _apply_top_k_filter(pub_logits, public_logits, top_k)
                probs = torch.softmax(pub_logits / tau_pub, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                public_token_count += 1
        else:
            z_bar = clip_and_aggregate(private_logits, c, s)
            if top_k > 0:
                # Without SVT we still need public logits for the vocabulary mask.
                public_logits = get_next_token_logits(
                    model, tokenizer, [public_prompt], generated_tokens, device,
                    micro_batch_size=micro_batch_size,
                )[0]
                z_bar = _apply_top_k_filter(z_bar, public_logits, top_k)
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
    micro_batch_size: int = 32,
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

    # Causal LMs require left-padding so that position -1 is always the
    # last real token for every sequence in the batch.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Compute privacy budget
    delta = privacy_config.delta
    if delta is None:
        delta = 1.0 / len(examples)
        privacy_config.delta = delta

    # Partition into batches by label
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

    for label, batches in batches_by_label.items():
        label_name = templates["labels"][label]
        for batch in batches:
            batch_idx += 1
            if verbose:
                print(f"  Batch {batch_idx}/{total_batches} "
                      f"(label={label_name}, size={len(batch)})")

            private_prompts, public_prompt = build_prompts(
                batch, dataset_name, text_column, label,
                tokenizer=tokenizer,
            )

            token_ids, n_priv, n_pub = generate_one_example(
                model, tokenizer,
                private_prompts, public_prompt,
                privacy_config, gen_config,
                device=device,
                micro_batch_size=micro_batch_size,
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

    tokenizer.padding_side = orig_padding_side

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
