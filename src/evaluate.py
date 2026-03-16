"""
Evaluation utilities for synthetic data quality.

Supports two evaluation modes from the paper:
1. In-context learning (ICL): use synthetic examples as few-shot exemplars
2. Fine-tuning: train a classifier on synthetic data, evaluate on real test data

Reference: Sections 5.1 and 5.2 of Amin et al. (2024).
"""

import json
import os
from typing import List, Dict, Optional, Tuple

from src.generate import SyntheticExample
from src.config import PROMPT_TEMPLATES


def format_icl_prompt(
    synthetic_examples: List[SyntheticExample],
    test_text: str,
    dataset_name: str,
    num_shots: int = 4,
) -> str:
    """Format synthetic examples as an in-context learning prompt.

    Follows the evaluation format from Figure 2 in the paper.

    Args:
        synthetic_examples: generated synthetic data.
        test_text: the real test example to classify.
        dataset_name: name of the dataset for label formatting.
        num_shots: number of synthetic examples to include.

    Returns:
        Formatted ICL prompt string.
    """
    templates = PROMPT_TEMPLATES[dataset_name]

    lines = ["Classify the following examples:"]
    for ex in synthetic_examples[:num_shots]:
        lines.append(f"Text: {ex.text}")
        lines.append(f"Answer: {ex.label_name}")

    lines.append("")
    lines.append(f"Text: {test_text}")
    lines.append("Answer:")

    return "\n".join(lines)


def save_synthetic_data(
    synthetic_examples: List[SyntheticExample],
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save synthetic examples to a JSONL file.

    Args:
        synthetic_examples: list of SyntheticExample.
        output_path: path to write the file.
        metadata: optional dict of experiment metadata to include in header.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        if metadata:
            f.write(json.dumps({"_metadata": metadata}) + "\n")
        for ex in synthetic_examples:
            record = {
                "text": ex.text,
                "label": ex.label,
                "label_name": ex.label_name,
                "num_private_tokens": ex.num_private_tokens,
                "num_public_tokens": ex.num_public_tokens,
                "num_total_tokens": ex.num_total_tokens,
            }
            f.write(json.dumps(record) + "\n")


def load_synthetic_data(input_path: str) -> Tuple[List[SyntheticExample], Optional[dict]]:
    """Load synthetic examples from a JSONL file.

    Returns:
        (list of SyntheticExample, metadata dict or None)
    """
    examples = []
    metadata = None

    with open(input_path) as f:
        for line in f:
            record = json.loads(line.strip())
            if "_metadata" in record:
                metadata = record["_metadata"]
                continue
            examples.append(SyntheticExample(
                text=record["text"],
                label=record["label"],
                label_name=record["label_name"],
                num_private_tokens=record["num_private_tokens"],
                num_public_tokens=record["num_public_tokens"],
                num_total_tokens=record["num_total_tokens"],
            ))

    return examples, metadata


def compute_generation_stats(examples: List[SyntheticExample]) -> dict:
    """Compute summary statistics over generated examples."""
    if not examples:
        return {}

    total_tokens = [e.num_total_tokens for e in examples]
    priv_tokens = [e.num_private_tokens for e in examples]
    pub_tokens = [e.num_public_tokens for e in examples]

    total_priv = sum(priv_tokens)
    total_pub = sum(pub_tokens)
    total_all = total_priv + total_pub

    label_counts: Dict[int, int] = {}
    for e in examples:
        label_counts[e.label] = label_counts.get(e.label, 0) + 1

    return {
        "num_examples": len(examples),
        "label_distribution": label_counts,
        "total_tokens": sum(total_tokens),
        "mean_tokens_per_example": sum(total_tokens) / len(examples),
        "total_private_tokens": total_priv,
        "total_public_tokens": total_pub,
        "public_token_fraction": total_pub / max(1, total_all),
        "max_private_tokens_in_example": max(priv_tokens),
    }
