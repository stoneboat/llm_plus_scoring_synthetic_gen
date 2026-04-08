"""
Evaluation utilities for synthetic data quality.

Supports two evaluation modes from the paper:
1. In-context learning (ICL): use synthetic examples as few-shot exemplars
2. Fine-tuning: train a classifier on synthetic data, evaluate on real test data

Reference: Sections 5.1 and 5.2 of Amin et al. (2024).

This module is now primarily a compatibility facade:

- simple JSONL I/O re-exported from ``src.artifacts``
- generation diagnostics re-exported from ``src.runtime``
- BERT fine-tuning re-exported from ``src.evaluation.finetune``

``format_icl_prompt`` remains defined here because it is intentionally a
different prompt format from ``src.evaluation.icl.build_icl_prompt`` and is
kept for backward compatibility.
"""

from typing import List

from src.runtime.generation import SyntheticExample
from src.config import PROMPT_TEMPLATES
from src.artifacts import save_synthetic_data, load_synthetic_data  # noqa: F401
from src.runtime import compute_generation_stats  # noqa: F401

# Phase 5: finetune_and_evaluate is now the evaluation layer's finetune_bert.
# Imported here as a re-export so all existing callers remain unaffected.
from src.evaluation.finetune import finetune_bert as finetune_and_evaluate  # noqa: F401


def format_icl_prompt(
    synthetic_examples: List[SyntheticExample],
    test_text: str,
    dataset_name: str,
    num_shots: int = 4,
) -> str:
    """Format synthetic examples as an in-context learning prompt.

    Follows the evaluation format from Figure 2 in the paper.

    Note: this is the simple "Classify the following examples / Answer:"
    format.  The balanced per-label evaluation format used in
    ``src.evaluation.icl.build_icl_prompt`` is a separate function
    with different prompt structure.

    Args:
        synthetic_examples: generated synthetic data.
        test_text: the real test example to classify.
        dataset_name: name of the dataset for label formatting.
        num_shots: number of synthetic examples to include.

    Returns:
        Formatted ICL prompt string.
    """
    templates = PROMPT_TEMPLATES[dataset_name]  # noqa: F841 (templates referenced for future label use)

    lines = ["Classify the following examples:"]
    for ex in synthetic_examples[:num_shots]:
        lines.append(f"Text: {ex.text}")
        lines.append(f"Answer: {ex.label_name}")

    lines.append("")
    lines.append(f"Text: {test_text}")
    lines.append("Answer:")

    return "\n".join(lines)
