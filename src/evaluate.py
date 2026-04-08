"""
Evaluation utilities for synthetic data quality.

Supports two evaluation modes from the paper:
1. In-context learning (ICL): use synthetic examples as few-shot exemplars
2. Fine-tuning: train a classifier on synthetic data, evaluate on real test data

Reference: Sections 5.1 and 5.2 of Amin et al. (2024).

Phase 5 note: BERT fine-tuning has been consolidated into
src/evaluation/finetune.py.  ``finetune_and_evaluate`` is now a re-export of
``src.evaluation.finetune.finetune_bert`` with an identical signature, so all
existing callers (scripts, sweep) are unaffected.

The following functions remain canonical here (not moved) because they are
independently tested, widely imported, and serve roles beyond evaluation:
  - save_synthetic_data    — write generated examples to a simple JSONL file
  - load_synthetic_data    — read JSONL with batch-marker semantics
  - compute_generation_stats — generation diagnostics (not downstream eval)
  - format_icl_prompt      — simple ICL prompt formatter (format differs from
                             the balanced evaluation formatter in evaluation/icl.py)
"""

import json
import os
from typing import List, Dict, Optional, Tuple

import torch
from sklearn.metrics import accuracy_score, classification_report

from src.runtime.generation import SyntheticExample
from src.config import PROMPT_TEMPLATES

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


def save_synthetic_data(
    synthetic_examples: List[SyntheticExample],
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Save synthetic examples to a JSONL file.

    Writes a simple JSONL without batch markers or fsync.  For the full
    checkpoint-format writer (batch markers, fsync, crash safety) use
    ``src.artifacts.append_completed_batch``.

    Args:
        synthetic_examples: list of SyntheticExample.
        output_path: path to write the file.
        metadata: optional dict of experiment metadata to include in header.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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

    Handles both the simple format (written by ``save_synthetic_data``) and the
    full checkpoint format (with ``_batch_complete`` markers).  When markers are
    present, only examples from completed batches are returned.

    Returns:
        (list of SyntheticExample, metadata dict or None)
    """
    examples = []
    metadata = None
    batch_examples: Dict[str, List[SyntheticExample]] = {}
    completed_batch_ids: List[str] = []
    saw_batch_markers = False

    with open(input_path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # Allow interrupted checkpoint files with a truncated tail line.
                print(f"Warning: ignoring unreadable JSONL tail at line {line_no} in {input_path}")
                break
            if "_metadata" in record:
                metadata = record["_metadata"]
                continue
            if "_batch_complete" in record:
                saw_batch_markers = True
                completed_batch_ids.append(record["_batch_complete"]["batch_id"])
                continue

            example = SyntheticExample(
                text=record["text"],
                label=record["label"],
                label_name=record["label_name"],
                num_private_tokens=record["num_private_tokens"],
                num_public_tokens=record["num_public_tokens"],
                num_total_tokens=record["num_total_tokens"],
            )
            batch_id = record.get("batch_id")
            if batch_id is not None:
                batch_examples.setdefault(batch_id, []).append(example)
            examples.append(example)

    if saw_batch_markers:
        completed_examples: List[SyntheticExample] = []
        for batch_id in completed_batch_ids:
            completed_examples.extend(batch_examples.get(batch_id, []))
        examples = completed_examples

    return examples, metadata


def compute_generation_stats(examples: List[SyntheticExample]) -> dict:
    """Compute summary statistics over generated examples.

    Returns generation diagnostics (token counts, label distribution) rather
    than downstream task metrics.  Used by scripts after generation completes.
    """
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
