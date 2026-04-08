"""
Evaluation result helpers.

Phase 5: extracted from scripts/evaluate_downstream.py.

Provides a single shared helper for persisting evaluation result dicts to JSON.
Callers (scripts and auxiliary tools) should use ``save_eval_results`` rather than
inlining ``json.dump`` + ``os.makedirs`` at each call site.
"""

import json
import os


def save_eval_results(results: dict, output_path: str) -> None:
    """Write an evaluation results dict to a JSON file.

    Creates parent directories if they do not exist.

    Args:
        results: evaluation result dict (e.g. from ``finetune_bert`` or
            ``icl_evaluate``), optionally enriched with metadata fields
            (``mode``, ``source``, ``dataset``, ``eval_time_seconds``, etc.).
        output_path: path for the output ``*.json`` file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")
