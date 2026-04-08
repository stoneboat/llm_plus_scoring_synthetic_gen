"""
Simple JSONL helpers for non-checkpoint synthetic-data artifacts.

These helpers intentionally differ from the crash-safe checkpoint writer in
``jsonl_writer.py``:

- no ``_batch_complete`` markers
- no per-line ``fsync``
- suitable for one-shot outputs such as sweep artifacts

They remain compatible with ``load_synthetic_data`` so callers can read both
simple JSONL files and checkpoint-format JSONL files through one function.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

from src.runtime.generation import SyntheticExample


def save_synthetic_data(
    synthetic_examples: List[SyntheticExample],
    output_path: str,
    metadata: Optional[dict] = None,
) -> None:
    """Write a simple synthetic-data JSONL file.

    Args:
        synthetic_examples: examples to serialize.
        output_path: destination ``.jsonl`` path.
        metadata: optional ``_metadata`` header payload.
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


def load_synthetic_data(
    input_path: str,
) -> Tuple[List[SyntheticExample], Optional[dict]]:
    """Load synthetic examples from either simple or checkpoint JSONL.

    When ``_batch_complete`` markers are present, only examples from completed
    batches are returned. A malformed tail line is treated as an interrupted
    append and ignored.
    """
    examples: List[SyntheticExample] = []
    metadata: Optional[dict] = None
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
                print(
                    f"Warning: ignoring unreadable JSONL tail at line "
                    f"{line_no} in {input_path}"
                )
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
