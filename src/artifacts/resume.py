"""
Resume / recovery helpers for checkpointed generation runs.

``load_resume_state`` reads a checkpoint JSONL file produced by the artifact
writer and returns the already-completed state so that a new run can skip
finished batches and merge their examples at the end.

Recovery semantics (checkpoint_format=1):
- Only examples from batches that have a ``_batch_complete`` marker are
  returned as ``loaded_examples``.  Any example records following the last
  marker (a truncated-tail from a crash mid-batch) are silently ignored.
- A ``json.JSONDecodeError`` on any line is treated as a truncated tail;
  parsing stops there.  This makes the file usable after an interrupted write.
"""

import json
from typing import Dict, List, Optional, Set, Tuple

from src.batching.base import BatchDescriptor
from src.runtime.generation import SyntheticExample


def load_resume_state(
    output_path: str,
) -> Tuple[Optional[dict], List[SyntheticExample], Set[str], Dict[str, BatchDescriptor]]:
    """Parse a checkpoint JSONL file and return the completed state.

    Args:
        output_path: path to an existing checkpoint ``.jsonl`` file.

    Returns:
        A 4-tuple ``(metadata, loaded_examples, completed_batch_ids, batch_descriptors)``:

        metadata
            The ``_metadata`` dict from the file header, or ``None`` if the
            header line is absent or unreadable.

        loaded_examples
            ``SyntheticExample`` objects from **completed** batches only,
            in batch-completion order.

        completed_batch_ids
            Set of batch IDs for which a ``_batch_complete`` marker was found.

        batch_descriptors
            Dict mapping batch_id → ``BatchDescriptor`` for every batch seen
            (completed or not).  Useful for diagnostic reporting.
    """
    metadata: Optional[dict] = None
    batch_examples: Dict[str, List[SyntheticExample]] = {}
    batch_descriptors: Dict[str, BatchDescriptor] = {}
    completed_batch_ids: List[str] = []   # ordered; deduped via set at the end

    with open(output_path) as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: ignoring unreadable JSONL tail at line "
                    f"{line_no} in {output_path}"
                )
                break

            if "_metadata" in record:
                metadata = record["_metadata"]
                continue

            if "_batch_complete" in record:
                info = record["_batch_complete"]
                batch_id = info["batch_id"]
                batch_descriptors[batch_id] = BatchDescriptor(
                    batch_id=batch_id,
                    batch_index=info["batch_index"],
                    total_batches=info["total_batches"],
                    label=info["label"],
                    label_name=info["label_name"],
                    batch_size=info["batch_size"],
                )
                completed_batch_ids.append(batch_id)
                continue

            batch_id = record.get("batch_id")
            if batch_id is None:
                continue

            batch_examples.setdefault(batch_id, []).append(SyntheticExample(
                text=record["text"],
                label=record["label"],
                label_name=record["label_name"],
                num_private_tokens=record["num_private_tokens"],
                num_public_tokens=record["num_public_tokens"],
                num_total_tokens=record["num_total_tokens"],
            ))

            batch_index = record.get("batch_index")
            total_batches = record.get("total_batches")
            if batch_index is not None and total_batches is not None:
                batch_descriptors.setdefault(batch_id, BatchDescriptor(
                    batch_id=batch_id,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    label=record["label"],
                    label_name=record["label_name"],
                    batch_size=record.get("source_batch_size", 0),
                ))

    completed_batch_set = set(completed_batch_ids)
    loaded_examples: List[SyntheticExample] = []
    for batch_id in completed_batch_ids:
        loaded_examples.extend(batch_examples.get(batch_id, []))

    return metadata, loaded_examples, completed_batch_set, batch_descriptors
