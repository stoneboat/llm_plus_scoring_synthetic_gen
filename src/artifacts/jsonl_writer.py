"""
Crash-safe JSONL artifact writing for checkpointed generation runs.

All writes are atomic at the line level: each JSON line is flushed and
fsync-ed immediately after writing, so a crash between lines leaves a
valid (possibly truncated-tail) file that ``load_resume_state`` can recover.

JSONL schema (checkpoint_format=1):

  Line 1:   {"_metadata": { ... run parameters ... }}
  Per example: {
      "text": str,
      "label": int,
      "label_name": str,
      "num_private_tokens": int,
      "num_public_tokens": int,
      "num_total_tokens": int,
      "batch_id": str,
      "batch_index": int,
      "total_batches": int,
      "source_batch_size": int,
  }
  Per batch: {"_batch_complete": {
      "batch_id": str, "batch_index": int, "total_batches": int,
      "label": int, "label_name": str, "batch_size": int, "num_examples": int,
  }}

``source_batch_size`` is written on example records (not on the marker) so
that resume can reconstruct the BatchDescriptor from example records alone,
without relying solely on the marker.
"""

import json
import os
from typing import IO, List

from src.batching.base import BatchDescriptor
from src.runtime.generation import SyntheticExample


def append_line(handle: IO[str], record: dict) -> None:
    """Write one JSON record to *handle*, flushing and fsync-ing immediately."""
    handle.write(json.dumps(record) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def batch_record(ex: SyntheticExample, descriptor: BatchDescriptor) -> dict:
    """Build the per-example JSONL record (without ``source_batch_size``)."""
    return {
        "text": ex.text,
        "label": ex.label,
        "label_name": ex.label_name,
        "num_private_tokens": ex.num_private_tokens,
        "num_public_tokens": ex.num_public_tokens,
        "num_total_tokens": ex.num_total_tokens,
        "batch_id": descriptor.batch_id,
        "batch_index": descriptor.batch_index,
        "total_batches": descriptor.total_batches,
    }


def write_metadata_header(output_path: str, metadata: dict) -> None:
    """Create *output_path* and write the ``{"_metadata": ...}`` header line.

    Creates parent directories if they do not exist.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        append_line(f, {"_metadata": metadata})


def append_completed_batch(
    output_path: str,
    descriptor: BatchDescriptor,
    examples: List[SyntheticExample],
) -> None:
    """Append example records and a ``_batch_complete`` marker to *output_path*.

    Each example record includes ``source_batch_size`` (the number of
    private prompts in the originating batch) so that resume can reconstruct
    the BatchDescriptor even in the absence of the marker.

    The ``_batch_complete`` marker is written last.  A crash before it means
    the batch will be re-generated on resume (safe, because the tail is
    ignored).  A crash after it means the batch is considered done on resume.
    """
    with open(output_path, "a") as f:
        for ex in examples:
            record = batch_record(ex, descriptor)
            record["source_batch_size"] = descriptor.batch_size
            append_line(f, record)

        append_line(f, {
            "_batch_complete": {
                "batch_id": descriptor.batch_id,
                "batch_index": descriptor.batch_index,
                "total_batches": descriptor.total_batches,
                "label": descriptor.label,
                "label_name": descriptor.label_name,
                "batch_size": descriptor.batch_size,
                "num_examples": len(examples),
            }
        })
