"""
Artifact writing / checkpoint / resume layer.

Phase 4: extracted from scripts/run_experiment.py.

Provides crash-safe JSONL I/O, resume state loading, and run-metadata
construction as reusable library functions so that scripts become thin
wrappers and other callers (e.g. sweep scripts) can share the same
checkpoint format without copy-pasting.

Public surface:
    append_line              — write one JSON record with fsync
    batch_record             — build the per-example JSONL dict
    write_metadata_header    — create the output file and write ``_metadata``
    append_completed_batch   — append example records + ``_batch_complete`` marker
    load_resume_state        — parse an existing checkpoint JSONL
    build_run_metadata       — assemble the run-metadata dict
"""

from src.artifacts.jsonl_writer import (  # noqa: F401
    append_line,
    batch_record,
    write_metadata_header,
    append_completed_batch,
)
from src.artifacts.resume import load_resume_state  # noqa: F401
from src.artifacts.metadata import build_run_metadata  # noqa: F401

__all__ = [
    "append_line",
    "batch_record",
    "write_metadata_header",
    "append_completed_batch",
    "load_resume_state",
    "build_run_metadata",
]
