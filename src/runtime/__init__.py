"""
Runtime / orchestration layer for private-prediction synthetic text generation.

Phase 4: batch-level orchestration and dataset-level generation loop extracted
from src/generate.py into this package so that the logic is independently
importable and testable without pulling in the full generate.py compat surface.

Public surface:
    SyntheticExample       — per-generated-example result dataclass
    run_batch_generation   — Algorithm 1 outer loop for one batch
    run_dataset_generation — full dataset-level orchestration loop
"""

from src.runtime.generation import (  # noqa: F401
    SyntheticExample,
    run_batch_generation,
    run_dataset_generation,
)

__all__ = [
    "SyntheticExample",
    "run_batch_generation",
    "run_dataset_generation",
]
