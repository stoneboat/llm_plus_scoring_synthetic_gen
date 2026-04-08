"""
Runtime / orchestration layer for private-prediction synthetic text generation.

Phase 4: batch-level orchestration and dataset-level generation loop extracted
from src/generate.py into this package so that the logic is independently
importable and testable without pulling in the full generate.py compat surface.

Public surface:
    SyntheticExample       — per-generated-example result dataclass
    run_batch_generation   — Algorithm 1 outer loop for one batch
    run_dataset_generation — full dataset-level orchestration loop
    compute_generation_stats — summary statistics over generated examples
"""

from src.runtime.generation import (  # noqa: F401
    SyntheticExample,
    run_batch_generation,
    run_dataset_generation,
)
from src.runtime.stats import compute_generation_stats  # noqa: F401

__all__ = [
    "SyntheticExample",
    "run_batch_generation",
    "run_dataset_generation",
    "compute_generation_stats",
]
