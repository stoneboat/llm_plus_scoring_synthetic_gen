"""
Batching layer for the private prediction pipeline.

Exports the public interface: BatchDescriptor, BatchingPolicy,
HashLabelBatchingPolicy, and the underlying helper functions.
"""

from src.batching.base import BatchDescriptor, BatchingPolicy
from src.batching.hash_label_policy import (
    HashLabelBatchingPolicy,
    assign_to_batch,
    partition_by_label,
)

__all__ = [
    "BatchDescriptor",
    "BatchingPolicy",
    "HashLabelBatchingPolicy",
    "assign_to_batch",
    "partition_by_label",
]
