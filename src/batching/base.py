"""
Base types for the batching layer.

BatchDescriptor is the stable, frozen identifier carried alongside each
generation batch.  It was previously defined in src/generate.py; it lives
here now so that downstream code can import it without pulling in the full
generation stack.

BatchingPolicy is the abstract base for batch-assignment strategies.
Only one concrete implementation exists today (HashLabelBatchingPolicy),
but the ABC makes the contract explicit and allows Phase 3 to swap it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class BatchDescriptor:
    """Stable identifier and metadata for one generation batch.

    Frozen so it can be used as a dict key and passed safely across
    call boundaries.  The batch_id is a SHA-256 hex digest that depends
    only on the dataset name, label, and the sorted texts in the batch
    — it does not change if other batches change.
    """

    batch_id: str
    batch_index: int
    total_batches: int
    label: int
    label_name: str
    batch_size: int


class BatchingPolicy(ABC):
    """Abstract interface for partitioning examples into generation batches.

    A BatchingPolicy takes a flat list of examples and returns a nested
    structure: label → list of batches, where each batch is a list of
    example dicts.

    The fixed-assignment property (each example's batch membership depends
    only on that example, not on the set of other examples) must be
    preserved by all implementations.  This is what makes the parallel-
    composition argument in the paper hold.
    """

    @abstractmethod
    def partition(
        self,
        examples: List[dict],
        label_column: str,
        text_column: str,
        batch_size: int,
    ) -> Dict[int, List[List[dict]]]:
        """Partition examples into stable, disjoint batches grouped by label.

        Args:
            examples: flat list of example dicts.
            label_column: key for the integer label.
            text_column: key for the text field.
            batch_size: target number of examples per batch.

        Returns:
            Dict mapping label int → list of batches, where each batch
            is a list of example dicts of size approximately batch_size.
        """
        ...
