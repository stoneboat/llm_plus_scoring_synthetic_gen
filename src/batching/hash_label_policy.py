"""
Hash-label batching policy: deterministic, label-aware, fixed-assignment.

Implements Assumption 1 from the paper using SHA-256 hashing so that a
record's batch membership depends only on that record (and its label).
Adding or removing one example affects at most one batch, which is what
allows the paper's parallel-composition argument.

The functions assign_to_batch and partition_by_label were previously
defined in src/generate.py.  They are unchanged; only their home has moved.
src/generate.py re-imports them from here for backward compatibility.
"""

import hashlib
import math
from typing import Dict, List

from src.batching.base import BatchingPolicy


def assign_to_batch(prompt_text: str, num_batches: int) -> int:
    """Assign a prompt to a batch using SHA-256 hashing (Assumption 1).

    The assignment depends only on prompt_text itself, not on the ordering
    or presence of other prompts.

    Args:
        prompt_text: the text to hash (typically label + "\\n" + example text).
        num_batches: total number of buckets.

    Returns:
        Bucket index in [0, num_batches).
    """
    h = int(hashlib.sha256(prompt_text.encode()).hexdigest(), 16)
    return h % num_batches


def partition_by_label(
    examples: List[dict],
    label_column: str,
    text_column: str,
    batch_size: int,
) -> Dict[int, List[List[dict]]]:
    """Partition examples into stable, disjoint batches grouped by label.

    Returns a dict mapping label → list of batches, where each batch
    is a list of examples of size approximately batch_size.

    The assignment is hash-based rather than slice-based so a record's batch
    membership depends only on that record. This is the fixed-assignment
    property needed for the paper's parallel-composition argument: adding or
    removing one example affects only the batch containing that example, rather
    than shifting many later examples between batches.

    Within each batch, examples are sorted by text for a fully reproducible
    ordering (the hash already determines which bucket; the sort breaks ties).

    Args:
        examples: flat list of example dicts (must have label_column and
            text_column keys).
        label_column: key for the integer label.
        text_column: key for the text field.
        batch_size: target number of examples per bucket (actual sizes may vary).

    Returns:
        Dict[int, List[List[dict]]] — label → non-empty batches.
    """
    by_label: Dict[int, List[dict]] = {}
    for ex in examples:
        label = ex[label_column]
        by_label.setdefault(label, []).append(ex)

    batches_by_label: Dict[int, List[List[dict]]] = {}
    for label, label_examples in by_label.items():
        num_batches = max(1, math.ceil(len(label_examples) / batch_size))
        buckets: List[List[dict]] = [[] for _ in range(num_batches)]

        for ex in label_examples:
            # Include the label in the hash key so identical texts from
            # different labels cannot collide into the same label-local bucket.
            key = f"{label}\n{ex[text_column]}"
            batch_idx = assign_to_batch(key, num_batches)
            buckets[batch_idx].append(ex)

        # Drop empty buckets and sort deterministically for reproducibility.
        non_empty_batches = []
        for bucket in buckets:
            if not bucket:
                continue
            bucket.sort(key=lambda ex: ex[text_column])
            non_empty_batches.append(bucket)

        batches_by_label[label] = non_empty_batches

    return batches_by_label


class HashLabelBatchingPolicy(BatchingPolicy):
    """Concrete BatchingPolicy using label-aware SHA-256 hashing.

    This is the only batching policy currently implemented.  It wraps
    the free functions above so that callers can program to the
    BatchingPolicy interface if they choose to.
    """

    def partition(
        self,
        examples: List[dict],
        label_column: str,
        text_column: str,
        batch_size: int,
    ) -> Dict[int, List[List[dict]]]:
        return partition_by_label(examples, label_column, text_column, batch_size)
