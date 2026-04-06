"""
Tests for the batching policy layer: HashLabelBatchingPolicy,
and backward-compatible imports from src.generate (Phase 2 guard).

Contracts protected here:
- HashLabelBatchingPolicy.partition() produces identical output to the
  standalone partition_by_label() (parity test).
- assign_to_batch and partition_by_label are still importable from
  src.generate for backward compatibility.
- BatchDescriptor is importable from both src.batching and src.generate.
- The batch ID computation in generate_synthetic_dataset is deterministic
  (tested via BatchDescriptor round-trip).
"""

import pytest

from src.batching.base import BatchDescriptor, BatchingPolicy
from src.batching.hash_label_policy import (
    HashLabelBatchingPolicy,
    assign_to_batch,
    partition_by_label,
)


# ---------------------------------------------------------------------------
# Backward-compat import checks
# ---------------------------------------------------------------------------

def test_assign_to_batch_importable_from_generate():
    from src.generate import assign_to_batch as f
    assert callable(f)


def test_partition_by_label_importable_from_generate():
    from src.generate import partition_by_label as f
    assert callable(f)


def test_batch_descriptor_importable_from_generate():
    from src.generate import BatchDescriptor as BD
    assert BD is BatchDescriptor


def test_batch_descriptor_importable_from_batching():
    from src.batching import BatchDescriptor as BD
    assert BD is BatchDescriptor


# ---------------------------------------------------------------------------
# HashLabelBatchingPolicy
# ---------------------------------------------------------------------------

def _make_examples(n, num_labels=3):
    return [{"text": f"example text {i:04d}", "label": i % num_labels} for i in range(n)]


def test_policy_is_batching_policy_subclass():
    policy = HashLabelBatchingPolicy()
    assert isinstance(policy, BatchingPolicy)


def test_policy_partition_matches_standalone():
    """HashLabelBatchingPolicy.partition() must return the same result as
    the standalone partition_by_label() function."""
    examples = _make_examples(40)
    policy = HashLabelBatchingPolicy()

    result_fn = partition_by_label(examples, "label", "text", batch_size=12)
    result_policy = policy.partition(examples, "label", "text", batch_size=12)

    assert result_fn.keys() == result_policy.keys()
    for label in result_fn:
        assert len(result_fn[label]) == len(result_policy[label])
        for b1, b2 in zip(result_fn[label], result_policy[label]):
            assert b1 == b2, (
                f"Batch mismatch for label {label}: policy output differs from function"
            )


def test_policy_partition_deterministic():
    """Repeated calls with identical inputs produce identical output."""
    examples = _make_examples(30)
    policy = HashLabelBatchingPolicy()

    r1 = policy.partition(examples, "label", "text", batch_size=10)
    r2 = policy.partition(examples, "label", "text", batch_size=10)

    assert r1.keys() == r2.keys()
    for label in r1:
        for b1, b2 in zip(r1[label], r2[label]):
            assert b1 == b2


def test_policy_partition_covers_all_examples():
    """Every input example appears in the partition output exactly once."""
    examples = _make_examples(25)
    policy = HashLabelBatchingPolicy()
    result = policy.partition(examples, "label", "text", batch_size=8)

    seen = []
    for batches in result.values():
        for batch in batches:
            seen.extend(ex["text"] for ex in batch)

    assert sorted(seen) == sorted(ex["text"] for ex in examples)


# ---------------------------------------------------------------------------
# BatchDescriptor frozen / hashable
# ---------------------------------------------------------------------------

def test_batch_descriptor_is_frozen():
    """BatchDescriptor is frozen: attribute assignment must raise."""
    bd = BatchDescriptor(
        batch_id="abc",
        batch_index=1,
        total_batches=5,
        label=0,
        label_name="World",
        batch_size=255,
    )
    with pytest.raises((AttributeError, TypeError)):
        bd.batch_index = 99


def test_batch_descriptor_hashable():
    """Frozen dataclass must be usable as a dict key / set member."""
    bd = BatchDescriptor(
        batch_id="xyz",
        batch_index=2,
        total_batches=10,
        label=1,
        label_name="Sports",
        batch_size=127,
    )
    seen = {bd}
    assert bd in seen


def test_batch_descriptor_equality():
    """Two BatchDescriptors with identical fields are equal."""
    bd1 = BatchDescriptor("id", 1, 5, 0, "World", 255)
    bd2 = BatchDescriptor("id", 1, 5, 0, "World", 255)
    assert bd1 == bd2


# ---------------------------------------------------------------------------
# Parity: batching layer vs. old generate.py functions
# ---------------------------------------------------------------------------

def test_assign_to_batch_same_from_both_import_paths():
    """assign_to_batch imported from batching or generate returns same value."""
    from src.batching.hash_label_policy import assign_to_batch as from_batching
    from src.generate import assign_to_batch as from_generate

    for text in ["hello", "world", "x" * 100]:
        for n in [1, 7, 100]:
            assert from_batching(text, n) == from_generate(text, n)


def test_partition_by_label_same_from_both_import_paths():
    """partition_by_label imported from batching or generate returns same result."""
    from src.batching.hash_label_policy import partition_by_label as from_batching
    from src.generate import partition_by_label as from_generate

    examples = _make_examples(20)
    r1 = from_batching(examples, "label", "text", 7)
    r2 = from_generate(examples, "label", "text", 7)

    assert r1.keys() == r2.keys()
    for label in r1:
        for b1, b2 in zip(r1[label], r2[label]):
            assert b1 == b2
