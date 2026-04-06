"""
Tests for batch assignment / batching determinism (Phase 1 guard).

Contracts protected here:
- assign_to_batch is a pure deterministic function of (text, num_batches).
- partition_by_label produces identical output across repeated calls
  (hash-based, not index-based, so record membership depends only on
  that record, not on the ordering of other records).
- Label is included in the hash key so identical texts from different
  labels cannot collide into the same label-local bucket.
- Within-batch examples are sorted by text for reproducible ordering.
- Every input example appears in exactly one batch.
"""

import pytest
from src.generate import assign_to_batch, partition_by_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_examples(n, num_labels=3):
    return [{"text": f"text example {i:04d}", "label": i % num_labels} for i in range(n)]


# ---------------------------------------------------------------------------
# assign_to_batch
# ---------------------------------------------------------------------------

def test_assign_to_batch_deterministic():
    """Same (text, num_batches) always returns the same bucket index."""
    text = "hello world this is a test"
    n = 10
    results = [assign_to_batch(text, n) for _ in range(5)]
    assert len(set(results)) == 1, "assign_to_batch must be deterministic"


def test_assign_to_batch_output_in_range():
    """Output is always in [0, num_batches)."""
    for text in ["a", "hello", "the quick brown fox jumps over the lazy dog"]:
        for n in [1, 3, 7, 100]:
            idx = assign_to_batch(text, n)
            assert 0 <= idx < n, f"idx {idx} out of [0, {n})"


def test_assign_to_batch_single_bucket():
    """With num_batches=1, every text maps to bucket 0."""
    for text in ["a", "b", "c", "long text " * 20]:
        assert assign_to_batch(text, 1) == 0


# ---------------------------------------------------------------------------
# partition_by_label
# ---------------------------------------------------------------------------

def test_partition_by_label_deterministic():
    """Repeated calls with identical inputs produce identical batch structures."""
    examples = _make_examples(30)
    r1 = partition_by_label(examples, "label", "text", batch_size=10)
    r2 = partition_by_label(examples, "label", "text", batch_size=10)
    assert r1.keys() == r2.keys()
    for label in r1:
        assert len(r1[label]) == len(r2[label]), (
            f"Label {label}: different number of batches across calls"
        )
        for b1, b2 in zip(r1[label], r2[label]):
            assert b1 == b2, f"Batch contents differ for label {label}"


def test_partition_by_label_covers_all_examples():
    """Every input example appears in exactly one batch."""
    examples = _make_examples(30)
    batches_by_label = partition_by_label(examples, "label", "text", batch_size=10)
    seen_texts = []
    for batches in batches_by_label.values():
        for batch in batches:
            for ex in batch:
                seen_texts.append(ex["text"])
    assert sorted(seen_texts) == sorted(e["text"] for e in examples), (
        "Not all examples are covered, or some appear more than once"
    )


def test_partition_by_label_groups_by_label():
    """All examples within any single batch share the same label."""
    examples = _make_examples(30)
    batches_by_label = partition_by_label(examples, "label", "text", batch_size=10)
    for label, batches in batches_by_label.items():
        for batch in batches:
            for ex in batch:
                assert ex["label"] == label, (
                    f"Example with label={ex['label']} found in label={label} batch"
                )


def test_partition_by_label_produces_correct_keys():
    """Keys of the returned dict are the unique label values."""
    examples = _make_examples(30, num_labels=4)
    batches_by_label = partition_by_label(examples, "label", "text", batch_size=10)
    expected_labels = {0, 1, 2, 3}
    assert set(batches_by_label.keys()) == expected_labels


def test_partition_within_batch_sorted_by_text():
    """Examples within each batch are sorted by text (deterministic ordering)."""
    examples = [{"text": f"text {i:04d}", "label": 0} for i in range(20)]
    batches_by_label = partition_by_label(examples, "label", "text", batch_size=5)
    for batches in batches_by_label.values():
        for batch in batches:
            texts = [ex["text"] for ex in batch]
            assert texts == sorted(texts), (
                "Within-batch examples must be sorted by text"
            )


def test_partition_label_aware_hash():
    """Label is included in the hash key: same text, different label -> different hash."""
    # This directly tests the "f'{label}\\n{text}'" keying in partition_by_label.
    # assign_to_batch("0\nfoo", n) should differ from assign_to_batch("1\nfoo", n)
    # for any n > 1 (true with overwhelming probability for SHA-256).
    key_label0 = "0\nfoo bar baz"
    key_label1 = "1\nfoo bar baz"
    n = 100  # use a large n to make collision astronomically unlikely
    assert assign_to_batch(key_label0, n) != assign_to_batch(key_label1, n), (
        "Label-prefixed keys should hash to different buckets"
    )


def test_partition_single_example_per_label():
    """A single example per label results in one batch per label."""
    examples = [{"text": "only text", "label": lbl} for lbl in range(3)]
    batches_by_label = partition_by_label(examples, "label", "text", batch_size=10)
    for label, batches in batches_by_label.items():
        total_examples = sum(len(b) for b in batches)
        assert total_examples == 1, (
            f"Expected 1 example for label {label}, got {total_examples}"
        )
