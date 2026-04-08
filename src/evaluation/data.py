"""
Evaluation-time data loading helpers.

Phase 5: extracted from scripts/evaluate_downstream.py.

Provides three complementary loaders:

load_test_set         — load the real held-out test split via the dataset
                        adapter registry; used by all evaluation modes.
load_real_train       — load the real training split as a synthetic-data
                        replacement baseline (upper-bound reference point).
load_synthetic_for_eval — load a saved synthetic JSONL file and return
                        (texts, labels, label_names, metadata) so that callers
                        do not need to import SyntheticExample directly.

All dataset access goes through the adapter registry (src.datasets.registry)
so that HuggingFace dataset routing and caching logic stays in one place.
Synthetic-JSONL loading delegates to src.evaluate.load_synthetic_data so that
batch-marker semantics and the tested round-trip contract are not duplicated.
"""

from typing import Dict, List, Optional, Tuple

from src.datasets.registry import get_adapter


def load_test_set(
    dataset_name: str,
    cache_dir: str = "data/datasets",
    max_examples: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    """Load the real held-out test split.

    Args:
        dataset_name: key for the dataset adapter registry (e.g. ``"agnews"``).
        cache_dir: HuggingFace cache directory.
        max_examples: cap on the number of examples returned (``None`` = all).

    Returns:
        ``(texts, labels)`` — parallel lists of strings and integer label IDs.
    """
    adapter = get_adapter(dataset_name)
    print(f"Loading test set: {adapter.hf_name} (test)...")
    examples = adapter.load("test", num_examples=max_examples, cache_dir=cache_dir)
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    print(f"  {len(texts)} test examples, {len(set(labels))} classes")
    return texts, labels


def load_real_train(
    dataset_name: str,
    cache_dir: str = "data/datasets",
    max_examples: Optional[int] = None,
) -> Tuple[List[str], List[int]]:
    """Load the real training split (upper-bound baseline comparison).

    Args:
        dataset_name: key for the dataset adapter registry.
        cache_dir: HuggingFace cache directory.
        max_examples: cap on the number of examples returned (``None`` = all).

    Returns:
        ``(texts, labels)`` — parallel lists of strings and integer label IDs.
    """
    adapter = get_adapter(dataset_name)
    print(f"Loading real training set: {adapter.hf_name} (train)...")
    examples = adapter.load("train", num_examples=max_examples, cache_dir=cache_dir)
    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    print(f"  {len(texts)} training examples")
    return texts, labels


def load_synthetic_for_eval(
    input_path: str,
) -> Tuple[List[str], List[int], List[str], Optional[dict]]:
    """Load a saved synthetic JSONL and extract evaluation-ready lists.

    Delegates to ``src.evaluate.load_synthetic_data`` so that batch-marker
    resume semantics and the existing round-trip contract are preserved.

    The import of ``load_synthetic_data`` is deferred to function body to
    avoid a circular import (src.evaluate → src.evaluation → src.evaluate).

    Args:
        input_path: path to a synthetic ``.jsonl`` file produced by
            ``src.artifacts`` or ``src.evaluate.save_synthetic_data``.

    Returns:
        A 4-tuple ``(texts, labels, label_names, metadata)``:

        texts
            List of generated text strings.
        labels
            Integer label IDs (parallel to texts).
        label_names
            Human-readable label names (parallel to texts).
        metadata
            The ``_metadata`` dict from the file header, or ``None``.
    """
    from src.evaluate import load_synthetic_data  # deferred: avoids circular import
    examples, metadata = load_synthetic_data(input_path)
    texts = [e.text for e in examples]
    labels = [e.label for e in examples]
    label_names = [e.label_name for e in examples]
    return texts, labels, label_names, metadata
