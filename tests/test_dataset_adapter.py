"""
Tests for the dataset layer: DatasetAdapter, TaskSpec, registry (Phase 2 guard).

Contracts protected here:
- get_adapter returns the correct HFTextClassificationAdapter for each name.
- DATASET_CHOICES is sorted and matches REGISTRY keys.
- Each adapter exposes num_labels and label_names consistent with
  PROMPT_TEMPLATES["labels"] (parity with old config-based data).
- load() is deferred to HuggingFace; we mock it here so no network is needed.
- HFTextClassificationAdapter.load() normalizes rows to {"text": ..., "label": ...}.
- HFTextClassificationAdapter.load() rejects unknown split names.
- TaskSpec fields are accessible via adapter convenience properties.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.datasets.base import DatasetAdapter, TaskSpec
from src.datasets.text_classification import HFTextClassificationAdapter
from src.datasets.registry import REGISTRY, DATASET_CHOICES, get_adapter
from src.prompts.text_classification import PROMPT_TEMPLATES


# ---------------------------------------------------------------------------
# Registry smoke tests
# ---------------------------------------------------------------------------

def test_dataset_choices_sorted():
    assert DATASET_CHOICES == sorted(DATASET_CHOICES), (
        "DATASET_CHOICES must be sorted"
    )


def test_dataset_choices_match_registry():
    assert set(DATASET_CHOICES) == set(REGISTRY.keys())


def test_get_adapter_known_names():
    for name in ["agnews", "dbpedia", "imdb", "yelp", "trec"]:
        adapter = get_adapter(name)
        assert adapter.name == name


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        get_adapter("nonexistent_dataset")


def test_all_adapters_are_dataset_adapter_instances():
    for name, adapter in REGISTRY.items():
        assert isinstance(adapter, DatasetAdapter), (
            f"REGISTRY['{name}'] is not a DatasetAdapter"
        )


# ---------------------------------------------------------------------------
# TaskSpec / adapter metadata parity with PROMPT_TEMPLATES
# ---------------------------------------------------------------------------

def test_adapter_num_labels_matches_prompt_templates():
    """num_labels must be consistent with the number of labels in PROMPT_TEMPLATES."""
    for name, adapter in REGISTRY.items():
        template_labels = PROMPT_TEMPLATES[name]["labels"]
        assert adapter.num_labels == len(template_labels), (
            f"adapter['{name}'].num_labels={adapter.num_labels} does not match "
            f"len(PROMPT_TEMPLATES['{name}']['labels'])={len(template_labels)}"
        )


def test_adapter_label_names_match_prompt_templates():
    """label_names must agree with PROMPT_TEMPLATES['labels'] for each dataset."""
    for name, adapter in REGISTRY.items():
        template_labels = PROMPT_TEMPLATES[name]["labels"]
        assert adapter.label_names == template_labels, (
            f"adapter['{name}'].label_names differs from PROMPT_TEMPLATES labels"
        )


def test_adapter_task_spec_properties():
    """num_labels and label_names are also accessible via TaskSpec."""
    for name, adapter in REGISTRY.items():
        assert adapter.task.num_labels == adapter.num_labels
        assert adapter.task.label_names == adapter.label_names


# ---------------------------------------------------------------------------
# Adapter per-dataset metadata spot-checks
# ---------------------------------------------------------------------------

def test_agnews_adapter_metadata():
    a = get_adapter("agnews")
    assert a.hf_name == "fancyzhx/ag_news"
    assert a.train_split == "train"
    assert a.test_split == "test"
    assert a.hf_text_column == "text"
    assert a.hf_label_column == "label"
    assert a.num_labels == 4
    assert a.label_names[0] == "World"
    assert a.label_names[3] == "Sci/Tech"


def test_trec_adapter_metadata():
    a = get_adapter("trec")
    assert a.hf_name == "CogComp/trec"
    assert a.hf_label_column == "coarse_label"
    assert a.num_labels == 6


def test_dbpedia_adapter_metadata():
    a = get_adapter("dbpedia")
    assert a.hf_name == "fancyzhx/dbpedia_14"
    assert a.hf_text_column == "content"
    assert a.num_labels == 14
    assert a.label_names[13] == "Book"


# ---------------------------------------------------------------------------
# HFTextClassificationAdapter.load() — mocked to avoid HF network calls
# ---------------------------------------------------------------------------

def _make_fake_ds(rows):
    """Return a fake HuggingFace Dataset-like object."""
    fake = MagicMock()
    fake.__iter__ = MagicMock(return_value=iter(rows))
    fake.__len__ = MagicMock(return_value=len(rows))
    return fake


def test_load_normalizes_columns():
    """load() maps raw HF columns to {"text": ..., "label": ...}."""
    raw_rows = [
        {"text": "hello world", "label": 0},
        {"text": "sports news", "label": 1},
    ]
    fake_ds = _make_fake_ds(raw_rows)
    adapter = get_adapter("agnews")

    with patch("datasets.load_dataset",
               return_value=fake_ds) as mock_load:
        result = adapter.load("train", cache_dir="/tmp")

    mock_load.assert_called_once_with(
        adapter.hf_name, split=adapter.train_split, cache_dir="/tmp"
    )
    assert len(result) == 2
    assert result[0] == {"text": "hello world", "label": 0}
    assert result[1] == {"text": "sports news", "label": 1}


def test_load_routes_train_split():
    """load('train') uses train_split, not test_split."""
    fake_ds = _make_fake_ds([{"text": "x", "label": 0}])
    adapter = get_adapter("agnews")

    with patch("datasets.load_dataset",
               return_value=fake_ds) as mock_load:
        adapter.load("train")

    _, kwargs = mock_load.call_args
    assert kwargs["split"] == adapter.train_split


def test_load_routes_test_split():
    """load('test') uses test_split, not train_split."""
    fake_ds = _make_fake_ds([{"text": "x", "label": 0}])
    adapter = get_adapter("agnews")

    with patch("datasets.load_dataset",
               return_value=fake_ds) as mock_load:
        adapter.load("test")

    _, kwargs = mock_load.call_args
    assert kwargs["split"] == adapter.test_split


def test_load_unknown_split_raises():
    """load() with an unrecognized split name raises ValueError."""
    adapter = get_adapter("agnews")
    with pytest.raises(ValueError, match="Unknown split"):
        adapter.load("validation")


def test_load_non_standard_column_names():
    """Adapter correctly maps non-standard HF column names (e.g. dbpedia 'content')."""
    raw_rows = [{"content": "some company text", "label": 0}]
    fake_ds = _make_fake_ds(raw_rows)
    adapter = get_adapter("dbpedia")

    with patch("datasets.load_dataset",
               return_value=fake_ds):
        result = adapter.load("train")

    assert result[0]["text"] == "some company text"
    assert result[0]["label"] == 0


def test_load_trec_coarse_label_column():
    """trec adapter maps 'coarse_label' to 'label'."""
    raw_rows = [{"text": "What is X?", "coarse_label": 3}]
    fake_ds = _make_fake_ds(raw_rows)
    adapter = get_adapter("trec")

    with patch("datasets.load_dataset",
               return_value=fake_ds):
        result = adapter.load("train")

    assert result[0]["label"] == 3


def test_load_subsamples_when_num_examples_given():
    """When num_examples < len(ds), shuffle+select is called."""
    fake_ds = _make_fake_ds([{"text": f"t{i}", "label": 0} for i in range(100)])
    fake_ds.shuffle = MagicMock(return_value=fake_ds)
    fake_ds.select = MagicMock(return_value=_make_fake_ds([{"text": "t0", "label": 0}]))
    adapter = get_adapter("agnews")

    with patch("datasets.load_dataset", return_value=fake_ds):
        adapter.load("train", num_examples=1)

    fake_ds.shuffle.assert_called_once_with(seed=42)
    fake_ds.select.assert_called_once_with(range(1))
