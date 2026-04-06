"""
Dataset registry for the private prediction pipeline.

REGISTRY maps canonical dataset names to their HFTextClassificationAdapter
instances.  This is the single authoritative source for:
  - HuggingFace dataset IDs,
  - train / test split names,
  - raw column name mappings,
  - number of labels and label names.

Previously, equivalent (but inconsistently structured) DATASET_HF_MAP dicts
were duplicated in each of the three scripts.  All scripts now import from
here instead.

To add a new dataset later, add one entry to REGISTRY.  Nothing else changes.
"""

from typing import Dict

from src.datasets.base import DatasetAdapter, TaskSpec
from src.datasets.text_classification import HFTextClassificationAdapter


REGISTRY: Dict[str, HFTextClassificationAdapter] = {
    "agnews": HFTextClassificationAdapter(
        name="agnews",
        hf_name="fancyzhx/ag_news",
        train_split="train",
        test_split="test",
        hf_text_column="text",
        hf_label_column="label",
        task=TaskSpec(
            num_labels=4,
            label_names={0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
        ),
    ),
    "dbpedia": HFTextClassificationAdapter(
        name="dbpedia",
        hf_name="fancyzhx/dbpedia_14",
        train_split="train",
        test_split="test",
        hf_text_column="content",
        hf_label_column="label",
        task=TaskSpec(
            num_labels=14,
            label_names={
                0: "Company", 1: "School", 2: "Artist", 3: "Athlete",
                4: "Politician", 5: "Transportation", 6: "Building",
                7: "Nature", 8: "Village", 9: "Animal", 10: "Plant",
                11: "Album", 12: "Film", 13: "Book",
            },
        ),
    ),
    "imdb": HFTextClassificationAdapter(
        name="imdb",
        hf_name="stanfordnlp/imdb",
        train_split="train",
        test_split="test",
        hf_text_column="text",
        hf_label_column="label",
        task=TaskSpec(
            num_labels=2,
            label_names={0: "Negative", 1: "Positive"},
        ),
    ),
    "yelp": HFTextClassificationAdapter(
        name="yelp",
        hf_name="fancyzhx/yelp_polarity",
        train_split="train",
        test_split="test",
        hf_text_column="text",
        hf_label_column="label",
        task=TaskSpec(
            num_labels=2,
            label_names={0: "Negative", 1: "Positive"},
        ),
    ),
    "trec": HFTextClassificationAdapter(
        name="trec",
        hf_name="CogComp/trec",
        train_split="train",
        test_split="test",
        hf_text_column="text",
        hf_label_column="coarse_label",
        task=TaskSpec(
            num_labels=6,
            label_names={
                0: "Abbreviation", 1: "Entity", 2: "Description",
                3: "Human", 4: "Location", 5: "Number",
            },
        ),
    ),
}

#: Sorted list of supported dataset names, for use in argparse choices.
DATASET_CHOICES = sorted(REGISTRY.keys())


def get_adapter(name: str) -> DatasetAdapter:
    """Look up a DatasetAdapter by canonical name.

    Args:
        name: dataset name (e.g. "agnews").

    Returns:
        The corresponding DatasetAdapter.

    Raises:
        ValueError: if name is not in the registry.
    """
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported datasets: {DATASET_CHOICES}"
        )
    return REGISTRY[name]
