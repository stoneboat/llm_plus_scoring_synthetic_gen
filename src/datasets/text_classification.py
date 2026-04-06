"""
Concrete DatasetAdapter for HuggingFace text-classification datasets.

HFTextClassificationAdapter handles the column-name mapping, split routing,
and subsampling that was previously duplicated across the three scripts.

All five datasets currently supported by the pipeline are configured in
src/datasets/registry.py using this adapter.
"""

from dataclasses import dataclass
from typing import List, Optional

from src.datasets.base import DatasetAdapter, TaskSpec


@dataclass
class HFTextClassificationAdapter(DatasetAdapter):
    """DatasetAdapter backed by a HuggingFace text-classification dataset.

    Attributes:
        name: canonical short name (e.g. "agnews").
        hf_name: HuggingFace dataset identifier (e.g. "fancyzhx/ag_news").
        train_split: HF split name for training data (usually "train").
        test_split: HF split name for evaluation data (usually "test").
        hf_text_column: raw column name for text in the HF dataset.
        hf_label_column: raw column name for the integer label.
        task: TaskSpec with num_labels and label_names.
    """

    name: str
    hf_name: str
    train_split: str
    test_split: str
    hf_text_column: str
    hf_label_column: str
    task: TaskSpec

    def load(
        self,
        split: str,
        num_examples: Optional[int] = None,
        cache_dir: str = "data/datasets",
    ) -> List[dict]:
        """Load and normalize a dataset split.

        Args:
            split: "train" or "test".
            num_examples: subsample to this many examples (shuffle seed=42).
            cache_dir: HuggingFace cache directory.

        Returns:
            List of {"text": str, "label": int} dicts.

        Raises:
            ValueError: if split is not "train" or "test".
        """
        from datasets import load_dataset  # deferred so tests can mock it

        if split == "train":
            hf_split = self.train_split
        elif split == "test":
            hf_split = self.test_split
        else:
            raise ValueError(
                f"Unknown split '{split}' for dataset '{self.name}'. "
                "Use 'train' or 'test'."
            )

        ds = load_dataset(self.hf_name, split=hf_split, cache_dir=cache_dir)

        if num_examples is not None and num_examples < len(ds):
            ds = ds.shuffle(seed=42).select(range(num_examples))

        return [
            {
                "text": row[self.hf_text_column],
                "label": row[self.hf_label_column],
            }
            for row in ds
        ]
