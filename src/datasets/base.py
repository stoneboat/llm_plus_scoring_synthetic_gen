"""
Abstract base types for the dataset layer.

TaskSpec carries the task-level metadata (label vocabulary) that is needed
by prompt builders and downstream evaluators.

DatasetAdapter is the abstract base for dataset connectors.  Concrete
implementations hide HuggingFace dataset IDs, column name mappings, and
split names from the rest of the pipeline.

The normalized example shape is a plain dict with "text" (str) and
"label" (int) keys — the same shape used throughout the current codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TaskSpec:
    """Task-level metadata for a classification dataset.

    Attributes:
        num_labels: number of distinct integer labels.
        label_names: mapping from integer label to human-readable name.
    """

    num_labels: int
    label_names: Dict[int, str]


class DatasetAdapter(ABC):
    """Abstract interface for dataset connectors.

    A DatasetAdapter knows how to load a dataset split and normalize
    its rows into the pipeline's common shape: {"text": str, "label": int}.

    It also exposes task-level metadata (num_labels, label_names) so that
    prompt builders and evaluators do not need to hard-code per-dataset
    constants.
    """

    #: Canonical short name used in CLI args, filenames, and registry keys.
    name: str

    #: Task metadata (label vocabulary).
    task: TaskSpec

    @abstractmethod
    def load(
        self,
        split: str,
        num_examples: Optional[int] = None,
        cache_dir: str = "data/datasets",
    ) -> List[dict]:
        """Load a dataset split and return normalized example dicts.

        Args:
            split: "train" or "test".
            num_examples: if given, subsample this many examples with seed=42.
            cache_dir: HuggingFace cache directory.

        Returns:
            List of dicts with keys "text" (str) and "label" (int).
        """
        ...

    # ------------------------------------------------------------------
    # Convenience properties (delegate to self.task)
    # ------------------------------------------------------------------

    @property
    def num_labels(self) -> int:
        return self.task.num_labels

    @property
    def label_names(self) -> Dict[int, str]:
        return self.task.label_names
