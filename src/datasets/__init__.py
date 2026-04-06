"""
Dataset layer for the private prediction pipeline.

Exports the public interface: DatasetAdapter, TaskSpec,
HFTextClassificationAdapter, and the registry helpers.
"""

from src.datasets.base import DatasetAdapter, TaskSpec
from src.datasets.text_classification import HFTextClassificationAdapter
from src.datasets.registry import REGISTRY, DATASET_CHOICES, get_adapter

__all__ = [
    "DatasetAdapter",
    "TaskSpec",
    "HFTextClassificationAdapter",
    "REGISTRY",
    "DATASET_CHOICES",
    "get_adapter",
]
