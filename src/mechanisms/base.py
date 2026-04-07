"""
Abstract base class for generation mechanisms.

A Mechanism owns the per-token decision loop: clipped-logit aggregation,
SVT gating, top-k filtering, and private/public token accounting.
It depends on a ModelBackend for inference and decoding, but is otherwise
independent of the model framework.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from src.backends.base import ModelBackend


class Mechanism(ABC):
    """Interface for a token-generation mechanism (Algorithm 1, inner loop)."""

    @abstractmethod
    def generate_example(
        self,
        private_prompts: List[str],
        public_prompt: str,
        backend: ModelBackend,
        remaining_budget: int,
        max_total_tokens: int,
    ) -> Tuple[List[int], int, int]:
        """Generate one synthetic example.

        Produces tokens until EOS, ``max_total_tokens``, or
        ``remaining_budget`` private tokens are exhausted.

        Args:
            private_prompts: the per-example private prompt strings.
            public_prompt: the shared public (non-sensitive) prompt.
            backend: model backend used for inference.
            remaining_budget: how many private tokens this example may
                consume from the batch's total budget *r*.
            max_total_tokens: hard cap on total tokens (private + public).

        Returns:
            ``(token_ids, num_private_tokens, num_public_tokens)``
        """
