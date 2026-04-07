"""
Abstract base class for model backends.

A ModelBackend encapsulates all HuggingFace (or other framework) specifics:
tokenization, micro-batch inference, padding-side management, and decoding.
The generation and mechanism layers interact with the backend only through
this interface, so swapping the underlying model framework is isolated to a
single concrete subclass.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor


class ModelBackend(ABC):
    """Interface between the generation mechanism and the underlying LLM."""

    @property
    @abstractmethod
    def eos_token_id(self) -> Optional[int]:
        """The end-of-sequence token ID (None if the model has no EOS)."""

    @property
    @abstractmethod
    def padding_side(self) -> str:
        """Current tokenizer padding side ('left' or 'right')."""

    @padding_side.setter
    @abstractmethod
    def padding_side(self, side: str) -> None:
        """Set the tokenizer padding side."""

    @abstractmethod
    def get_next_token_logits(
        self,
        prompts: List[str],
        generated_tokens: List[int],
    ) -> Tensor:
        """Return next-token logits for every prompt.

        Args:
            prompts: list of prompt strings.
            generated_tokens: token IDs generated so far, appended verbatim
                to each prompt before inference.

        Returns:
            Float tensor of shape ``(len(prompts), vocab_size)`` on the
            backend's device.
        """

    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs to a string."""
