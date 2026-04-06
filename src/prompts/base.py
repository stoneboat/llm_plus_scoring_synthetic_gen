"""
Abstract base class for prompt builders.

A PromptBuilder owns prompt rendering for one task type.  It accepts
example dicts and a tokenizer and returns ready-to-use prompt strings.
The concrete implementations keep the task-specific templates and public-seed
logic, which previously lived inline in src/generate.py::build_prompts.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class PromptBuilder(ABC):
    """Abstract prompt builder.

    Implementations must produce prompt strings that are directly fed
    to the LLM.  When a tokenizer is supplied they must wrap the content
    in the model's native chat template (via tokenizer.apply_chat_template).
    """

    @abstractmethod
    def build_prompts(
        self,
        examples: List[dict],
        text_column: str,
        label: int,
        tokenizer=None,
    ) -> Tuple[List[str], str]:
        """Build private prompts and the corresponding public prompt.

        Args:
            examples: batch of example dicts (each has at least text_column).
            text_column: key for the text field in each dict.
            label: integer label for this batch.
            tokenizer: optional HuggingFace tokenizer.  When provided the
                prompts are wrapped in the model's chat template.

        Returns:
            (private_prompts, public_prompt) — a list of per-example private
            prompt strings and a single public (non-sensitive) prompt string.
        """
        ...
