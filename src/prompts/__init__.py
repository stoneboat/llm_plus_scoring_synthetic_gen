"""
Prompt-construction layer for the private prediction pipeline.

Exports the public interface: PromptBuilder, TextClassificationPromptBuilder,
PROMPT_TEMPLATES, and the backward-compatible build_prompts helper.
"""

from src.prompts.base import PromptBuilder
from src.prompts.text_classification import (
    PROMPT_TEMPLATES,
    TextClassificationPromptBuilder,
    build_prompts,
)

__all__ = [
    "PromptBuilder",
    "TextClassificationPromptBuilder",
    "PROMPT_TEMPLATES",
    "build_prompts",
]
