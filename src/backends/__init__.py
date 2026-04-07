"""
Model backend abstractions (Phase 3).

Public surface:
    ModelBackend               -- abstract base class
    HuggingFaceCausalLM        -- HF AutoModelForCausalLM + AutoTokenizer backend
"""

from src.backends.base import ModelBackend
from src.backends.huggingface_causal_lm import HuggingFaceCausalLM

__all__ = ["ModelBackend", "HuggingFaceCausalLM"]
