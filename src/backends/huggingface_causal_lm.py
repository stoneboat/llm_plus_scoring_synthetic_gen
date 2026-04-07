"""
HuggingFace causal-LM backend.

Wraps a ``transformers`` model + tokenizer pair.  All tokenization,
micro-batch inference, padding-side management, and decoding live here so
that the mechanism layer remains framework-agnostic.
"""

from typing import List, Optional

import torch
from torch import Tensor

from src.backends.base import ModelBackend


class HuggingFaceCausalLM(ModelBackend):
    """Concrete backend for HuggingFace AutoModelForCausalLM + AutoTokenizer.

    Args:
        model: a HuggingFace causal language model (already loaded, on
            ``device``).
        tokenizer: the corresponding tokenizer.
        device: torch device string (default ``"cuda"``).
        micro_batch_size: number of prompts to forward in one pass.
            Reduce if you encounter GPU OOM errors on large-vocabulary
            models such as Gemma-2 (256k vocab).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = "cuda",
        micro_batch_size: int = 32,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._micro_batch_size = micro_batch_size

    # ------------------------------------------------------------------
    # ModelBackend interface
    # ------------------------------------------------------------------

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._tokenizer.eos_token_id

    @property
    def padding_side(self) -> str:
        return self._tokenizer.padding_side

    @padding_side.setter
    def padding_side(self, side: str) -> None:
        self._tokenizer.padding_side = side

    @property
    def tokenizer(self):
        """Expose the raw tokenizer (needed by the prompt builder for chat templates)."""
        return self._tokenizer

    @torch.no_grad()
    def get_next_token_logits(
        self,
        prompts: List[str],
        generated_tokens: List[int],
    ) -> Tensor:
        """Run LLM inference to get next-token logits for a set of prompts.

        Each prompt is concatenated with the generated tokens so far, then
        we extract the logits at the last position.

        The tokenizer's ``padding_side`` must be ``"left"`` before calling
        this method.  Causal LMs require left-padding so that position ``-1``
        always corresponds to the last real token for every sequence in the
        batch.

        Processes prompts in micro-batches to avoid GPU OOM when ``batch_size``
        is large (e.g. 255) and the model has a large vocabulary (e.g.
        Gemma-2 with 256k tokens).

        Returns:
            Logits tensor of shape ``(len(prompts), vocab_size)``.
        """
        all_last_logits: List[Tensor] = []

        for i in range(0, len(prompts), self._micro_batch_size):
            chunk = prompts[i : i + self._micro_batch_size]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            # Preserve exact continuation tokens by appending token IDs
            # directly.  Decoding then re-tokenizing can alter
            # whitespace/subword boundaries.
            if generated_tokens:
                batch_len = inputs["input_ids"].shape[0]
                gen = torch.tensor(
                    generated_tokens,
                    dtype=inputs["input_ids"].dtype,
                    device=self._device,
                ).unsqueeze(0).expand(batch_len, -1)
                gen_attn = torch.ones(
                    (batch_len, gen.shape[1]),
                    dtype=inputs["attention_mask"].dtype,
                    device=self._device,
                )
                inputs = {
                    "input_ids": torch.cat([inputs["input_ids"], gen], dim=1),
                    "attention_mask": torch.cat(
                        [inputs["attention_mask"], gen_attn], dim=1
                    ),
                }

            outputs = self._model(**inputs)
            last_logits = outputs.logits[:, -1, :].cpu()
            all_last_logits.append(last_logits)

        return torch.cat(all_last_logits, dim=0).to(self._device)

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
