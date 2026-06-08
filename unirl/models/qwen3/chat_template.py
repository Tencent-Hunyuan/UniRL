"""Qwen3ChatTemplateStage — ``Texts → Qwen3ARConditions``.

Implements ``EmbedStage[Texts, Qwen3ARConditions]`` from
:mod:`unirl.models.types.embedding`. Applies the bundle
tokenizer's chat template (with ``add_generation_prompt=True``) so the
AR stage starts from the canonical assistant-turn prefix the model was
trained against.

An optional ``system_instruction`` string is prepended as a ``system``
message — callers that need byte-for-byte parity with an SFT template
(e.g. Qwen3's ``qwen3_nothink`` recipe) pass the exact string here. The
stage does not interpret it.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Texts

from .bundle import Qwen3Bundle
from .conditions import Qwen3ARConditions


class Qwen3ChatTemplateStage(EmbedStage[Texts, Qwen3ARConditions]):
    """Apply the Qwen3 chat template, right-pad in batch, return AR conditions."""

    def __init__(
        self,
        bundle: Qwen3Bundle,
        *,
        system_instruction: Optional[str] = None,
        max_prompt_length: int = 4096,
        enable_thinking: bool = False,
    ) -> None:
        self.bundle = bundle
        self.system_instruction = system_instruction
        self.max_prompt_length = int(max_prompt_length)
        # MUST agree with the rollout engine's chat template
        # (rollout.config.chat_template_kwargs.enable_thinking) or train/rollout
        # prompts diverge and the importance ratio breaks.
        self.enable_thinking = bool(enable_thinking)

    def embed(self, p: Texts) -> Qwen3ARConditions:
        """Tokenize ``p.texts`` via the chat template and pack into AR conditions."""
        tokenizer = self.bundle.tokenizer
        device = self.bundle.device

        per_sample_ids: List[torch.Tensor] = []
        for text in p.texts:
            messages: List[dict] = []
            if self.system_instruction is not None:
                messages.append({"role": "system", "content": self.system_instruction})
            messages.append({"role": "user", "content": text})
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
                tokenize=True,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_prompt_length,
            )
            # apply_chat_template returns [1, L]; squeeze the leading dim.
            per_sample_ids.append(ids[0].to(device=device, dtype=torch.long))

        # Right-pad to the in-batch max so the AR loop can use a single tensor.
        max_len = max(int(t.shape[0]) for t in per_sample_ids)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError(
                "Qwen3ChatTemplateStage.embed: tokenizer has no pad_token_id; "
                "Qwen3Bundle.from_config sets pad_token=eos_token when absent — "
                "check the bundle was constructed via from_config."
            )

        batch = len(per_sample_ids)
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
        for i, t in enumerate(per_sample_ids):
            n = int(t.shape[0])
            input_ids[i, :n] = t
            attention_mask[i, :n] = 1

        return Qwen3ARConditions(prompt=TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask))


__all__ = ["Qwen3ChatTemplateStage"]
