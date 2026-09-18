"""WAN21TextEmbedStage — UMT5 prompt encoding → TextEmbedCondition."""

from __future__ import annotations

from typing import Any, List, Protocol, runtime_checkable

import torch

from unirl.models.types.embedding import EmbedStage
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts


@runtime_checkable
class _TextEncoderBundle(Protocol):
    """Structural Protocol for bundles this stage can encode against."""

    text_encoder: Any
    tokenizer: Any
    device: torch.device
    max_sequence_length: int


class WAN21TextEmbedStage(EmbedStage[Texts, TextEmbedCondition]):
    """WAN 2.1 UMT5 text → TextEmbedCondition stage."""

    def __init__(
        self,
        bundle: _TextEncoderBundle,
        *,
        max_sequence_length: int = 512,
    ) -> None:
        self.bundle = bundle
        self.max_sequence_length = int(
            max_sequence_length if max_sequence_length is not None else bundle.max_sequence_length
        )

    def embed(self, p: Texts) -> TextEmbedCondition:
        """Encode prompts into a ``TextEmbedCondition``."""
        return self._encode(list(p.texts))

    def _encode(self, prompts: List[str]) -> TextEmbedCondition:
        bundle = self.bundle
        device = bundle.device

        text_inputs = bundle.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        with torch.no_grad():
            encoder_out = bundle.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            embeds = encoder_out.last_hidden_state

        embeds = embeds * attention_mask.unsqueeze(-1).to(dtype=embeds.dtype)

        return TextEmbedCondition(
            embeds=embeds,
            pooled=None,
            attn_mask=attention_mask,
        )


__all__ = ["WAN21TextEmbedStage"]
