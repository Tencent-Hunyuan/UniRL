"""MiniMax-H3 conditions -- typed container for the diffusion stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from unirl.config.require import require
from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.conditions import TextEmbedCondition


@dataclass
class MiniMaxH3Conditions(Batch):
    """Conditions passed to the MiniMax-H3 diffusion stage."""

    text: Optional[TextEmbedCondition] = concat_field(default=None)

    @classmethod
    def from_dict(cls, d: dict) -> "MiniMaxH3Conditions":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return {name: value for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}

    def trim_text_padding(self) -> "MiniMaxH3Conditions":
        """Remove right-padding before building the packed H3 sequence."""
        require(
            self.text is not None and self.text.embeds is not None,
            "MiniMaxH3Conditions.trim_text_padding: text embeddings are required",
        )
        embeds = self.text.embeds
        mask = self.text.attn_mask
        if mask is None:
            return self
        require(
            mask.ndim == 2 and tuple(mask.shape) == tuple(embeds.shape[:2]),
            f"MiniMaxH3Conditions.trim_text_padding: attention mask shape {tuple(mask.shape)} must match "
            f"text embedding batch/sequence shape {tuple(embeds.shape[:2])}",
        )
        valid = mask.to(dtype=torch.bool)
        lengths = valid.sum(dim=1)
        require(
            lengths.numel() > 0 and bool((lengths == lengths[0]).all()),
            f"MiniMaxH3Conditions.trim_text_padding: one packed forward requires one text length, got "
            f"{lengths.tolist()}; slice to micro_batch_size=1 first",
        )
        expected = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
        require(
            torch.equal(valid, expected),
            "MiniMaxH3Conditions.trim_text_padding: attention mask must be contiguous right-padding",
        )
        length = int(lengths[0].item())
        if length == int(embeds.shape[1]):
            return self
        return type(self)(
            text=TextEmbedCondition(
                embeds=embeds[:, :length],
                pooled=self.text.pooled,
                attn_mask=mask[:, :length],
            )
        )


__all__ = ["MiniMaxH3Conditions"]
