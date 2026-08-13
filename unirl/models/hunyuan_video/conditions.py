"""HunyuanVideoConditions — ``text_llama [B, seq, 4096]`` plus ``pooled_clip`` embeds ``[B, 768]``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import (
    Condition,
    TextEmbedCondition,
)


@dataclass
class HunyuanVideoConditions(Batch):
    """Typed conditions container for HunyuanVideo-1.0 diffusion."""

    text_llama: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, transport=True, default=None)
    pooled_clip: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, transport=True, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "HunyuanVideoConditions":
        """Build from the generic ``Conditions`` dict shape."""
        text_llama = d.get("text_llama")
        pooled_clip = d.get("pooled_clip")
        if not isinstance(text_llama, TextEmbedCondition):
            raise TypeError(
                f"HunyuanVideoConditions.from_dict: expected d['text_llama'] "
                f"to be a TextEmbedCondition, got "
                f"{type(text_llama).__name__ if text_llama is not None else 'None'}"
            )
        if not isinstance(pooled_clip, TextEmbedCondition):
            raise TypeError(
                f"HunyuanVideoConditions.from_dict: expected d['pooled_clip'] "
                f"to be a TextEmbedCondition, got "
                f"{type(pooled_clip).__name__ if pooled_clip is not None else 'None'}"
            )
        return cls(
            text_llama=text_llama,
            pooled_clip=pooled_clip,
        )

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for packing into ``Part.conditions``."""
        if self.text_llama is None or self.pooled_clip is None:
            raise ValueError(
                "HunyuanVideoConditions.to_dict: both text_llama and pooled_clip "
                "must be set (the transformer requires both encoder streams)."
            )
        out: Dict[str, Condition] = {
            "text_llama": self.text_llama,
            "pooled_clip": self.pooled_clip,
        }
        return out


__all__ = ["HunyuanVideoConditions"]
