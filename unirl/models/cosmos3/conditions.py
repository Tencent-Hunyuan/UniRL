"""Typed conditions for Cosmos3's packed omnimodal stage (supervised and generation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field, packed_field
from unirl.types.conditions import Condition, Modality, TextTokenCondition


@dataclass
class Cosmos3SFTCondition(Condition):
    """Per-sample packed text tokens + optional action targets beside a clean video latent (README.md)."""

    modality: ClassVar[Modality] = Modality.MULTIMODAL

    input_ids: Optional[torch.Tensor] = packed_field(default=None)
    fps: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    flow_shifts: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    actions: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)


@dataclass
class Cosmos3Conditions(Batch):
    """Generation conditions: right-padded cond / uncond prompt token ids ``[B, L]`` with masks."""

    text: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "Cosmos3Conditions":
        """Build from the generic ``Part.conditions`` dict."""
        text = d.get("text")
        if not isinstance(text, TextTokenCondition):
            raise TypeError(
                f"Cosmos3Conditions.from_dict: expected d['text'] to be a TextTokenCondition, "
                f"got {type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextTokenCondition):
            raise TypeError(
                f"Cosmos3Conditions.from_dict: expected d['negative_text'] to be a "
                f"TextTokenCondition or absent, got {type(negative_text).__name__}"
            )
        return cls(text=text, negative_text=negative_text)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Part.conditions`` dict."""
        if self.text is None:
            raise ValueError("Cosmos3Conditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        return out


__all__ = ["Cosmos3Conditions", "Cosmos3SFTCondition"]
