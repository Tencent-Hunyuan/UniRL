"""SD3Conditions — typed conditions container for the SD3 diffusion stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextEmbedCondition


@dataclass
class SD3Conditions(Batch):
    """Typed conditions container for SD3 diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "SD3Conditions":
        """Build from the generic ``Conditions`` dict shape."""
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"SD3Conditions.from_dict: expected d['text'] to be a TextEmbedCondition, "
                f"got {type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"SD3Conditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        return cls(text=text, negative_text=negative_text)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for packing into ``Part.conditions``."""
        if self.text is None:
            raise ValueError("SD3Conditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        return out


__all__ = ["SD3Conditions"]
