"""Qwen3ARConditions — typed conditions container for the Qwen3 AR stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import Condition, TextTokenCondition


@dataclass
class Qwen3ARConditions(Batch):
    """Typed conditions container for the Qwen3 AR stage."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "Qwen3ARConditions":
        """Build from the generic ``Conditions`` dict shape."""
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"Qwen3ARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got "
                f"{type(prompt).__name__ if prompt is not None else 'None'}"
            )
        return cls(prompt=prompt)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape for packing into ``Part.conditions``."""
        if self.prompt is None:
            raise ValueError("Qwen3ARConditions.to_dict: prompt field is None")
        return {"prompt": self.prompt}


__all__ = ["Qwen3ARConditions"]
