from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition


@dataclass
class QwenVLARConditions(Batch):
    """Conditions for Qwen2.5-VL autoregressive generation.

    pixel_values and image_grid_thw are stored as per-sample lists
    (FieldKind.CONCAT).  Each list element is one sample's tensor, so
    the list length equals batch_size and the Batched framework
    automatically extends / slices / selects it alongside prompt.
    This ensures correct handling when multiple rollout workers'
    conditions are concatenated — SHARED fields would silently drop
    all but the first worker's image data.
    """

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    pixel_values: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    image_grid_thw: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QwenVLARConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"QwenVLARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got "
                f"{type(prompt).__name__ if prompt is not None else 'None'}"
            )
        return cls(
            prompt=prompt,
            pixel_values=d.get("pixel_values"),
            image_grid_thw=d.get("image_grid_thw"),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("QwenVLARConditions.to_dict: prompt field is None")
        return {
            "prompt": self.prompt,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
        }


__all__ = ["QwenVLARConditions"]
