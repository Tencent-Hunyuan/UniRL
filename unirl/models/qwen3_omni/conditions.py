"""Typed text and TMRoPE video conditions for Qwen3-Omni."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition


@dataclass
class Qwen3OmniARConditions(Batch):
    """AR inputs; per-sample media fields must remain ``FieldKind.CONCAT``."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    pixel_values_videos: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    video_grid_thw: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    video_second_per_grid: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    input_features: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    feature_attention_mask: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Qwen3OmniARConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                f"Qwen3OmniARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got {type(prompt).__name__ if prompt is not None else 'None'}"
            )
        return cls(
            prompt=prompt,
            pixel_values_videos=d.get("pixel_values_videos"),
            video_grid_thw=d.get("video_grid_thw"),
            video_second_per_grid=d.get("video_second_per_grid"),
            input_features=d.get("input_features"),
            feature_attention_mask=d.get("feature_attention_mask"),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("Qwen3OmniARConditions.to_dict: prompt field is None")
        out: Dict[str, Any] = {"prompt": self.prompt}
        if self.pixel_values_videos is not None:
            out["pixel_values_videos"] = self.pixel_values_videos
        if self.video_grid_thw is not None:
            out["video_grid_thw"] = self.video_grid_thw
        if self.video_second_per_grid is not None:
            out["video_second_per_grid"] = self.video_second_per_grid
        if self.input_features is not None:
            out["input_features"] = self.input_features
        if self.feature_attention_mask is not None:
            out["feature_attention_mask"] = self.feature_attention_mask
        return out


__all__ = ["Qwen3OmniARConditions"]
