"""Typed text and multimodal TMRoPE conditions for Qwen3-Omni."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition

_PAIR_FIELDS = (
    ("pixel_values", "image_grid_thw"),
    ("pixel_values_videos", "video_grid_thw"),
    ("input_features", "feature_attention_mask"),
)

_LIST_FIELDS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "video_second_per_grid",
    "input_features",
    "feature_attention_mask",
    "use_audio_in_video",
)


@dataclass
class Qwen3OmniARConditions(Batch):
    """AR inputs; per-sample media fields must remain ``FieldKind.CONCAT``."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    pixel_values: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    image_grid_thw: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    pixel_values_videos: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    video_grid_thw: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    video_second_per_grid: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    input_features: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    feature_attention_mask: Optional[List[Any]] = field(kind=FieldKind.CONCAT, default=None)
    use_audio_in_video: Optional[List[bool]] = field(kind=FieldKind.CONCAT, default=None)

    def __post_init__(self) -> None:
        lengths: Dict[str, int] = {}
        if self.prompt is not None and getattr(self.prompt, "input_ids", None) is not None:
            lengths["prompt"] = int(self.prompt.input_ids.shape[0])
        for name in _LIST_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, list):
                raise TypeError(f"Qwen3OmniARConditions.{name} must be a list or None, got {type(value).__name__}.")
            lengths[name] = len(value)
        if lengths:
            batch_size = next(iter(lengths.values()))
            mismatched = {name: size for name, size in lengths.items() if size != batch_size}
            if mismatched:
                raise ValueError(
                    f"Qwen3OmniARConditions per-sample media lists must share one batch size; got {lengths}."
                )
        for left, right in _PAIR_FIELDS:
            left_value = getattr(self, left)
            right_value = getattr(self, right)
            if (left_value is None) != (right_value is None):
                raise ValueError(f"Qwen3OmniARConditions.{left} and .{right} must both be set or both be None.")

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
            pixel_values=d.get("pixel_values"),
            image_grid_thw=d.get("image_grid_thw"),
            pixel_values_videos=d.get("pixel_values_videos"),
            video_grid_thw=d.get("video_grid_thw"),
            video_second_per_grid=d.get("video_second_per_grid"),
            input_features=d.get("input_features"),
            feature_attention_mask=d.get("feature_attention_mask"),
            use_audio_in_video=d.get("use_audio_in_video"),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("Qwen3OmniARConditions.to_dict: prompt field is None")
        out: Dict[str, Any] = {"prompt": self.prompt}
        if self.pixel_values is not None:
            out["pixel_values"] = self.pixel_values
        if self.image_grid_thw is not None:
            out["image_grid_thw"] = self.image_grid_thw
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
        if self.use_audio_in_video is not None:
            out["use_audio_in_video"] = self.use_audio_in_video
        return out


__all__ = ["Qwen3OmniARConditions"]
