"""Shared reward data types."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from unirl.distributed.tensor.batch import Batch, concat_field, max_field, shared_field


class RewardType(Enum):
    """Types of reward computation."""

    IMAGE_TEXT_ALIGNMENT = "image_text_alignment"
    AESTHETIC = "aesthetic"
    CUSTOM = "custom"


@dataclass
class RewardRequest(Batch):
    """Row-aligned, DP-shardable request for reward computation."""

    primitives: Dict[str, Any] = concat_field(default_factory=dict)
    generated: Dict[str, Any] = concat_field(default_factory=dict)
    metadata: Optional[List[Optional[Dict[str, Any]]]] = concat_field(default=None)
    prompt_ids: Optional[List[str]] = concat_field(default=None)
    sample_ids: Optional[List[str]] = concat_field(default=None)
    group_ids: Optional[List[str]] = concat_field(default=None)
    response_lengths: Optional[List[int]] = concat_field(default=None)
    reward_types: List[RewardType] = shared_field(default_factory=lambda: [RewardType.IMAGE_TEXT_ALIGNMENT])
    return_components: bool = shared_field(default=False)
    audio_sample_rate: Optional[int] = shared_field(default=None)
    max_new_tokens: Optional[int] = shared_field(default=None)

    def __post_init__(self) -> None:
        batch_size = self.batch_size
        named_values = [
            *((f"generated[{key!r}]", value) for key, value in self.generated.items() if value is not None),
            *((f"primitives[{key!r}]", value) for key, value in self.primitives.items() if value is not None),
        ]
        for name, value in named_values:
            try:
                size = len(value)
            except TypeError:
                raise TypeError(f"RewardRequest.{name} must be row-aligned and sized, got {type(value).__name__}.")
            if size != batch_size:
                raise ValueError(f"RewardRequest.{name} has {size} rows, expected {batch_size}.")

        for name in ("metadata", "prompt_ids", "sample_ids", "group_ids", "response_lengths"):
            value = getattr(self, name)
            if value is not None and len(value) != batch_size:
                raise ValueError(f"RewardRequest.{name} has {len(value)} rows, expected {batch_size}.")

        if (self.response_lengths is None) != (self.max_new_tokens is None):
            raise ValueError("RewardRequest.response_lengths and max_new_tokens must be provided together.")

    def slice(self, start: int, end: int) -> "RewardRequest":
        """Slice rows through select so packed TensorRefs remain zero-copy views."""
        return self.select(range(int(start), int(end)))

    @property
    def prompts(self) -> List[str]:
        prim = self.primitives.get("text")
        if prim is None:
            return []
        return list(prim.texts)

    @property
    def images(self) -> Optional[List[Union[Image.Image, torch.Tensor]]]:
        prim = self.generated.get("image")
        if prim is None:
            return None
        from unirl.utils.media import tensor_frame_to_pil

        return [tensor_frame_to_pil(image.pixels) for image in prim.to_list()]

    @property
    def videos(self) -> Optional[List[torch.Tensor]]:
        prim = self.generated.get("video")
        if prim is None:
            return None
        return [v.frames.permute(1, 0, 2, 3).contiguous() for v in prim.to_list()]

    @property
    def texts(self) -> Optional[List[str]]:
        prim = self.generated.get("text")
        if prim is None:
            return None
        return list(prim.texts)

    @property
    def audio(self) -> Optional[List[torch.Tensor]]:
        """Generated audio waveforms, one ``[C, L]`` (or ``[L]``) tensor per sample."""
        prim = self.generated.get("audio")
        if prim is None:
            return None
        return [a.waveform for a in prim.to_list()]

    @property
    def batch_size(self) -> int:
        for v in self.generated.values():
            if v is not None:
                return len(v)
        for v in self.primitives.values():
            if v is not None:
                return len(v)
        return 0

    @property
    def is_video(self) -> bool:
        return "video" in self.generated

    @property
    def has_audio(self) -> bool:
        return "audio" in self.generated


@dataclass
class RewardResponse(Batch):
    """Response from reward computation."""

    rewards: List[float] = concat_field(default_factory=list)
    component_rewards: Dict[str, List[float]] = concat_field(default_factory=dict)
    successes: List[bool] = concat_field(default_factory=list)
    errors: List[Optional[str]] = concat_field(default_factory=list)
    compute_time: float = max_field(default=0.0)

    @property
    def batch_size(self) -> int:
        return len(self.rewards)


__all__ = [
    "RewardRequest",
    "RewardResponse",
    "RewardType",
]
