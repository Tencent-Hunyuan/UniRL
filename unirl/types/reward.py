"""Shared reward data types."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from PIL import Image

from unirl.distributed.tensor.batch import Batch, concat_field, max_field
from unirl.types.primitives import PrimitiveValue, Texts, primitive_modality_key

PromptSource = Literal["original", "generation"]
PROMPT_SOURCES: tuple[PromptSource, ...] = ("original", "generation")
DEFAULT_PROMPT_SOURCE: PromptSource = "generation"


@dataclass
class RewardRequest:
    """Request for reward computation."""

    generated: Dict[str, PrimitiveValue] = field(default_factory=dict)
    conditioning: Dict[str, PrimitiveValue] = field(default_factory=dict)
    original_prompt: Optional[Texts] = None
    generation_prompt: Optional[Texts] = None
    metadata: Optional[List[Optional[Dict[str, Any]]]] = None
    sample_ids: Optional[List[str]] = None
    group_ids: Optional[List[str]] = None
    audio_sample_rate: Optional[int] = None

    def __post_init__(self) -> None:
        if self.original_prompt is not None and not isinstance(self.original_prompt, Texts):
            raise TypeError(
                f"RewardRequest.original_prompt must be Texts or None, got {type(self.original_prompt).__name__}."
            )
        if self.generation_prompt is not None and not isinstance(self.generation_prompt, Texts):
            raise TypeError(
                f"RewardRequest.generation_prompt must be Texts or None, got {type(self.generation_prompt).__name__}."
            )
        if "text" in self.conditioning:
            raise ValueError(
                "RewardRequest.conditioning must contain only non-text condition media; "
                "use original_prompt/generation_prompt for text."
            )

        aligned: List[tuple[str, int]] = []
        for owner, values in (("generated", self.generated), ("conditioning", self.conditioning)):
            for key, value in values.items():
                actual_key = primitive_modality_key(value)
                if key != actual_key:
                    raise ValueError(
                        f"RewardRequest.{owner}[{key!r}] contains {type(value).__name__}, "
                        f"whose canonical modality key is {actual_key!r}."
                    )
                aligned.append((f"{owner}[{key!r}]", len(value)))
        if self.original_prompt is not None:
            aligned.append(("original_prompt", len(self.original_prompt)))
        if self.generation_prompt is not None:
            aligned.append(("generation_prompt", len(self.generation_prompt)))
        if self.metadata is not None:
            aligned.append(("metadata", len(self.metadata)))
        if self.sample_ids is not None:
            aligned.append(("sample_ids", len(self.sample_ids)))
        if self.group_ids is not None:
            aligned.append(("group_ids", len(self.group_ids)))

        if aligned:
            expected = aligned[0][1]
            mismatched = [f"{name}={size}" for name, size in aligned[1:] if size != expected]
            if mismatched:
                raise ValueError(
                    f"RewardRequest fields must share one batch size; {aligned[0][0]}={expected}, "
                    f"mismatched {', '.join(mismatched)}."
                )

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
        for v in (self.original_prompt, self.generation_prompt):
            if v is not None:
                return len(v)
        for v in self.conditioning.values():
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
    "DEFAULT_PROMPT_SOURCE",
    "PROMPT_SOURCES",
    "PromptSource",
    "RewardRequest",
    "RewardResponse",
]
