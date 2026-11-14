"""Shared reward data types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from PIL import Image

from unirl.distributed.tensor.batch import Batch, concat_field, max_field, shared_field
from unirl.types.primitives import PrimitiveValue, Texts, primitive_modality_key

PromptSource = Literal["original", "generation"]
PROMPT_SOURCES: tuple[PromptSource, ...] = ("original", "generation")
DEFAULT_PROMPT_SOURCE: PromptSource = "generation"


@dataclass
class RewardRequest(Batch):
    """Row-aligned, DP-shardable request for reward computation."""

    generated: Dict[str, PrimitiveValue] = concat_field(default_factory=dict)
    conditioning: Dict[str, PrimitiveValue] = concat_field(default_factory=dict)
    original_prompt: Optional[Texts] = concat_field(default=None)
    generation_prompt: Optional[Texts] = concat_field(default=None)
    metadata: Optional[List[Optional[Dict[str, Any]]]] = concat_field(default=None)
    sample_ids: Optional[List[str]] = concat_field(default=None)
    group_ids: Optional[List[str]] = concat_field(default=None)
    # Generated AR token counts; RewardService shapes truncated rows against max_new_tokens.
    response_lengths: Optional[List[int]] = concat_field(default=None)
    audio_sample_rate: Optional[int] = shared_field(default=None)
    max_new_tokens: Optional[int] = shared_field(default=None)

    def __post_init__(self) -> None:
        if (self.response_lengths is None) != (self.max_new_tokens is None):
            raise ValueError("RewardRequest.response_lengths and max_new_tokens must be provided together.")
        if "text" in self.conditioning:
            raise ValueError(
                "RewardRequest.conditioning must contain only non-text condition media; "
                "use original_prompt/generation_prompt for text."
            )

        batch_sizes: Dict[str, int] = {}
        for owner, values in (("generated", self.generated), ("conditioning", self.conditioning)):
            for key, value in values.items():
                actual_key = primitive_modality_key(value)
                if key != actual_key:
                    raise ValueError(
                        f"RewardRequest.{owner}[{key!r}] contains {type(value).__name__}, "
                        f"whose canonical modality key is {actual_key!r}."
                    )
                batch_sizes[f"{owner}[{key!r}]"] = len(value)
        for name, value in (
            ("original_prompt", self.original_prompt),
            ("generation_prompt", self.generation_prompt),
            ("metadata", self.metadata),
            ("sample_ids", self.sample_ids),
            ("group_ids", self.group_ids),
            ("response_lengths", self.response_lengths),
        ):
            if value is not None:
                batch_sizes[name] = len(value)

        if len(set(batch_sizes.values())) > 1:
            details = ", ".join(f"{name}={size}" for name, size in batch_sizes.items())
            raise ValueError(f"RewardRequest fields must share one batch size; got {details}.")

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
