"""Video reward scorer."""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

from .registry import (
    resolve_builtin_reward_scorer_class,
    resolve_builtin_reward_spec_class,
)


class VideoRewardScorer(RewardBackend):
    """Specialized reward scorer for video generation."""

    input_kind = "video"

    def __init__(self, *, config: "VideoSpec", base_device: str) -> None:
        inner_model = str(config.inner_model_name or "pickscore").strip().lower()
        super().__init__(
            model_name=inner_model,
            batch_size=config.batch_size,
        )
        self.temporal_weight = config.temporal_weight
        self.alignment_weight = config.alignment_weight
        self.sample_frames = config.sample_frames

        inner_scorer_cls = resolve_builtin_reward_scorer_class(inner_model)
        inner_spec_cls = resolve_builtin_reward_spec_class(inner_model)
        inner_spec = inner_spec_cls()
        # Outer VideoSpec overrides what the inner Spec accepts. OCR has no
        # device/batch_size; everything else does — hasattr keeps this generic.
        overrides = {
            field_name: getattr(config, field_name)
            for field_name in ("device", "batch_size")
            if hasattr(inner_spec, field_name)
        }
        if overrides:
            inner_spec = dataclasses.replace(inner_spec, **overrides)
        self.frame_scorer = inner_scorer_cls(config=inner_spec, base_device=base_device)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        if not request.is_video:
            return self.frame_scorer.compute_rewards(request)

        start = time.time()
        videos = request.videos
        prompts = request.prompts

        try:
            rewards = []
            component_rewards = {
                "alignment": [],
                "temporal": [],
            }

            for video, prompt in zip(videos, prompts):
                frames = self._sample_frames(video)
                from torchvision.transforms.functional import to_tensor

                from unirl.types.primitives import Images, Texts

                frame_pixels = torch.stack([to_tensor(f) for f in frames])
                frame_request = RewardRequest(
                    primitives={"text": Texts(texts=[prompt] * len(frames))},
                    generated={"image": Images(pixels=frame_pixels)},
                )
                frame_response = self.frame_scorer.compute_rewards(frame_request)
                alignment_reward = sum(frame_response.rewards) / len(frame_response.rewards)
                temporal_reward = self._compute_temporal_consistency(video)
                total_reward = self.alignment_weight * alignment_reward + self.temporal_weight * temporal_reward

                rewards.append(total_reward)
                component_rewards["alignment"].append(alignment_reward)
                component_rewards["temporal"].append(temporal_reward)

            return RewardResponse(
                rewards=rewards,
                component_rewards=component_rewards,
                successes=[True] * len(rewards),
                errors=[None] * len(rewards),
                compute_time=time.time() - start,
            )
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * len(videos),
                successes=[False] * len(videos),
                errors=[str(e)] * len(videos),
                compute_time=time.time() - start,
            )

    def _sample_frames(self, video: torch.Tensor) -> List[Image.Image]:
        from torchvision.transforms.functional import to_pil_image

        if video.dim() == 4:
            video = video.permute(1, 0, 2, 3)
        elif video.dim() == 5:
            video = video.squeeze(0).permute(1, 0, 2, 3)

        num_frames = video.shape[0]
        indices = torch.linspace(0, num_frames - 1, self.sample_frames).long()

        frames = []
        for idx in indices:
            frame = video[idx]
            if frame.max() <= 1.0:
                frame = (frame * 255).byte()
            frames.append(to_pil_image(frame))

        return frames

    def _compute_temporal_consistency(self, video: torch.Tensor) -> float:
        if video.dim() == 4:
            video = video.permute(1, 0, 2, 3)
        elif video.dim() == 5:
            video = video.squeeze(0).permute(1, 0, 2, 3)

        frame_diffs = []
        for i in range(len(video) - 1):
            diff = (video[i] - video[i + 1]).abs().mean()
            frame_diffs.append(diff.item())

        avg_diff = sum(frame_diffs) / len(frame_diffs) if frame_diffs else 0
        return max(0.0, 1 - avg_diff)

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

    def is_available(self) -> bool:
        return self.frame_scorer.is_available()

    def offload(self) -> None:
        self.frame_scorer.offload()

    def onload(self) -> None:
        self.frame_scorer.onload()

    def dispose(self) -> None:
        self.frame_scorer.dispose()


@dataclass
class VideoSpec(BaseRewardComponentSpec):
    """Typed config for the Video reward component.

    Wraps an inner frame-level scorer (selected by ``inner_model_name``) and
    blends frame-level alignment with temporal consistency.
    """

    batch_size: int = 8
    device: str = "auto"
    inner_model_name: str = "pickscore"
    temporal_weight: float = 0.3
    alignment_weight: float = 0.7
    sample_frames: int = 8
