"""VideoAlign differentiable reward — consumes ``[B, C, T, H, W]`` float video in ``[-1, 1]``."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.base import LocalRewardBackend
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .wrapper import VideoRewardWrapper

logger = logging.getLogger(__name__)


class VideoAlignRewardScorer(LocalRewardBackend):
    """Qwen2-VL VideoAlign reward — VQ + MQ + TA → scalar per sample."""

    canonical_model_name = "videoalign"
    input_kind = "video"

    def __init__(self, *, config: "VideoAlignSpec", base_device: str) -> None:
        self._spec = config
        super().__init__(
            model_name=self.canonical_model_name,
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
        )

    def covers_prompt_video(self) -> bool:
        return self._spec.w_ta > 0.0

    def _load_model(self) -> None:
        logger.info(
            "VideoAlignRewardScorer: loading VideoRewardWrapper from %s",
            self._spec.reward_model_path,
        )
        self.model = VideoRewardWrapper(
            checkpoint_dir=self._spec.reward_model_path,
            device=self.device,
            dtype=torch.bfloat16,
            use_norm=self._spec.use_norm,
            resize_height=self._spec.resize_height,
            resize_width=self._spec.resize_width,
            micro_batch_size=self._spec.micro_batch_size,
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        raise NotImplementedError("VideoAlignRewardScorer is REFL-only; use compute_rewards_differentiable().")

    def compute_rewards_differentiable(
        self,
        media_tensor: torch.Tensor,
        prompts: List[str],
        records: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        del records
        if media_tensor.ndim != 5:
            raise ValueError(f"VideoAlignRewardScorer expects [B,C,T,H,W], got {tuple(media_tensor.shape)}")
        if len(prompts) != int(media_tensor.shape[0]):
            raise ValueError(
                f"VideoAlignRewardScorer: prompts length {len(prompts)} != batch size {int(media_tensor.shape[0])}."
            )

        per_sample_videos: List[torch.Tensor] = []
        for v in media_tensor:
            v = v.to(self.device).permute(1, 0, 2, 3).clamp(-1.0, 1.0).contiguous()
            per_sample_videos.append(v)

        if self._spec.reward_num_frames > 0:
            ds: List[torch.Tensor] = []
            for v in per_sample_videos:
                if v.shape[0] > self._spec.reward_num_frames:
                    idx = torch.linspace(
                        0,
                        v.shape[0] - 1,
                        self._spec.reward_num_frames,
                        device=v.device,
                    ).long()
                    v = v[idx]
                ds.append(v)
            per_sample_videos = ds

        with torch.enable_grad():
            scores = self.model.forward_scores(
                per_sample_videos,
                prompts,
                use_norm=self._spec.use_norm,
            )

        reward = self._spec.w_vq * scores["VQ"] + self._spec.w_mq * scores["MQ"] + self._spec.w_ta * scores["TA"]
        return reward.float()

    def offload(self) -> None:
        if self.model is None:
            return
        self.model.model.cpu()
        torch.cuda.empty_cache()

    def onload(self) -> None:
        if self.model is None:
            return
        self.model.model.to(self.device)

    def dispose(self) -> None:
        self.offload()


@dataclass
class VideoAlignSpec(BaseRewardComponentSpec):
    """Typed config for :class:`VideoAlignRewardScorer`."""

    reward_model_path: str = ""

    device: str = "auto"
    batch_size: int = 1

    resize_height: int = 336
    resize_width: int = 588
    micro_batch_size: int = 1
    reward_num_frames: int = 36

    use_norm: bool = True
    w_vq: float = 1.0
    w_mq: float = 1.0
    w_ta: float = 1.0

    def __post_init__(self) -> None:
        if not self.reward_model_path:
            raise ValueError(
                "VideoAlignRewardScorer: ``reward_model_path`` must be set "
                "(the directory containing ``model_config.json`` + the "
                "``checkpoint-*`` subdir)."
            )
        if self.resize_height <= 0 or self.resize_width <= 0:
            raise ValueError(
                f"VideoAlignSpec: resize_height and resize_width must be > 0; "
                f"got ({self.resize_height}, {self.resize_width})."
            )
        if self.micro_batch_size <= 0:
            raise ValueError(f"VideoAlignSpec: micro_batch_size must be > 0; got {self.micro_batch_size}.")
        if not (math.isfinite(self.w_vq) and math.isfinite(self.w_mq) and math.isfinite(self.w_ta)):
            raise ValueError("VideoAlignSpec: w_vq, w_mq, and w_ta must be finite numbers.")


__all__ = [
    "VideoAlignRewardScorer",
    "VideoAlignSpec",
]
