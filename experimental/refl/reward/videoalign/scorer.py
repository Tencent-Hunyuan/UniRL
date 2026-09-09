"""VideoAlign differentiable reward — consumes ``[B, C, T, H, W]`` float video in ``[-1, 1]``."""

from __future__ import annotations

import logging
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
        super().__init__(
            model_name=self.canonical_model_name,
            device=resolve_device(config.device, base_device),
            batch_size=int(config.batch_size),
            reward_model_path=config.reward_model_path,
            resize_height=config.resize_height,
            resize_width=config.resize_width,
            micro_batch_size=config.micro_batch_size,
            reward_num_frames=config.reward_num_frames,
            use_norm=config.use_norm,
            w_vq=config.w_vq,
            w_mq=config.w_mq,
            w_ta=config.w_ta,
            differentiable=config.differentiable,
        )

    def _load_model(self) -> None:
        reward_model_path = str(self.model_kwargs["reward_model_path"])
        if not reward_model_path:
            raise ValueError(
                "VideoAlignRewardScorer: ``reward_model_path`` must be set "
                "(the directory containing ``model_config.json`` + the "
                "``checkpoint-*`` subdir)."
            )

        self._w_vq = float(self.model_kwargs["w_vq"])
        self._w_mq = float(self.model_kwargs["w_mq"])
        self._w_ta = float(self.model_kwargs["w_ta"])
        self._use_norm = bool(self.model_kwargs["use_norm"])
        self._reward_num_frames = int(self.model_kwargs["reward_num_frames"])
        self._differentiable = bool(self.model_kwargs["differentiable"])

        logger.info(
            "VideoAlignRewardScorer: loading VideoRewardWrapper from %s",
            reward_model_path,
        )
        self.model = VideoRewardWrapper(
            checkpoint_dir=reward_model_path,
            device=self.device,
            dtype=torch.bfloat16,
            use_norm=self._use_norm,
            resize_height=int(self.model_kwargs["resize_height"]),
            resize_width=int(self.model_kwargs["resize_width"]),
            micro_batch_size=int(self.model_kwargs["micro_batch_size"]),
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

        if self._reward_num_frames > 0:
            ds: List[torch.Tensor] = []
            for v in per_sample_videos:
                if v.shape[0] > self._reward_num_frames:
                    idx = torch.linspace(
                        0,
                        v.shape[0] - 1,
                        self._reward_num_frames,
                        device=v.device,
                    ).long()
                    v = v[idx]
                ds.append(v)
            per_sample_videos = ds

        autograd_ctx = torch.enable_grad if self._differentiable else torch.no_grad
        with autograd_ctx():
            scores = self.model.forward_scores(
                per_sample_videos,
                prompts,
                use_norm=self._use_norm,
            )

        reward = self._w_vq * scores["VQ"] + self._w_mq * scores["MQ"] + self._w_ta * scores["TA"]
        return reward.float()

    def offload(self) -> None:
        if self.model is not None and getattr(self.model, "model", None) is not None:
            self.model.model.cpu()
            torch.cuda.empty_cache()

    def onload(self) -> None:
        if self.model is not None and getattr(self.model, "model", None) is not None:
            self.model.model.to(self.device)

    def is_available(self) -> bool:
        return bool(self._is_loaded)

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

    differentiable: bool = True


__all__ = [
    "VideoAlignRewardScorer",
    "VideoAlignSpec",
]
