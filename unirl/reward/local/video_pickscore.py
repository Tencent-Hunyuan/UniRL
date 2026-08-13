"""VideoPickScore reward scorer — PickScore on the first frame of a video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest
from unirl.utils.media import tensor_frame_to_pil

from .pickscore import PickScoreRewardScorer

if TYPE_CHECKING:
    from PIL import Image


class VideoPickScoreScorer(PickScoreRewardScorer):
    """PickScore applied to the first frame of each video."""

    canonical_model_name = "videopickscore"
    input_kind = "video"

    @staticmethod
    def _extract_first_frame(video: torch.Tensor) -> "Image.Image":
        """Extract the first frame of a channel-first video tensor."""
        if not isinstance(video, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(video).__name__}")
        v = video
        if v.dim() == 5:
            v = v.squeeze(0)
        if v.dim() == 4:
            c = int(v.shape[0])
            if c not in (1, 3, 4):
                raise ValueError(
                    f"Expected channel-first (C, T, H, W) with C in (1, 3, 4); "
                    f"got shape {tuple(v.shape)}. Verify that the upstream "
                    f"engine.decode_latents returns channel-first video tensors."
                )
            frame = v[:, 0, :, :]
        elif v.dim() == 3:
            c = int(v.shape[0])
            if c not in (1, 3, 4):
                raise ValueError(f"Expected channel-first (C, H, W) with C in (1, 3, 4); got shape {tuple(v.shape)}.")
            frame = v
        else:
            raise ValueError(f"Unexpected video tensor shape: {tuple(video.shape)}")

        frame = frame.detach().cpu()
        if not frame.is_floating_point():
            frame = frame.float() / 255.0
        elif frame.numel() > 0 and frame.max() > 1.0:
            frame = (frame / 255.0).clamp(0.0, 1.0)
        else:
            frame = frame.clamp(0.0, 1.0)

        return tensor_frame_to_pil(frame)

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        if request.is_video:
            from torchvision.transforms.functional import to_tensor

            from unirl.types.primitives import Images

            pil_frames = [self._extract_first_frame(v) for v in request.videos]
            frame_pixels = torch.stack([to_tensor(f) for f in pil_frames])
            request = RewardRequest(
                primitives=dict(request.primitives),
                generated={"image": Images.from_dense(frame_pixels)},
                prompt_ids=request.prompt_ids,
                sample_ids=request.sample_ids,
                group_ids=request.group_ids,
                metadata=request.metadata,
                reward_types=request.reward_types,
                return_components=request.return_components,
            )
        return super()._compute_model_rewards(request)


@dataclass
class VideoPickScoreSpec(BaseRewardComponentSpec):
    """Typed config for the VideoPickScore reward component."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
