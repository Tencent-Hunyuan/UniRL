"""VideoPickScore reward scorer — PickScore on representative frames of a video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch

from unirl.reward.base import PromptRewardComponentSpec
from unirl.types.reward import RewardRequest
from unirl.utils.media import tensor_frame_to_pil

from .pickscore import PickScoreRewardScorer

if TYPE_CHECKING:
    from PIL import Image


class VideoPickScoreScorer(PickScoreRewardScorer):
    """PickScore applied to selected frames of each video (``frame_selection``)."""

    canonical_model_name = "videopickscore"
    input_kind = "video"

    def __init__(self, *, config: "VideoPickScoreSpec", base_device: str) -> None:
        if config.frame_selection not in ("first", "middle", "uniform"):
            raise ValueError(
                "VideoPickScoreSpec.frame_selection must be 'first', 'middle', or 'uniform'; "
                f"got {config.frame_selection!r}"
            )
        if config.frame_aggregation not in ("mean", "topk_mean_blend"):
            raise ValueError(
                "VideoPickScoreSpec.frame_aggregation must be 'mean' or 'topk_mean_blend'; "
                f"got {config.frame_aggregation!r}"
            )
        if config.num_score_frames < 1 or config.topk_frames < 1:
            raise ValueError(
                "VideoPickScoreSpec.num_score_frames and topk_frames must be >= 1; "
                f"got {config.num_score_frames} and {config.topk_frames}"
            )
        if not 0.0 <= config.all_frame_mean_weight <= 1.0:
            raise ValueError(
                f"VideoPickScoreSpec.all_frame_mean_weight must lie in [0, 1], got {config.all_frame_mean_weight}"
            )
        # PickScoreRewardScorer.__init__ consumes only device/batch_size/
        # processor_id/model_id, so the frame choice is captured here.
        super().__init__(config=config, base_device=base_device)
        self.frame_selection = config.frame_selection
        self.num_score_frames = config.num_score_frames
        self.frame_aggregation = config.frame_aggregation
        self.topk_frames = config.topk_frames
        self.all_frame_mean_weight = config.all_frame_mean_weight

    def covers_prompt_video(self) -> bool:
        return True

    @staticmethod
    def _extract_frames(video: torch.Tensor, which: str, count: int) -> List["Image.Image"]:
        """Extract representative frames of a channel-first video tensor."""
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
            t = int(v.shape[1])
            if which == "uniform":
                indices = torch.linspace(0, t - 1, steps=min(count, t)).round().long().tolist()
            else:
                indices = [(t // 2) if which == "middle" else 0]
            frames = [v[:, index, :, :] for index in indices]
        elif v.dim() == 3:
            c = int(v.shape[0])
            if c not in (1, 3, 4):
                raise ValueError(f"Expected channel-first (C, H, W) with C in (1, 3, 4); got shape {tuple(v.shape)}.")
            frames = [v]
        else:
            raise ValueError(f"Unexpected video tensor shape: {tuple(video.shape)}")

        images = []
        for frame in frames:
            frame = frame.detach().cpu()
            if not frame.is_floating_point():
                frame = frame.float() / 255.0
            elif frame.numel() > 0 and frame.max() > 1.0:
                frame = (frame / 255.0).clamp(0.0, 1.0)
            else:
                frame = frame.clamp(0.0, 1.0)
            images.append(tensor_frame_to_pil(frame))
        return images

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        if not request.is_video:
            return super()._compute_model_rewards(request)

        from torchvision.transforms.functional import to_tensor

        from unirl.types.primitives import Images

        per_video = [self._extract_frames(v, self.frame_selection, self.num_score_frames) for v in request.videos]
        score_count = len(per_video[0])
        if any(len(frames) != score_count for frames in per_video):
            raise RuntimeError("VideoPickScoreScorer sampled inconsistent frame counts across the batch")
        per_frame_scores = []
        for frame_index in range(score_count):
            frame_pixels = torch.stack([to_tensor(frames[frame_index]) for frames in per_video])
            frame_request = RewardRequest(
                generated={"image": Images.from_dense(frame_pixels)},
                conditioning=dict(request.conditioning),
                original_prompt=request.original_prompt,
                generation_prompt=request.generation_prompt,
                sample_ids=request.sample_ids,
                group_ids=request.group_ids,
                metadata=request.metadata,
            )
            per_frame_scores.append(torch.tensor(super()._compute_model_rewards(frame_request), dtype=torch.float32))
        scores = torch.stack(per_frame_scores, dim=0)
        all_frame_mean = scores.mean(dim=0)
        if self.frame_aggregation == "mean":
            return all_frame_mean.tolist()
        topk_mean = scores.topk(min(self.topk_frames, score_count), dim=0).values.mean(dim=0)
        weight = self.all_frame_mean_weight
        return ((1.0 - weight) * topk_mean + weight * all_frame_mean).tolist()


@dataclass
class VideoPickScoreSpec(PromptRewardComponentSpec):
    """Typed config for the VideoPickScore reward component."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    # Which frames represent the clip. "first" (default) preserves the
    # historical behaviour for every existing recipe. "middle" takes T // 2 --
    # use it when clips can open on a fade or a reveal, where frame 0 is nearly
    # blank and that blankness lands straight in the reward. "uniform" scores
    # `num_score_frames` evenly spaced frames and aggregates them.
    frame_selection: str = "first"
    num_score_frames: int = 1
    # "mean" averages every scored frame; "topk_mean_blend" mixes the mean of
    # the best `topk_frames` with the all-frame mean by `all_frame_mean_weight`.
    frame_aggregation: str = "mean"
    topk_frames: int = 3
    all_frame_mean_weight: float = 0.25
