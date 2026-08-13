"""VideoCLIPDelta — an edit-delta CLIP reward for video-to-video."""

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

    from unirl.types.primitives import Video


class VideoCLIPDeltaScorer(PickScoreRewardScorer):
    """PickScore-to-target minus CLIP-similarity-to-source, on the first frame."""

    canonical_model_name = "videoclipdelta"
    input_kind = "video"

    def __init__(self, *, config: "VideoCLIPDeltaSpec", base_device: str) -> None:
        self.lambda_source = float(getattr(config, "lambda_source", 0.25))
        self.num_score_frames = max(1, int(getattr(config, "num_score_frames", 3)))
        self.source_sim_floor = float(getattr(config, "source_sim_floor", 0.3))
        super().__init__(config=config, base_device=base_device)

    @staticmethod
    def _sample_frames_pil(video: "Video", k: int) -> list["Image.Image"]:
        """K evenly-spaced frames of a per-sample ``Video`` (frames ``[T, C, H, W]``)."""
        frames = video.frames
        if frames is None or frames.ndim != 4:
            raise ValueError(
                "VideoCLIPDeltaScorer: expected per-sample frames [T, C, H, W], got "
                f"{None if frames is None else tuple(frames.shape)}"
            )
        total = int(frames.shape[0])
        idx = torch.linspace(0, total - 1, steps=int(k)).round().long().clamp_(0, total - 1).tolist()
        pils: list["Image.Image"] = []
        for j in idx:
            frame = frames[j].detach().cpu()
            if not frame.is_floating_point():
                frame = frame.float() / 255.0
            elif frame.numel() > 0 and frame.max() > 1.0:
                frame = (frame / 255.0).clamp(0.0, 1.0)
            else:
                frame = frame.clamp(0.0, 1.0)
            pils.append(tensor_frame_to_pil(frame))
        return pils

    def _embed_images(self, pil_images: List["Image.Image"]) -> torch.Tensor:
        inputs = self.processor(images=pil_images, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        emb = self.model.get_image_features(**inputs)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}
        emb = self.model.get_text_features(**inputs)
        return emb / emb.norm(p=2, dim=-1, keepdim=True)

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        edited = request.generated.get("video")
        if edited is None:
            raise ValueError(
                "VideoCLIPDeltaScorer: request.generated['video'] is missing; this scorer needs input_kind='video'."
            )
        source = request.primitives.get("video")
        if source is None:
            raise ValueError(
                "VideoCLIPDeltaScorer: request.primitives['video'] is missing — the V2V condition video must reach "
                "the reward (only V2V recipes provide it). Use a V2V recipe, or switch back to VideoPickScoreScorer."
            )

        prompts = request.prompts
        edited_videos = edited.to_list()
        source_videos = source.to_list()
        n = len(edited_videos)
        if len(source_videos) != n or len(prompts) != n:
            raise ValueError(
                f"VideoCLIPDeltaScorer: misaligned counts edited={n}, source={len(source_videos)}, "
                f"prompts={len(prompts)}."
            )

        k = self.num_score_frames
        edited_frames = [f for v in edited_videos for f in self._sample_frames_pil(v, k)]
        source_frames = [f for v in source_videos for f in self._sample_frames_pil(v, k)]

        rewards: List[float] = []
        with torch.no_grad():
            scale = self.model.logit_scale.exp() / 26.0
            for v_lo in range(0, n, self.batch_size):
                v_hi = min(v_lo + self.batch_size, n)
                nb = v_hi - v_lo
                e = edited_frames[v_lo * k : v_hi * k]
                s = source_frames[v_lo * k : v_hi * k]
                p = prompts[v_lo:v_hi]

                edited_emb = self._embed_images(e)
                source_emb = self._embed_images(s)
                text_emb = self._embed_texts(p).repeat_interleave(k, dim=0)

                text_cos = (text_emb * edited_emb).sum(dim=-1)
                source_cos = (edited_emb * source_emb).sum(dim=-1).clamp(min=self.source_sim_floor)
                text_align = (scale * text_cos).view(nb, k).mean(dim=1)
                source_sim = (scale * source_cos).view(nb, k).mean(dim=1)

                reward = text_align - self.lambda_source * source_sim
                rewards.extend(reward.float().cpu().tolist())
        return rewards


@dataclass
class VideoCLIPDeltaSpec(BaseRewardComponentSpec):
    """Typed config for the VideoCLIPDelta reward component."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    lambda_source: float = 0.25
    source_sim_floor: float = 0.3
    num_score_frames: int = 3
