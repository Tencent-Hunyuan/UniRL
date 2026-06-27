"""VideoCLIPDelta — an edit-delta CLIP reward for video-to-video.

Plain PickScore on a content-anchored V2V output starts near its ceiling
(~0.8): the edit caption mostly describes content the *source* video already
shows, so an un-edited output already scores high and there is little headroom
to train on. This scorer subtracts how much the edited frame still looks like
the SOURCE condition frame, so the reward measures the *edit*, not the content
that was free to begin with:

    reward = pickscore(edited_first_frame, target_caption)
             - lambda_source * cos(edited_first_frame, source_first_frame)

- The first term (identical scaling to ``PickScoreRewardScorer``) keeps the
  edit on-target for the caption.
- The second term (CLIP image-image cosine to the source) is high when the
  output barely changed, so an un-edited V2V output nets ~0 and the reward only
  climbs as the model actually applies the requested edit.

The source condition frame is available because ``RewardService`` copies the
rollout request's ``primitives`` (including the ``video`` condition) into the
reward request and ``repeat_interleave``s them to per-sample alignment.
"""

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
    """PickScore-to-target minus CLIP-similarity-to-source, on the first frame.

    Inherits CLIP/PickScore model loading from ``PickScoreRewardScorer``; only
    the reward computation differs (it also reads the source condition video
    from ``request.primitives['video']``).
    """

    canonical_model_name = "videoclipdelta"
    input_kind = "video"

    def __init__(self, *, config: "VideoCLIPDeltaSpec", base_device: str) -> None:
        self.lambda_source = float(getattr(config, "lambda_source", 0.25))
        # Score K evenly-spaced frames (not just the first) so the edit reward
        # reflects the whole clip rather than a single-frame proxy.
        self.num_score_frames = max(1, int(getattr(config, "num_score_frames", 3)))
        # Cap on the source-divergence reward: clamp the edited-vs-source CLIP
        # cosine to this floor so once a frame is "different enough" there is no
        # further reward for diverging more. This bounds the penalty's
        # contribution and zeroes its gradient past the floor, which stops the
        # reward-hacking failure mode where the policy destroys content just to
        # look maximally unlike the source.
        self.source_sim_floor = float(getattr(config, "source_sim_floor", 0.3))
        super().__init__(config=config, base_device=base_device)

    # ------------------------------------------------------------------
    # Frame + embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_frames_pil(video: "Video", k: int) -> list["Image.Image"]:
        """K evenly-spaced frames of a per-sample ``Video`` (frames ``[T, C, H, W]``).

        Always returns exactly ``k`` frames (indices repeat when ``T < k``) so the
        batch flattens to a fixed ``n * k`` layout for the per-video mean.
        """
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

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

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
        # Flatten to n*k frames so each video contributes exactly k frames.
        edited_frames = [f for v in edited_videos for f in self._sample_frames_pil(v, k)]
        source_frames = [f for v in source_videos for f in self._sample_frames_pil(v, k)]

        rewards: List[float] = []
        with torch.no_grad():
            # Both terms use the SAME PickScore scaling (logit_scale / 26), so
            # lambda_source is a pure, scale-invariant relative weight between
            # "match the target text" and "stop looking like the source".
            scale = self.model.logit_scale.exp() / 26.0
            for v_lo in range(0, n, self.batch_size):
                v_hi = min(v_lo + self.batch_size, n)
                nb = v_hi - v_lo
                e = edited_frames[v_lo * k : v_hi * k]
                s = source_frames[v_lo * k : v_hi * k]
                p = prompts[v_lo:v_hi]

                edited_emb = self._embed_images(e)  # [nb*k, d]
                source_emb = self._embed_images(s)  # [nb*k, d]
                text_emb = self._embed_texts(p).repeat_interleave(k, dim=0)  # [nb*k, d]

                # Per-frame cosines, averaged over the k frames of each video.
                text_cos = (text_emb * edited_emb).sum(dim=-1)
                # Cap the source-divergence reward: clamp the cosine from below so
                # diverging past the floor earns nothing more (and gets no gradient).
                source_cos = (edited_emb * source_emb).sum(dim=-1).clamp(min=self.source_sim_floor)
                text_align = (scale * text_cos).view(nb, k).mean(dim=1)
                source_sim = (scale * source_cos).view(nb, k).mean(dim=1)

                reward = text_align - self.lambda_source * source_sim
                rewards.extend(reward.float().cpu().tolist())
        return rewards


@dataclass
class VideoCLIPDeltaSpec(BaseRewardComponentSpec):
    """Typed config for the VideoCLIPDelta reward component.

    Mirrors ``VideoPickScoreSpec`` plus the edit-delta knobs:

    - ``lambda_source`` — weight on the "still looks like the source" penalty
      (both terms share the PickScore scale, so this is scale-invariant). Higher
      pushes the policy to change more from the condition video; lower keeps it
      closer to the source.
    - ``source_sim_floor`` — floor on the edited-vs-source CLIP cosine; diverging
      past it earns no extra reward (caps the penalty / kills its runaway gradient).
    - ``num_score_frames`` — number of evenly-spaced frames scored per clip.
    """

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    lambda_source: float = 0.25
    source_sim_floor: float = 0.3
    num_score_frames: int = 3
