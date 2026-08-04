"""VideoPickScore reward scorer — PickScore on ONE representative frame of a video."""

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
    """PickScore applied to ONE frame of each video (``frame_selection``).

    Inherits model loading and CLIP scoring from ``PickScoreRewardScorer``;
    the only addition is a pre-processing step that extracts a single frame
    from each video tensor before scoring.

    Because the score rests on one frame, WHICH frame is a real modelling
    choice, not a detail: a clip that opens on a fade or a reveal has a nearly
    blank frame 0, and that blankness becomes the reward. Default stays
    ``"first"`` so no existing recipe changes.

    ``input_kind = "video"`` is required so that the reward pipeline routes
    decoded video tensors into ``RewardRequest.videos`` (and sets
    ``request.is_video = True``) — without it, the request would arrive with
    only ``images`` populated and ``_extract_frame`` below would never
    run, silently degrading to scoring the middle-frame PIL preview.
    """

    canonical_model_name = "videopickscore"
    input_kind = "video"

    def __init__(self, *, config: "VideoPickScoreSpec", base_device: str) -> None:
        # PickScoreRewardScorer.__init__ consumes only device/batch_size/
        # processor_id/model_id, so the frame choice is captured here.
        super().__init__(config=config, base_device=base_device)
        self.frame_selection = str(getattr(config, "frame_selection", "first"))
        if self.frame_selection not in ("first", "middle"):
            raise ValueError(
                f"VideoPickScoreSpec.frame_selection must be 'first' or 'middle'; got {self.frame_selection!r}"
            )

    @staticmethod
    def _extract_frame(video: torch.Tensor, which: str = "first") -> "Image.Image":
        """Extract one frame of a channel-first video tensor.

        ``which='first'`` is the historical behaviour and the default.
        ``which='middle'`` takes ``T // 2`` instead: a clip that opens on a
        reveal (or a fade) makes frame 0 nearly blank, and since this scorer
        sees ONE frame that blankness lands directly in the reward. Measured on
        a MiniMax-H3 rollout whose frame 0 had std 0.015, scoring frame 0 cost
        -0.0815 against mid-clip -- several times the entire policy signal.

        Contract: input is the per-sample slice produced by
        ``extract_videos_from_output``, which iterates the leading batch
        dim of ``RolloutSamples.decoded_videos``. ``decoded_videos`` is
        always written by ``engine.decode_latents`` (channel-first
        ``(B, C, T, H, W)``), so per-item layout is always
        ``(C, T, H, W)``. Already-3D inputs are treated as a single
        channel-first frame.

        We deliberately do NOT try to disambiguate channel-first vs
        frame-first by inspecting leading dims: small ``T`` (e.g. WAN T2V
        with ``num_frames=3``) collapses the leading dims into the same
        ``{1, 3, 4}`` set and would silently score the wrong axis under
        the old heuristic.
        """
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
            frame = v[:, (t // 2) if which == "middle" else 0, :, :]
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

            pil_frames = [self._extract_frame(v, self.frame_selection) for v in request.videos]
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
    """Typed config for the VideoPickScore reward component.

    Mirrors ``PickScoreSpec`` plus ``frame_selection``.
    ``PickScoreRewardScorer.__init__`` consumes exactly ``device``,
    ``batch_size``, ``processor_id`` and ``model_id``, so
    ``VideoPickScoreScorer`` overrides ``__init__`` to pick up the extra
    field. A dedicated Spec (instead of reusing ``PickScoreSpec``)
    keeps Hydra's structured-config registry one-Spec-per-name and lets
    YAML reference this scorer as ``name: videopickscore``.
    """

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
    # Which single frame represents the clip. "first" (default) preserves the
    # historical behaviour for every existing recipe. "middle" takes T // 2 --
    # use it when clips can open on a fade or a reveal, where frame 0 is nearly
    # blank and that blankness lands straight in the reward.
    frame_selection: str = "first"
