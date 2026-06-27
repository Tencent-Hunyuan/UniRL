"""Video-family adapters.

Two output shapes live here:

* ``VideoAdapter`` — proper video output. The latent trajectory is video-form
  6-D ``[B, T+1, C, F, H, W]`` (an extra latent-frame axis vs the image path's
  5-D ``[B, T+1, C, H, W]``) and the decoded media is packed into a ragged
  :class:`~unirl.types.primitives.Videos` (``[total_T, C, H, W]``) instead of
  being dropped. WAN 2.1 T2V rides this base — its rollout output is consumed by
  the ``video_pickscore`` reward, the first such video reward consumer.

* ``MochiAdapter`` / ``HunyuanVideoAdapter`` — kept on the legacy image path
  (see note below) for behavioral parity with the old ``sglang`` engine. Migrate
  them onto ``VideoAdapter`` once each has a verified video reward baseline.

PARITY NOTE (image-path video families): the legacy ``sglang`` engine treated
every family — including the video ones — through the image path: it built an
image-form ``LatentSegment`` (``make_image_segment``) and *dropped* 4-D decoded
video with a warning (there was no video reward consumer yet). ``MochiAdapter`` /
``HunyuanVideoAdapter`` reproduce that exactly so the per-family parity gate
holds; only families with a real video consumer (WAN) move to ``VideoAdapter``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.utils.tracks import _cat_padded_rows
from unirl.types.conditions.text import TextEmbedCondition
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.segments.latent import make_video_segment


class VideoAdapter(ImageAdapter):
    """Base for true video-output families (6-D latent trajectory → ``Videos``).

    Reuses ``ImageAdapter``'s request side verbatim — ``build_sampling`` already
    forwards ``num_frames`` and the SDE/rollout pins are modality-agnostic — and
    overrides only the response-shape variation points: the segment is stamped
    ``Modality.VIDEO`` and carries the 6-D ``[B, T+1, C, F, H, W]`` trajectory,
    and the decoded media is packed as ``Videos`` rather than dropped.
    """

    #: RolloutResp track key (video, not image).
    track_name: str = "video"
    #: Modality stamp for the latent segment.
    segment_factory = staticmethod(make_video_segment)

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Video-form trajectory: collect, gate the 6-D shape, assemble.

        Video latents keep the extra frame axis throughout, so the trajectory is
        rank 6 ``[B, T+1, C, F, H, W]`` (vs the image path's rank 5). The downstream
        ``build_latent_segment`` is shape-agnostic past the T+1 invariant, so the
        only difference from the image path is the rank gate + the video segment
        factory.
        """
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 6:
            raise ValueError(
                f"{self.model_family}: expected a 6-D video-form trajectory "
                f"[B, T+1, C, F, H, W]; got rank {traj.ndim}, shape {tuple(traj.shape)}."
            )
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=req.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def build_decoded(self, req: RolloutReq, results: List[RawResult]):
        return utils.stack_decoded_videos(results)


@register_adapter("mochi")
class MochiAdapter(ImageAdapter):
    """Mochi — image-path parity (see module note); migrate to VideoAdapter when it has a video reward baseline."""

    pass




@register_adapter("ltx2")
class Ltx2T2VAdapter(VideoAdapter):
    """LTX-2 / LTX-2.3 T2V — ~2.4B video DiT, Gemma3 text encoding, proper video
    output (6-D trajectory → ``Videos``) consumed by ``video_pickscore``.

    First-cut on the generic single-text fuse (like ``Wan21T2VAdapter``): LTX2's
    primary condition is a single ``text`` stream (``LTX2Conditions.text``), unlike
    HunyuanVideo's dual encoder. CAVEAT — needs smoke validation: LTX2's trainside
    text path is Gemma3 → text CONNECTORS → ``video_embeds`` (the DiT consumes
    connector outputs, not raw Gemma). If sglang's LTX2 server returns the connector
    ``video_embeds`` as ``prompt_embeds`` the generic fuse suffices; if it returns
    raw Gemma hidden states, this adapter must apply the connectors here (override
    ``build_condition``) and route the result onto the ``text`` key. Confirm the
    sglang output shape/format with a 1-rollout EMBED dump before alignment.
    """

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """LTX-2 latents are PACKED token sequences, not a spatial video grid.

        WAN/HunyuanVideo carry a 6-D ``[B, T+1, C, F, H, W]`` trajectory, but LTX-2's
        DiT operates on a patchified token sequence, so the rollout trajectory is
        rank-4 ``[B, T+1, seq, dim]`` (e.g. ``[B, 11, 192, 128]``). ``VideoAdapter``'s
        strict 6-D gate rejects it; ``build_latent_segment`` itself only needs the
        ``T+1`` axis at dim 1 and is otherwise shape-agnostic, so accept the packed
        trajectory directly (the trainside replays the identical packed latents, so
        rollout↔replay stays aligned).
        """
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim < 3:
            raise ValueError(
                f"ltx2: expected a packed trajectory [B, T+1, ...]; got rank {traj.ndim}, "
                f"shape {tuple(traj.shape)}."
            )
        # LTX-2 co-denoises an AUDIO latent the video DiT cross-attends to; collect
        # the parallel audio trajectory and stamp it as ``segment.aux_latents`` so the
        # trainside ``LTX2DiffusionStage.replay`` replays the same per-step audio
        # (else it raises "aux_latents (audio trajectory) missing").
        aux_traj = utils.collect_aux_trajectory_latents(results)
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=req.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
            aux_trajectory=aux_traj,
        )

__all__ = ["Ltx2T2VAdapter", "VideoAdapter", "MochiAdapter", "Ltx2T2VAdapter"]
