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

from typing import List, Optional

from unirl.rollout.engine.sglang_diffusion import utils
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

    # Legacy image-path video family: drop 4-D decoded samples (incl. single-frame)
    # rather than squeezing them into images.
    squeeze_single_frame_4d = False


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(ImageAdapter):
    """HunyuanVideo — image-path parity (see module note); migrate to VideoAdapter when it has a video reward baseline."""

    # Legacy image-path video family: drop 4-D decoded samples (incl. single-frame)
    # rather than squeezing them into images.
    squeeze_single_frame_4d = False


@register_adapter("wan21")
class Wan21T2VAdapter(VideoAdapter):
    """WAN 2.1 T2V — proper video output consumed by ``video_pickscore``.

    The text/conditions path is the generic UMT5 fuse from ``ImageAdapter``
    (single text encoder; no CFG negative branch when ``guidance_scale <= 1``);
    only the video-output overrides on ``VideoAdapter`` apply. The sglang server
    resolves the WAN pipeline from ``model_path`` (the ``Wan-AI/Wan2.1-T2V-1.3B``
    -Diffusers checkpoint), so no extra ``boot_kwargs`` are needed.
    """

    pass


__all__ = ["VideoAdapter", "MochiAdapter", "HunyuanVideoAdapter", "Wan21T2VAdapter"]
