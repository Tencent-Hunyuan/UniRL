"""Video-family adapters: Mochi + HunyuanVideo."""

from __future__ import annotations

from typing import List, Optional

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.segments.latent import make_video_segment


class VideoAdapter(ImageAdapter):
    """Conversion for video-output families.

    SGLang exposes decoded samples as untyped tensors, so modality is restored
    here rather than inferred from shape: ``[C, T=1, H, W]`` remains a video.
    """

    track_name: str = "video"
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
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 6:
            raise ValueError(
                f"{self.model_family}: expected a 6-D video-form trajectory [B, T+1, C, F, H, W]; "
                f"got rank {traj.ndim}, shape {tuple(traj.shape)}."
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
        diffusion = req.sampling_params.get("diffusion")
        expected_num_frames = int(diffusion.num_frames) if diffusion is not None else None
        return utils.stack_decoded_videos(results, expected_num_frames=expected_num_frames)


@register_adapter("mochi")
class MochiAdapter(ImageAdapter):
    """Mochi placeholder.

    There is no in-repo Mochi bundle/recipe yet, so keep the legacy image-path
    parity behavior until a verified video implementation lands.
    """

    pass


@register_adapter("hunyuan_video")
class HunyuanVideoAdapter(VideoAdapter):
    """HunyuanVideo video adapter."""

    pass


__all__ = ["VideoAdapter", "MochiAdapter", "HunyuanVideoAdapter"]
