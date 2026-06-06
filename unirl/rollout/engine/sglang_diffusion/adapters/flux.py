"""FLUX-family image DiT adapters: plain FLUX + FLUX.2-Klein.

``FluxAdapter`` is the default image path (5-D passthrough). ``Flux2KleinAdapter``
overrides exactly two stages: ``build_segment`` (Klein's transformer is a pure
sequence model emitting packed ``[B, T, H*W, C_packed]`` tokens, so the trajectory
is unpacked to image form before segment assembly) and the schedule policy (Klein
needs a model-specific ``compute_mu`` the generic FlowMatch path can't synthesize).
The Dance-GRPO SDE label rides on the base ``resolve_sde_label``.
"""

from __future__ import annotations

from typing import List, Optional

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image_dit import ImageDiTAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import get_diffusion_params


@register_adapter("flux")
class FluxAdapter(ImageDiTAdapter):
    """FLUX image DiT — image-form 5-D trajectory throughout; default path."""

    pass


# FLUX.2 patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
_KLEIN_DOWNSAMPLE = 16


@register_adapter("flux2_klein")
class Flux2KleinAdapter(ImageDiTAdapter):
    """FLUX.2-Klein — packed sequence-style trajectory + model-specific schedule."""

    def validate(self) -> None:
        super().validate()
        require(
            callable(getattr(self.model_config, "build_schedule_policy", None)),
            "flux2_klein adapter requires model_config.build_schedule_policy() "
            "(Klein needs a model-specific compute_mu the generic FlowMatch path "
            "cannot synthesize from scheduler_config.json).",
        )

    def schedule_policy(self):
        return self.model_config.build_schedule_policy()

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Stage override: unpack Klein's packed trajectory before segment assembly."""
        traj = utils.collect_trajectory_latents(results)
        traj = self._unpack_packed(traj, req)
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=req.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def _unpack_packed(self, traj, req: RolloutReq):
        """Convert SGLang's packed ``[B, T, H*W, C]`` to image-form ``[B, T, C, H_pat, W_pat]``.

        Private helper, not an override seam — stages are the only derivation
        points. 5-D input passes through untouched (image-form arrivals).
        """
        if traj.ndim == 5:
            return traj
        if traj.ndim != 4:
            raise ValueError(
                f"flux2_klein: SGLang trajectory has rank {traj.ndim}, want 4 (packed) "
                f"or 5 (image-form); shape={tuple(traj.shape)}."
            )
        diffusion = get_diffusion_params(req.sampling_params)
        height = int(diffusion.height) if diffusion.height is not None else None
        width = int(diffusion.width) if diffusion.width is not None else None
        if height is None or width is None:
            raise ValueError(
                "flux2_klein: need height/width from req.sampling_params to unpack "
                "the packed [B, T, H*W, C] trajectory; both must be set."
            )
        if height % _KLEIN_DOWNSAMPLE or width % _KLEIN_DOWNSAMPLE:
            raise ValueError(
                f"flux2_klein: height ({height}) and width ({width}) must be "
                f"divisible by the VAE×patchify downsample ({_KLEIN_DOWNSAMPLE})."
            )
        h_pat = height // _KLEIN_DOWNSAMPLE
        w_pat = width // _KLEIN_DOWNSAMPLE
        B, T, S, C_packed = traj.shape
        if S != h_pat * w_pat:
            raise ValueError(
                f"flux2_klein: packed token count S={S} != h_pat*w_pat={h_pat * w_pat} "
                f"(from height={height}, width={width}). Schedule/recipe drift — fix the "
                f"source rather than silently reshape to a wrong spatial layout."
            )
        from unirl.models.flux2_klein.flux2_klein_utils import unpack_latents

        flat = traj.reshape(B * T, S, C_packed)
        return unpack_latents(flat, h_pat, w_pat).reshape(B, T, C_packed, h_pat, w_pat).contiguous()


__all__ = ["FluxAdapter", "Flux2KleinAdapter"]
