"""FLUX-family image adapters: plain FLUX + FLUX.2-Klein.

``FluxAdapter`` is the default image path (5-D passthrough). ``Flux2KleinAdapter``
overrides two stages: ``to_image_form`` (Klein's transformer is a pure sequence
model emitting packed ``[B, T, H*W, C_packed]`` tokens, so the trajectory is
unpacked to image form before segment assembly) and the schedule policy (Klein
needs a model-specific ``compute_mu`` the generic FlowMatch path can't synthesize).
The Dance-GRPO SDE label rides on the base ``resolve_sde_label``.
"""

from __future__ import annotations

from unirl.config.require import require
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.types.rollout_req import RolloutReq


@register_adapter("flux")
class FluxAdapter(ImageAdapter):
    """FLUX — image-form 5-D trajectory throughout; default path."""

    pass


# FLUX.2 patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
_KLEIN_DOWNSAMPLE = 16


@register_adapter("flux2_klein")
class Flux2KleinAdapter(ImageAdapter):
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

    def to_image_form(self, traj, req: RolloutReq):
        """Unpack Klein's packed ``[B, T, H*W, C]`` to image-form ``[B, T, C, H_pat, W_pat]``.

        Klein keeps the packed channels at patch resolution (a token→spatial
        reshape). 5-D input passes through untouched (image-form arrivals).
        """
        if traj.ndim == 5:
            return traj
        B, T, S, C, h_pat, w_pat = utils.validate_packed_trajectory(
            traj, req, family="flux2_klein", downsample=_KLEIN_DOWNSAMPLE, require_divisible=True
        )
        from unirl.models.flux2_klein.flux2_klein_utils import unpack_latents

        flat = traj.reshape(B * T, S, C)
        return unpack_latents(flat, h_pat, w_pat).reshape(B, T, C, h_pat, w_pat).contiguous()


__all__ = ["FluxAdapter", "Flux2KleinAdapter"]
