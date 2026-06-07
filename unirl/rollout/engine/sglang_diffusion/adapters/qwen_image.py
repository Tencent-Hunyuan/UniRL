"""Qwen-Image image DiT adapter — packed sequence trajectory, generic schedule.

Qwen-Image's transformer is a packed-token model like FLUX.2-Klein: SGLang's
denoising loop carries ``[B, S, C*4]`` tokens (2×2 patchify over a 16-channel
VAE latent), so the trajectory arrives packed ``[B, T+1, S, 64]`` and
``build_segment`` unpacks it before assembly. Unlike Klein (which keeps
packed channels at patch resolution), the unpack target is the **true**
channel form ``[B, T+1, 16, latent_h, latent_w]`` — exactly what the
trainside replay consumes (``models/qwen_image/diffusion.py`` stores segments
unpacked and packs only at the transformer boundary), so segments are
interchangeable between the trainside and sglang engines.

Everything else is the default path: the generic schedule policy reads
``use_dynamic_shifting`` / ``dynamic_shift_overrides`` / ``shift_terminal``
off the model config (no Klein-style factory needed — Qwen's μ is the linear
``calculate_dynamic_mu`` form), the ``transformer.`` LoRA prefix and
``text`` / ``negative_text`` condition fusion come from ``ImageDiTAdapter``,
and CFG stays on ``guidance_scale`` (the server's qwen pipeline applies the
same norm-preserving true-CFG blend as the trainside replay).
"""

from __future__ import annotations

from typing import List, Optional

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image_dit import ImageDiTAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import get_diffusion_params

# Qwen-Image patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
_QWEN_DOWNSAMPLE = 16


@register_adapter("qwen_image")
class QwenImageAdapter(ImageDiTAdapter):
    """Qwen-Image — packed sequence-style trajectory unpacked to true channels."""

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Stage override: unpack Qwen's packed trajectory before segment assembly."""
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
        """Convert SGLang's packed ``[B, T, S, C*4]`` to true-channel ``[B, T, C, H_lat, W_lat]``.

        Private helper, not an override seam — stages are the only derivation
        points. 5-D input passes through untouched (image-form arrivals).
        Grid arithmetic is the canonical ``latent_h = 2 * (height // 16)``
        (mirrors ``QwenImagePipeline.latent_shape`` / the server's
        ``prepare_latent_shape``), NOT ``height // 8`` — the two differ for
        dims that are multiples of 8 but not of 16.
        """
        if traj.ndim == 5:
            return traj
        if traj.ndim != 4:
            raise ValueError(
                f"qwen_image: SGLang trajectory has rank {traj.ndim}, want 4 (packed) "
                f"or 5 (image-form); shape={tuple(traj.shape)}."
            )
        diffusion = get_diffusion_params(req.sampling_params)
        height = int(diffusion.height) if diffusion.height is not None else None
        width = int(diffusion.width) if diffusion.width is not None else None
        if height is None or width is None:
            raise ValueError(
                "qwen_image: need height/width from req.sampling_params to unpack "
                "the packed [B, T, S, C*4] trajectory; both must be set."
            )
        latent_h = 2 * (height // _QWEN_DOWNSAMPLE)
        latent_w = 2 * (width // _QWEN_DOWNSAMPLE)
        B, T, S, C_packed = traj.shape
        if S != (latent_h // 2) * (latent_w // 2):
            raise ValueError(
                f"qwen_image: packed token count S={S} != (latent_h/2)*(latent_w/2)="
                f"{(latent_h // 2) * (latent_w // 2)} (from height={height}, "
                f"width={width}). Schedule/recipe drift — fix the source rather "
                f"than silently reshape to a wrong spatial layout."
            )
        from unirl.models.qwen_image.diffusion import _unpack_latents

        flat = traj.reshape(B * T, S, C_packed)
        unpacked = _unpack_latents(flat, latent_h=latent_h, latent_w=latent_w)
        return unpacked.reshape(B, T, C_packed // 4, latent_h, latent_w).contiguous()


__all__ = ["QwenImageAdapter"]
