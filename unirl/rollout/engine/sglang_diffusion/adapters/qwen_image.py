"""Qwen-Image image adapter — packed sequence trajectory, generic schedule.

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
``text`` / ``negative_text`` condition fusion come from ``ImageAdapter``,
and CFG stays on ``guidance_scale`` (the server's qwen pipeline applies the
same norm-preserving true-CFG blend as the trainside replay).
"""

from __future__ import annotations

from typing import List, Optional

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.rollout_req import RolloutReq

# Qwen-Image patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
_QWEN_DOWNSAMPLE = 16


@register_adapter("qwen_image")
class QwenImageAdapter(ImageAdapter):
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
        """Collect, unpack Qwen's packed ``[B, T, S, C*4]`` to true channels, assemble.

        Depatchifies 2×2 to the true 16-channel latent grid. Grid arithmetic is
        the canonical ``latent_h = 2 * (height // 16)`` (mirrors
        ``QwenImagePipeline.latent_shape`` / the server's ``prepare_latent_shape``),
        NOT ``height // 8`` — the two differ for dims that are multiples of 8 but
        not of 16. 5-D arrivals (image-form) skip the unpack.
        """
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 5:
            B, T, S, C, h_pat, w_pat = utils.validate_packed_trajectory(
                traj, req, family="qwen_image", downsample=_QWEN_DOWNSAMPLE
            )
            from unirl.models.qwen_image.diffusion import _unpack_latents

            flat = traj.reshape(B * T, S, C)
            unpacked = _unpack_latents(flat, latent_h=2 * h_pat, latent_w=2 * w_pat)
            traj = unpacked.reshape(B, T, C // 4, 2 * h_pat, 2 * w_pat).contiguous()
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=req.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    # build_condition: inherited from ImageAdapter. The engine emits Qwen-Image's
    # embeds-aligned ``prompt_embeds_mask`` (the mask the server's DiT attends under)
    # via ``_patches/patch_conditions``, and ``utils.tracks.fuse_text_conditions``
    # mounts it whenever it aligns with the embeds — so no model-specific backfill is
    # needed. If the mask is genuinely absent, trainside replay raises (fail loud)
    # rather than fabricating an all-ones mask that is wrong for mixed-length batches.


__all__ = ["QwenImageAdapter"]
