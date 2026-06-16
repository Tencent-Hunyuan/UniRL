"""Z-Image (S3-DiT single-stream) image adapter — frame-axis squeeze + caption mask.

Two Z-Image-specific overrides on top of the default ``ImageAdapter`` path; the
generic schedule policy (static ``model_config.shift`` — Z-Image is static-shift,
unlike Qwen-Image / Flux.2), the ``transformer.`` LoRA prefix, the SDE label, the
``Images`` decode, and the ``text`` / ``negative_text`` condition fusion are all
inherited.

* ``build_segment``: Z-Image's S3-DiT runs its denoise loop in 5-D image form
  ``[B, C, F=1, H, W]`` (``ZImageTransformer2DModel._as_image_list`` requires the
  frame axis), so the captured rollout trajectory arrives 6-D
  ``[B, T+1, C, 1, H, W]`` (``rollout_denoising_mixin`` stacks per-step 5-D latents
  along dim 1). The trainer-side segment and ``ZImageDiffusionStage.replay`` use
  image-form ``[B, T+1, C, H, W]``, so the singleton frame axis is squeezed here.
  A trajectory already in 5-D image form passes through unchanged.

* ``build_condition``: the stock Z-Image text stage (``zimage_postprocess_text``)
  trims a single-prompt encode to its real tokens and returns a bare
  ``[seq, hidden]`` with NO ``prompt_embeds_mask``; the engine then zero-pads
  mixed-length captions across forward chunks (``utils.tracks._cat_padded_rows``)
  with no mask to mark the pad. Replay's ``_caption_list`` forwards EVERY token
  when ``attn_mask is None``, so those zero-pad rows would be denoised as real
  caption tokens and dilute the GRPO policy gradient. Recover the validity mask
  from the embeds' non-zero rows whenever the engine emitted none; a real
  embeds-aligned mask (the multi-prompt encode path, which DOES emit
  ``prompt_embeds_mask``) is left untouched.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.rollout_req import RolloutReq


def _to_image_form_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """Squeeze Z-Image's singleton frame axis: 6-D ``[B,T,C,1,H,W]`` -> 5-D ``[B,T,C,H,W]``.

    A trajectory already in 5-D image form passes through unchanged. Raises on a
    non-singleton frame axis (a t2i rollout must have ``F==1``) rather than
    silently collapsing real frames.
    """
    if traj.ndim == 6:
        if int(traj.shape[3]) != 1:
            raise ValueError(
                f"z_image: trajectory has a non-singleton frame axis "
                f"(shape={tuple(traj.shape)}); expected [B, T, C, 1, H, W] for t2i."
            )
        return traj.squeeze(3)
    return traj


def _backfill_caption_mask(text: Optional[TextEmbedCondition]) -> Optional[TextEmbedCondition]:
    """Recover a per-token validity mask from embed non-zero rows when none was emitted.

    Only fires for token-level ``[B, T, D]`` embeds with no ``attn_mask`` (the
    single-prompt encode path). The cross-chunk pad is literally zeros
    (``_cat_padded_rows`` right-pads with ``new_zeros``) and real Qwen3 hidden
    states are non-zero, so ``(embeds != 0).any(-1)`` recovers exactly the valid
    rows; an unpadded single caption yields all-ones, which is also correct. A real
    emitted mask (multi-prompt encode) is returned untouched.
    """
    if text is None or text.attn_mask is not None or text.embeds is None or text.embeds.dim() != 3:
        return text
    mask = (text.embeds != 0).any(dim=-1).to(torch.long)
    return TextEmbedCondition(embeds=text.embeds, pooled=text.pooled, attn_mask=mask)


@register_adapter("z_image")
class ZImageAdapter(ImageAdapter):
    """Z-Image S3-DiT image adapter (image-form trajectory + caption mask backfill)."""

    def build_segment(
        self,
        req: RolloutReq,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        traj = _to_image_form_trajectory(utils.collect_trajectory_latents(results))
        if traj.ndim != 5:
            raise ValueError(
                f"z_image: expected a 5-D image-form trajectory [B, T, C, H, W] after the "
                f"frame squeeze; got rank {traj.ndim}, shape {tuple(traj.shape)}."
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

    def build_condition(self, results: List[RawResult]) -> Dict[str, object]:
        out = super().build_condition(results)
        for key in ("text", "negative_text"):
            if key in out:
                out[key] = _backfill_caption_mask(out[key])
        return out


__all__ = ["ZImageAdapter"]
