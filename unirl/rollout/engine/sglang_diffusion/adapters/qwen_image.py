"""Qwen-Image image adapter — packed sequence trajectory, generic schedule.

Qwen-Image's transformer is a packed-token model like FLUX.2-Klein: SGLang's
denoising loop carries ``[B, S, C*4]`` tokens (2×2 patchify over a 16-channel
VAE latent), so the trajectory arrives packed ``[B, T+1, S, 64]`` and the shared
``PackedImageAdapter`` unpacks it before assembly. Unlike Klein (which keeps
packed channels at patch resolution), Qwen's ``_depack`` target is the **true**
channel form ``[B, T+1, 16, latent_h, latent_w]`` — exactly what the trainside
replay consumes (``models/qwen_image/diffusion.py`` stores segments unpacked and
packs only at the transformer boundary), so segments are interchangeable between
the trainside and sglang engines.

Everything else is the default path: the generic schedule policy reads
``use_dynamic_shifting`` / ``dynamic_shift_overrides`` / ``shift_terminal``
off the model config (no Klein-style factory needed — Qwen's μ is the linear
``calculate_dynamic_mu`` form), the ``transformer.`` LoRA prefix and
``text`` / ``negative_text`` condition fusion come from ``ImageAdapter``,
and CFG stays on ``guidance_scale`` (the server's qwen pipeline applies the
same norm-preserving true-CFG blend as the trainside replay).
"""

from __future__ import annotations

from typing import Any, Dict, List

from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.packed_image import PackedImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.text import TextEmbedCondition


@register_adapter("qwen_image")
class QwenImageAdapter(PackedImageAdapter):
    """Qwen-Image — packed sequence-style trajectory unpacked to true channels."""

    def _depack(self, flat, *, h_pat, w_pat):
        """Qwen depatchifies 2×2 to the true 16-channel latent grid (latent = 2×patch).

        Grid arithmetic is the canonical ``latent_h = 2 * (height // 16)`` (mirrors
        ``QwenImagePipeline.latent_shape`` / the server's ``prepare_latent_shape``),
        NOT ``height // 8`` — the two differ for dims that are multiples of 8 but
        not of 16. ``h_pat`` / ``w_pat`` are already ``height // 16`` / ``width // 16``.
        """
        from unirl.models.qwen_image.diffusion import _unpack_latents

        return _unpack_latents(flat, latent_h=2 * h_pat, latent_w=2 * w_pat)

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Stage override: backfill the attention masks the serving stack drops.

        The sglang qwen text-encode postprocess returns padded embeds WITHOUT
        the parallel mask (``qwen_image_postprocess_text`` pads to the
        request's batch max and discards the lengths), but the trainside
        replay hard-requires ``attn_mask`` alongside ``embeds``
        (``models/qwen_image/diffusion.py``). Within one request every sample
        shares the prompt (the engine de-expands one group per forward), so
        the embeds are unpadded and an all-ones mask is EXACT — and matches
        what the server itself attended to. Cross-chunk length differences are
        then handled by the padded fusions (mask rows extended with zeros).
        """
        out = super().build_condition(results)
        for key in ("text", "negative_text"):
            cond = out.get(key)
            if cond is not None and cond.embeds is not None and cond.attn_mask is None:
                out[key] = TextEmbedCondition(
                    embeds=cond.embeds,
                    pooled=cond.pooled,
                    attn_mask=cond.embeds.new_ones(cond.embeds.shape[:2]),
                )
        return out


__all__ = ["QwenImageAdapter"]
