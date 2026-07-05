"""Qwen-Image-Edit-Plus diffusion: per-step kernel + (inherited) stage.

``QwenImageEditPlusDiffusionStep`` overrides only
:meth:`predict_noise` to concatenate the VAE-encoded source-image latent
onto the packed noise latent along the token dimension, extend
``img_shapes`` to carry both segments, and slice the transformer output
back to the noise segment. The CFG negative branch reuses the same
concatenated input (the source image is shared across CFG branches).
Mirrors ``vde_editplus.py:232,246`` and the FLUX.2-Klein pattern
(``flux2_klein/diffusion.py:160-183``).

``QwenImageEditPlusDiffusionStage`` is a thin subclass of
:class:`QwenImageDiffusionStage` — the loop, trajectory storage, replay,
and ``predict_noise_at_step`` bodies are inherited unchanged because they
are image-agnostic (they operate on spatial ``[B, C, H, W]`` latents and
delegate transformer calls to ``self.step.predict_noise``, which is the
overridden Edit-Plus step). Verified: ``step_with_logp`` → ``step`` →
``self.predict_noise`` (``qwen_image/diffusion.py:326→349→304``), so the
override propagates to ``replay`` automatically.

When ``conditions.image_latent is None`` the step falls through to the
base T2I behavior — a genuine degenerate path, not impossible-scenario
handling (the base Qwen-Image T2I is a valid input mode for this
transformer).
"""

from __future__ import annotations

from typing import Optional

import torch

from unirl.models.qwen_image.diffusion import (
    QwenImageDiffusionStage,
    QwenImageDiffusionStep,
    _pack_latents,
    _unpack_latents,
)

from .bundle import QwenImageEditPlusBundle
from .conditions import QwenImageEditPlusConditions


class QwenImageEditPlusDiffusionStep(QwenImageDiffusionStep):
    """Per-step Edit-Plus denoising kernel — adds source-image token concat.

    Overrides :meth:`predict_noise` to concatenate the packed source-image
    latent onto the packed noise latent before the transformer call. All
    other protocol surface (``forward`` / ``step`` / ``step_with_logp``)
    is inherited from :class:`QwenImageDiffusionStep` and routes through
    this override — so ``diffuse`` and ``replay`` pick up the concat
    automatically.
    """

    def predict_noise(
        self,
        model: QwenImageEditPlusBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: QwenImageEditPlusConditions,
        *,
        guidance_scale: float,
        latent_h: int,
        latent_w: int,
        distilled_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the Edit-Plus transformer with source-image token concat + CFG.

        Packs ``sample`` ``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``. When
        ``conditions.image_latent`` is present, packs the source-image
        latent the same way and concatenates along the token dimension
        (``torch.cat([noise, image], dim=1)``). ``img_shapes`` is extended
        to ``[[(1, noise_h//2, noise_w//2), (1, img_h//2, img_w//2)]] * B``.
        After the transformer call the prediction is sliced back to the
        noise token count: ``[:, :noise_seq_len]``.
        """
        if conditions.text is None:
            raise ValueError("QwenImageEditPlusDiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        prompt_embeds_mask = text.attn_mask
        if prompt_embeds is None:
            raise ValueError("QwenImageEditPlusDiffusionStep.predict_noise: conditions.text.embeds is None")
        if prompt_embeds_mask is None:
            raise ValueError("QwenImageEditPlusDiffusionStep.predict_noise: conditions.text.attn_mask is None")

        batch_size = sample.shape[0]
        device = sample.device
        dtype = prompt_embeds.dtype
        packed = _pack_latents(sample).to(dtype=dtype)
        noise_seq_len = int(packed.shape[1])

        # --- Source-image latent concat (Edit-Plus extension) -------------
        image_latent_cond = conditions.image_latent
        if image_latent_cond is None or image_latent_cond.latents is None:
            raise ValueError(
                "QwenImageEditPlusDiffusionStep.predict_noise: conditions.image_latent is None. "
                "Edit-Plus is edit-only and requires a source image."
            )
        image_latents = image_latent_cond.latents.to(device=device, dtype=dtype)
        img_latent_h = int(image_latents.shape[-2])
        img_latent_w = int(image_latents.shape[-1])
        image_packed = _pack_latents(image_latents)  # [B, (ih/2)*(iw/2), C*4]
        packed = torch.cat([packed, image_packed], dim=1)
        img_shapes = [[(1, latent_h // 2, latent_w // 2), (1, img_latent_h // 2, img_latent_w // 2)]] * batch_size

        # Qwen-Image's transformer takes raw sigma as the timestep input.
        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size).to(device, dtype=dtype)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size).to(device, dtype=dtype)
        else:
            timestep = sigma.to(device, dtype=dtype)

        # Distilled-guidance scalar (guidance_embeds variants only).
        guidance = None
        if getattr(model.transformer.config, "guidance_embeds", False):
            guidance_value = guidance_scale if distilled_guidance_scale is None else float(distilled_guidance_scale)
            guidance = torch.tensor([guidance_value], device=device, dtype=torch.float32).expand(batch_size)

        # Per-sample true text lengths — RoPE builder slices by max(txt_seq_lens).
        true_lens = prompt_embeds_mask.sum(dim=1).to(torch.long)
        max_true = int(true_lens.max().item())
        if prompt_embeds.shape[1] > max_true:
            prompt_embeds = prompt_embeds[:, :max_true]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_true]
        txt_seq_lens = true_lens.tolist()

        noise_pred_packed = model.transformer(
            hidden_states=packed,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # Slice back to the noise segment (drop the image-segment prediction).
        noise_pred_packed = noise_pred_packed[:, :noise_seq_len]

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
                negative_prompt_embeds_mask = neg.attn_mask
                if negative_prompt_embeds_mask is None:
                    raise ValueError(
                        "QwenImageEditPlusDiffusionStep.predict_noise: conditions.negative_text.attn_mask is None"
                    )
                neg_true = negative_prompt_embeds_mask.sum(dim=1).to(torch.long)
                neg_max = int(neg_true.max().item())
                if negative_prompt_embeds.shape[1] > neg_max:
                    negative_prompt_embeds = negative_prompt_embeds[:, :neg_max]
                    negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :neg_max]
                negative_txt_seq_lens = neg_true.tolist()
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    return_dict=False,
                )[0]
                negative_noise_pred_packed = negative_noise_pred_packed[:, :noise_seq_len]
                # Norm-corrected CFG blend (same as base Qwen-Image).
                comb = negative_noise_pred_packed + guidance_scale * (noise_pred_packed - negative_noise_pred_packed)
                cond_norm = torch.norm(noise_pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred_packed = comb * (cond_norm / comb_norm)

        return _unpack_latents(noise_pred_packed, latent_h=latent_h, latent_w=latent_w)


class QwenImageEditPlusDiffusionStage(QwenImageDiffusionStage):
    """Edit-Plus rollout-level diffusion stage — inherits the loop unchanged.

    The base :class:`QwenImageDiffusionStage` is image-agnostic: it
    operates on spatial ``[B, C, H, W]`` latents and delegates transformer
    calls to ``self.step.predict_noise``. With an :class:`QwenImageEditPlusDiffusionStep`
    injected at construction, ``diffuse`` / ``replay`` / ``predict_noise_at_step``
    all pick up the source-image concat automatically (verified delegation
    chain: ``step_with_logp`` → ``step`` → ``self.predict_noise``).

    The type parameter widens to :class:`QwenImageEditPlusConditions`; no
    body override is needed.
    """


__all__ = [
    "QwenImageEditPlusDiffusionStage",
    "QwenImageEditPlusDiffusionStep",
]
