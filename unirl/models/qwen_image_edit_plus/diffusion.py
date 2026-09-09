"""Qwen-Image-Edit-Plus diffusion: per-step kernel + (inherited) stage."""

from __future__ import annotations

from typing import Dict, List, Optional

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
    """Per-step Edit-Plus denoising kernel — adds source-image token concat."""

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
        """Run shape-homogeneous microbatches and restore sample order."""
        image_latent_cond = conditions.image_latent
        if image_latent_cond is None or not image_latent_cond.latents:
            raise ValueError(
                "QwenImageEditPlusDiffusionStep.predict_noise: conditions.image_latent is None. "
                "Edit-Plus is edit-only and requires a source image."
            )
        if len(image_latent_cond.latents) != int(sample.shape[0]):
            raise ValueError(
                "QwenImageEditPlusDiffusionStep.predict_noise: source-image latent count "
                f"{len(image_latent_cond.latents)} != sample batch {int(sample.shape[0])}"
            )

        groups: Dict[tuple[int, ...], List[int]] = {}
        for index, latent in enumerate(image_latent_cond.latents):
            if latent.ndim != 3:
                raise ValueError(
                    "QwenImageEditPlusDiffusionStep.predict_noise: each source-image latent "
                    f"must be [C, H, W], got {tuple(latent.shape)}"
                )
            groups.setdefault(tuple(latent.shape), []).append(index)

        result = None
        for indices in groups.values():
            batch_indices = torch.tensor(indices, device=sample.device, dtype=torch.long)
            sub_conditions = conditions.select(indices)
            sub_sigma = (
                sigma.index_select(0, batch_indices.to(sigma.device))
                if sigma.dim() > 0 and int(sigma.shape[0]) == int(sample.shape[0])
                else sigma
            )
            sub_image_latents = torch.stack(sub_conditions.image_latent.latents, dim=0)
            prediction = self._predict_noise_uniform(
                model,
                sample.index_select(0, batch_indices),
                sub_sigma,
                sub_conditions,
                sub_image_latents,
                guidance_scale=guidance_scale,
                latent_h=latent_h,
                latent_w=latent_w,
                distilled_guidance_scale=distilled_guidance_scale,
            )
            if result is None:
                result = prediction.new_empty(sample.shape)
            result.index_copy_(0, batch_indices, prediction)
        return result

    def _predict_noise_uniform(
        self,
        model: QwenImageEditPlusBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: QwenImageEditPlusConditions,
        image_latents: torch.Tensor,
        *,
        guidance_scale: float,
        latent_h: int,
        latent_w: int,
        distilled_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the Edit-Plus transformer with source-image token concat + CFG."""
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
        image_latents = image_latents.to(device=device, dtype=dtype)
        img_latent_h = int(image_latents.shape[-2])
        img_latent_w = int(image_latents.shape[-1])
        image_packed = _pack_latents(image_latents)
        packed = torch.cat([packed, image_packed], dim=1)
        img_shapes = [[(1, latent_h // 2, latent_w // 2), (1, img_latent_h // 2, img_latent_w // 2)]] * batch_size

        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size).to(device, dtype=dtype)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size).to(device, dtype=dtype)
        else:
            timestep = sigma.to(device, dtype=dtype)

        guidance = None
        if getattr(model.transformer.config, "guidance_embeds", False):
            guidance_value = guidance_scale if distilled_guidance_scale is None else float(distilled_guidance_scale)
            guidance = torch.tensor([guidance_value], device=device, dtype=torch.float32).expand(batch_size)

        max_true = int(prompt_embeds_mask.sum(dim=1).max().item())
        if prompt_embeds.shape[1] > max_true:
            prompt_embeds = prompt_embeds[:, :max_true]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_true]

        noise_pred_packed = model.transformer(
            hidden_states=packed,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

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
                neg_max = int(negative_prompt_embeds_mask.sum(dim=1).max().item())
                if negative_prompt_embeds.shape[1] > neg_max:
                    negative_prompt_embeds = negative_prompt_embeds[:, :neg_max]
                    negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :neg_max]
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                negative_noise_pred_packed = negative_noise_pred_packed[:, :noise_seq_len]
                comb = negative_noise_pred_packed + guidance_scale * (noise_pred_packed - negative_noise_pred_packed)
                cond_norm = torch.norm(noise_pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred_packed = comb * (cond_norm / comb_norm)

        return _unpack_latents(noise_pred_packed, latent_h=latent_h, latent_w=latent_w)


class QwenImageEditPlusDiffusionStage(QwenImageDiffusionStage):
    """Edit-Plus rollout-level diffusion stage — inherits the loop unchanged."""

    @staticmethod
    def _tile_conditions(
        conditions: QwenImageEditPlusConditions,
        repeats: int,
    ) -> QwenImageEditPlusConditions:
        """Tile text and ragged image conditions in step-major order."""
        return QwenImageEditPlusConditions.concat([conditions] * repeats)


__all__ = [
    "QwenImageEditPlusDiffusionStage",
    "QwenImageEditPlusDiffusionStep",
]
