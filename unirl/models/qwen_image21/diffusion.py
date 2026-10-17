"""Qwen-Image-2.1 diffusion: per-step kernel over length-grouped prompts + the shared Qwen-Image rollout stage."""

from __future__ import annotations

from typing import ClassVar, Optional, Tuple

import torch

from unirl.models.qwen_image.conditions import QwenImageConditions
from unirl.models.qwen_image.diffusion import QwenImageDiffusionStage, QwenImageDiffusionStep
from unirl.sde.kernels import StepStrategy

from .bundle import QwenImage21Bundle

VAE_SCALE_FACTOR = 16
LATENT_CHANNELS = 64
TOKENS_PER_IMAGE_SLOT = 4


class QwenImage21DiffusionStep(QwenImageDiffusionStep):
    """Per-step Qwen-Image-2.1 kernel; samples run grouped by prompt length so no padding enters RoPE (see README)."""

    def predict_noise(
        self,
        model: QwenImage21Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: QwenImageConditions,
        *,
        guidance_scale: float,
        latent_h: int,
        latent_w: int,
        distilled_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Velocity ``[B, 64, h, w]`` for latents ``[B, 64, h, w]`` at ``sigma``."""
        if guidance_scale != 1.0 or distilled_guidance_scale is not None:
            raise ValueError(
                "QwenImage21DiffusionStep.predict_noise: Qwen-Image-2.1 samples without CFG; set "
                f"guidance_scale=1.0 and distilled_guidance_scale=None (got {guidance_scale}, {distilled_guidance_scale})."
            )
        text = conditions.text
        if text is None or text.embeds is None or text.attn_mask is None:
            raise ValueError("QwenImage21DiffusionStep.predict_noise: conditions.text.embeds / attn_mask missing")
        if latent_h % 2 or latent_w % 2:
            raise ValueError(
                f"QwenImage21DiffusionStep.predict_noise: latent grid {latent_h}x{latent_w} must be even "
                "(one encoder image slot covers 2x2 latent tokens)."
            )

        batch_size = sample.shape[0]
        dtype = text.embeds.dtype
        timestep = (sigma.to(sample.device, torch.float32).expand(batch_size) * 1000).to(dtype) / 1000
        num_tokens = latent_h * latent_w
        lengths = text.attn_mask.sum(dim=1)

        result = sample.new_empty(sample.shape, dtype=dtype)
        for length in lengths.unique().tolist():
            idx = (lengths == length).nonzero(as_tuple=True)[0]
            group = len(idx)
            packed = sample.index_select(0, idx).to(dtype).flatten(2).transpose(1, 2)  # [G, h*w, 64]
            img_mask = torch.cat(
                [
                    torch.zeros(group, length, dtype=torch.bool, device=sample.device),
                    torch.ones(group, num_tokens // TOKENS_PER_IMAGE_SLOT, dtype=torch.bool, device=sample.device),
                ],
                dim=1,
            )
            out = model.transformer(
                hidden_states=packed,
                encoder_hidden_states=text.embeds.index_select(0, idx)[:, :length],
                timestep=timestep.index_select(0, idx),
                img_shapes=[[(1, latent_h, latent_w)]] * group,
                img_mask=img_mask,
                return_dict=False,
            )[0]
            velocity = out[:, -num_tokens:].transpose(1, 2).reshape(group, -1, latent_h, latent_w)
            result.index_copy_(0, idx, velocity.to(dtype))
        return result


class QwenImage21DiffusionStage(QwenImageDiffusionStage):
    """Qwen-Image-2.1 rollout-level stage: the Qwen-Image loop at 16x / 64-channel latent geometry."""

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("QwenImage21TransformerBlock",)

    def __init__(
        self,
        *,
        model: QwenImage21Bundle,
        step: QwenImage21DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str,
        trajectory_precision: str,
        logprob_precision: str,
        batch_replay_steps: bool,
    ) -> None:
        super().__init__(
            model=model,
            step=step,
            strategy=strategy,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
            vae_scale_factor=VAE_SCALE_FACTOR,
            latent_channels=LATENT_CHANNELS,
            batch_replay_steps=batch_replay_steps,
        )


__all__ = ["LATENT_CHANNELS", "QwenImage21DiffusionStage", "QwenImage21DiffusionStep", "VAE_SCALE_FACTOR"]
