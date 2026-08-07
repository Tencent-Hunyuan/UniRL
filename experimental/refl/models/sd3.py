"""SD3 image ReFL adaptation — the second family on the BPTT contract.

Ported from the legacy core path (``unirl/trainer/refl.py`` +
``unirl/train/refl/policy.py`` + ``unirl/models/draft.py``, removed in this
change): the same DRaFT-K direct reward backprop, expressed through the
``experimental.refl`` contract so ``pipeline_target`` is the only thing a
config swaps between WAN video and SD3 image ReFL.

Deliberate non-support: CFG under BPTT (``guidance_scale > 1``). The core
``SD3DiffusionStep`` batches both CFG branches through one forward, which
under grad would add a ``(1 - g) * d(uncond)/dθ`` term; the legacy path
always trained at ``guidance_scale == 1`` and so does this port — a wrong
config fails loudly.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch

from experimental.refl.models.types import DiffuseWithGradResult
from unirl.models.sd3.conditions import SD3Conditions
from unirl.models.sd3.diffusion import SD3DiffusionStage
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.models.sd3.vae import SD3VAEDecodeStage
from unirl.train.lora import adapters_disabled
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment

MAX_TORCH_SEED = (1 << 63) - 1


class Sd3ReflDiffusionStage(SD3DiffusionStage):
    """SD3 diffusion stage + REFL BPTT sampling override.

    Reuses the mainline single-forward ``SD3DiffusionStep.predict_noise``
    (no CFG at ``guidance_scale == 1``); adds the grad-window loop with
    optional per-step KL against the LoRA-disabled reference.
    """

    def generate_latents(
        self,
        batch_size: int,
        latent_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        base_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if base_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(base_seed) % (MAX_TORCH_SEED + 1))
            return torch.randn(batch_size, *latent_shape, device=device, dtype=dtype, generator=generator)
        return torch.randn(batch_size, *latent_shape, device=device, dtype=dtype)

    def diffuse_with_grad(
        self,
        conditions: SD3Conditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> DiffuseWithGradResult:
        """Differentiable SD3 sampling for REFL-style BPTT training.

        Same knobs as the WAN stages (``params.sampler_kwargs``:
        ``mid_timestep`` / ``final_timestep``; KL switched by the actor via
        ``kl_weight``). Returns the live-grad ``z_final`` + per-sample
        ``kl_loss`` ``[B]``.
        """
        if float(params.guidance_scale) > 1.0:
            raise ValueError(
                "Sd3ReflDiffusionStage.diffuse_with_grad: CFG under BPTT is not supported "
                f"(guidance_scale={params.guidance_scale}); train at guidance_scale=1.0."
            )
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Sd3ReflDiffusionStage.diffuse_with_grad: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        schedule = schedule.to(device)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(
                f"Sd3ReflDiffusionStage.diffuse_with_grad: schedule length {schedule.shape[0]} != T+1={T + 1}"
            )
        self.strategy.init_schedule(schedule)

        latent_hw = (int(params.height) // self.vae_scale_factor, int(params.width) // self.vae_scale_factor)
        latent_shape = (self.latent_channels, *latent_hw)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size or tuple(initial_latents.shape[1:]) != latent_shape:
                raise ValueError(
                    f"Sd3ReflDiffusionStage.diffuse_with_grad: initial_latents shape "
                    f"{tuple(initial_latents.shape)} != ({batch_size}, {latent_shape})."
                )
            latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            if params.seed is None:
                raise ValueError(
                    "REFL's fixed-noise regime needs an explicit sampling seed "
                    "(roles.py: params.seed is used verbatim every rollout/rank); set sampling.seed."
                )
            latents = self.generate_latents(
                batch_size=batch_size,
                latent_shape=latent_shape,
                device=device,
                dtype=self.trajectory_dtype,
                base_seed=int(params.seed),
            )

        sk: Dict[str, Any] = dict(params.sampler_kwargs or {})
        mid_timestep = int(sk.get("mid_timestep", 0))
        final_timestep = int(sk.get("final_timestep", T - 1))
        kl_weight = float(sk.get("kl_weight", 0.0))
        if not (0 <= mid_timestep <= final_timestep < T):
            raise ValueError(
                f"Sd3ReflDiffusionStage.diffuse_with_grad: require 0 <= mid_timestep <= "
                f"final_timestep < num_inference_steps, got mid={mid_timestep} "
                f"final={final_timestep} T={T}."
            )

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        sigma_max = float(schedule[1].item()) if int(schedule.shape[0]) > 1 else 0.99

        transformer = self.model.transformer
        kl_total = torch.zeros(batch_size, device=device, dtype=torch.float32)
        kl_steps = 0

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)
            grad_enabled = i >= mid_timestep

            pred_ctx = nullcontext() if grad_enabled else torch.no_grad()
            with pred_ctx, autocast_ctx:
                noise_pred = self.step.predict_noise(
                    self.model,
                    latents,
                    sigma,
                    conditions,
                    guidance_scale=1.0,
                )
            noise_pred = noise_pred.float()

            if kl_weight != 0.0 and grad_enabled:
                with torch.no_grad(), autocast_ctx, adapters_disabled(transformer):
                    ref_pred = self.step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        guidance_scale=1.0,
                    )
                sigma_f32 = sigma.to(dtype=torch.float32)
                kl_step = ((noise_pred.float() - ref_pred.float()) ** 2 / (2.0 * sigma_f32**2)).flatten(1).mean(dim=1)
                kl_total = kl_total + kl_step
                kl_steps += 1

            transition_ctx = nullcontext() if grad_enabled else torch.no_grad()
            with transition_ctx:
                new_latents, _, _ = self.step.forward(
                    strategy=self.strategy,
                    noise_pred=noise_pred,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=float(params.eta),
                    sigma_max=sigma_max,
                    step_index=i,
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if i >= final_timestep:
                break

        kl_loss = kl_total / max(kl_steps, 1) if kl_steps > 0 else kl_total
        return DiffuseWithGradResult(z_final=latents, kl_loss=kl_loss)


class Sd3ReflVAEDecodeStage(SD3VAEDecodeStage):
    """SD3 VAE decode + the BPTT entry point.

    Reuses the mainline grad decode path (fp32 VAE + activation
    checkpoint); returns pixels in ``[0, 1]`` — the range the core
    differentiable image scorers (PickScore et al.) expect.
    """

    def decode_with_grad(self, z_final: torch.Tensor) -> torch.Tensor:
        if z_final.ndim != 4:
            raise ValueError(
                f"Sd3ReflVAEDecodeStage.decode_with_grad: expected 4D z_final [B, C, H, W], got {tuple(z_final.shape)}"
            )
        segment = LatentSegment(latents=z_final.unsqueeze(1))
        return self.decode(segment, grad=True, activation_checkpoint=True).to_dense()


class Sd3ReflPipeline(SD3Pipeline):
    """SD3 pipeline for the REFL package (post-swaps diffusion + vae_decode)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        old = self.diffusion
        assert isinstance(old, SD3DiffusionStage), (
            f"Sd3ReflPipeline expects parent to build SD3DiffusionStage, got {type(old).__name__}"
        )
        self.diffusion = Sd3ReflDiffusionStage(
            model=old.model,
            step=old.step,
            strategy=old.strategy,
            autocast_precision=old.autocast_dtype,
            trajectory_precision=old.trajectory_dtype,
            logprob_precision=old.logprob_dtype,
            vae_scale_factor=old.vae_scale_factor,
            latent_channels=old.latent_channels,
        )
        self.vae_decode = Sd3ReflVAEDecodeStage(self.bundle)

    def build_refl_conditions(
        self,
        texts: Texts,
        *,
        images: Optional[Images] = None,
        params: DiffusionSamplingParams,
    ) -> SD3Conditions:
        """Text-only REFL conditioning (SD3 is T2I; ``images`` must be None)."""
        if images is not None:
            raise ValueError("Sd3ReflPipeline.build_refl_conditions: SD3 ReFL is text-to-image; images must be None.")
        guidance = float(params.guidance_scale)
        negative_prompt = (params.sampler_kwargs or {}).get("negative_prompt")
        negatives = (
            Texts(texts=[str(negative_prompt)] * len(texts.texts)) if negative_prompt and guidance > 1.0 else None
        )
        return self.build_conditions(texts, negatives=negatives, guidance_scale=guidance)


__all__ = ["Sd3ReflDiffusionStage", "Sd3ReflVAEDecodeStage", "Sd3ReflPipeline"]
