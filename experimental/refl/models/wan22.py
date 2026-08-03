"""Recipe-local WAN 2.2 step + stage + pipeline for REFL BPTT."""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch

from experimental.refl.models.types import DiffuseWithGradResult
from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from unirl.models.wan22.bundle import WAN22Bundle
from unirl.models.wan22.diffusion import WAN22DiffusionStage, WAN22DiffusionStep
from unirl.models.wan22.pipeline import WAN22Pipeline
from unirl.train.lora import adapters_disabled
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams

_WAN_TIMESTEP_SCALE: float = 1000.0

MAX_TORCH_SEED = (1 << 63) - 1


class Wan22ReflDiffusionStep(WAN22DiffusionStep):
    """Recipe-local WAN 2.2 single-branch denoising for REFL BPTT."""

    def predict_noise(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: WAN22Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: WAN21Conditions,
        *,
        branch: str,
        use_high_noise: Optional[bool] = None,
    ) -> torch.Tensor:
        """Run exactly one CFG branch through the currently active WAN 2.2 DiT."""
        if branch not in {"cond", "uncond"}:
            raise ValueError(f"Wan22ReflDiffusionStep.predict_noise: unknown branch={branch!r}")
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Wan22ReflDiffusionStep.predict_noise: conditions.text.embeds is None")

        prompt_embeds = conditions.text.embeds
        if branch == "cond":
            branch_embeds = prompt_embeds
        else:
            neg = conditions.negative_text
            branch_embeds = (
                neg.embeds if neg is not None and neg.embeds is not None else torch.zeros_like(prompt_embeds)
            )

        if use_high_noise is None:
            use_high_noise, _ = self._select_for_sigma(
                sigma,
                1.0,
                None,
                boundary_ratio=model.boundary_ratio,
            )

        batch_size = int(sample.shape[0])
        timestep = sigma * _WAN_TIMESTEP_SCALE
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif int(timestep.shape[0]) != batch_size:
            timestep = timestep.expand(batch_size)

        embeds_dtype = branch_embeds.dtype
        sample_cast = sample.to(dtype=embeds_dtype)

        image_latent = conditions.image_latent
        if image_latent is not None and image_latent.latents is not None:
            sample_cat = torch.cat(
                [sample_cast, image_latent.latents.to(device=sample_cast.device, dtype=embeds_dtype)],
                dim=1,
            )
        else:
            sample_cat = sample_cast

        extra: Dict[str, Any] = {}
        image_embed = conditions.image_embed
        image_embeds = image_embed.embeds if image_embed is not None and image_embed.embeds is not None else None
        if image_embeds is not None:
            extra["encoder_hidden_states_image"] = image_embeds.to(device=sample_cast.device, dtype=embeds_dtype)

        return model.transformer(
            use_high_noise=bool(use_high_noise),
            hidden_states=sample_cat,
            encoder_hidden_states=branch_embeds,
            timestep=timestep,
            return_dict=False,
            **extra,
        )[0]


class Wan22ReflDiffusionStage(WAN22DiffusionStage):
    """WAN 2.2 diffusion stage + REFL BPTT sampling override."""

    def generate_latents(
        self,
        batch_size: int,
        latent_shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        base_seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        High-level function for generating initial latents.

        Args:
            batch_size: Total number of samples
            latent_shape: Shape of a single latent (C, H, W) or (C, T, H, W)
            device: Device for the tensor
            dtype: Data type for the tensor

        Returns:
            Latent tensor [batch_size, *latent_shape]
        """
        if base_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(base_seed) % (MAX_TORCH_SEED + 1))
            return torch.randn(
                batch_size,
                *latent_shape,
                device=device,
                dtype=dtype,
                generator=generator,
            )
        return torch.randn(
            batch_size,
            *latent_shape,
            device=device,
            dtype=dtype,
        )

    def diffuse_with_grad(
        self,
        conditions: WAN21Conditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> DiffuseWithGradResult:
        """Differentiable WAN 2.2 sampling for REFL-style BPTT training.

        Returns :class:`DiffuseWithGradResult` with the live-grad
        ``z_final`` + per-sample ``kl_loss`` ``[B]``.
        """

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Wan22ReflDiffusionStage.diffuse_with_grad: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        schedule = schedule.to(device)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(
                f"Wan22ReflDiffusionStage.diffuse_with_grad: schedule length {schedule.shape[0]} != T+1={T + 1}"
            )
        self.strategy.init_schedule(schedule)

        latent_shape = self._latent_shape(
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        )
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"Wan22ReflDiffusionStage.diffuse_with_grad: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != tuple(latent_shape):
                raise ValueError(
                    f"Wan22ReflDiffusionStage.diffuse_with_grad: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {tuple(latent_shape)} "
                    f"for num_frames={int(params.num_frames)}, "
                    f"height={int(params.height)}, width={int(params.width)}."
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

        sk: Dict[str, Any] = dict(getattr(params, "sampler_kwargs", {}) or {})
        mid_timestep = int(sk.get("mid_timestep", 0))
        final_timestep = int(sk.get("final_timestep", T - 1))
        kl_weight = float(sk.get("kl_weight", 0.0))
        if not (0 <= mid_timestep <= final_timestep < T):
            raise ValueError(
                f"Wan22ReflDiffusionStage.diffuse_with_grad: require 0 <= mid_timestep <= "
                f"final_timestep < num_inference_steps, got mid={mid_timestep} "
                f"final={final_timestep} T={T}."
            )

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        sigma_max = float(schedule[1].item()) if int(schedule.shape[0]) > 1 else 0.99

        guidance_scale = float(params.guidance_scale)
        guidance_scale_2 = (
            params.guidance_scale_2 if params.guidance_scale_2 is not None else self.model.guidance_scale_2
        )
        boundary_ratio = float(self.model.boundary_ratio)
        step = self.step
        if not isinstance(step, Wan22ReflDiffusionStep):
            raise TypeError(
                f"Wan22ReflDiffusionStage.diffuse_with_grad requires Wan22ReflDiffusionStep, got {type(step).__name__}."
            )

        dual = self.model.transformer
        kl_total = torch.zeros(batch_size, device=device, dtype=torch.float32)
        kl_steps = 0

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)

            use_high_noise, active_guidance = step._select_for_sigma(
                sigma,
                guidance_scale,
                guidance_scale_2,
                boundary_ratio=boundary_ratio,
            )
            use_cfg = active_guidance > 1.0
            grad_enabled = i >= mid_timestep

            if use_cfg:
                cond_ctx = nullcontext() if grad_enabled else torch.no_grad()
                with cond_ctx, autocast_ctx:
                    noise_pred_cond = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="cond",
                        use_high_noise=use_high_noise,
                    )
                with torch.no_grad(), autocast_ctx:
                    noise_pred_uncond = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="uncond",
                        use_high_noise=use_high_noise,
                    )
                noise_pred_cond = noise_pred_cond.float()
                noise_pred_uncond = noise_pred_uncond.float()
                noise_pred = noise_pred_uncond + active_guidance * (noise_pred_cond - noise_pred_uncond)
                kl_pred = noise_pred_cond
            else:
                pred_ctx = nullcontext() if grad_enabled else torch.no_grad()
                with pred_ctx, autocast_ctx:
                    noise_pred = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="cond",
                        use_high_noise=use_high_noise,
                    )
                noise_pred = noise_pred.float()
                kl_pred = noise_pred

            if kl_weight != 0.0 and grad_enabled:
                with torch.no_grad(), autocast_ctx, adapters_disabled(dual):
                    ref_pred = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="cond",
                        use_high_noise=use_high_noise,
                    )
                sigma_f32 = sigma.to(dtype=torch.float32)
                kl_step = ((kl_pred.float() - ref_pred.float()) ** 2 / (2.0 * sigma_f32**2)).flatten(1).mean(dim=1)
                kl_total = kl_total + kl_step
                kl_steps += 1

            transition_ctx = nullcontext() if grad_enabled else torch.no_grad()
            with transition_ctx:
                new_latents, _, _ = step.forward(
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


class Wan22ReflPipeline(WAN22Pipeline):
    """WAN 2.2 pipeline for the REFL recipe (Scheme A: post-swap ``self.diffusion``)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        old = self.diffusion
        assert isinstance(old, WAN22DiffusionStage), (
            f"Wan22ReflPipeline expects parent to build WAN22DiffusionStage, got {type(old).__name__}"
        )
        self.diffusion = Wan22ReflDiffusionStage(
            model=old.model,
            step=Wan22ReflDiffusionStep(),
            strategy=old.strategy,
            autocast_precision=old.autocast_dtype,
            trajectory_precision=old.trajectory_dtype,
            logprob_precision=old.logprob_dtype,
        )

    def build_refl_conditions(
        self,
        texts: Texts,
        *,
        images: Optional[Images] = None,
        params: DiffusionSamplingParams,
    ) -> WAN21Conditions:
        """Full REFL conditioning: text + CFG negative + optional I2V image.

        Mirrors the condition assembly of :meth:`WAN22Pipeline.generate` —
        the negative branch is encoded against the **effective** guidance
        ``max(guidance_scale, guidance_scale_2)`` (WAN22 routes CFG by sigma
        across the low/high-noise DiTs), and image slots are attached with
        the diffusion geometry from ``params``. An explicit negative prompt
        rides ``params.sampler_kwargs['negative_prompt']``.
        """
        primary_g = float(params.guidance_scale)
        low_g = float(params.guidance_scale_2) if params.guidance_scale_2 is not None else primary_g
        effective_guidance = max(primary_g, low_g)
        negative_prompt = (params.sampler_kwargs or {}).get("negative_prompt")
        negatives = (
            Texts(texts=[str(negative_prompt)] * len(texts.texts))
            if negative_prompt and effective_guidance > 1.0
            else None
        )
        conds = self.build_conditions(texts, negatives=negatives, guidance_scale=effective_guidance)

        if images is not None:
            if images.pixels is None or int(images.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"Wan22ReflPipeline.build_refl_conditions: image count "
                    f"{None if images.pixels is None else int(images.pixels.shape[0])} "
                    f"!= text count {len(texts.texts)}"
                )
            image_latent = WAN21ImageLatentEncodeStage(
                self.bundle,
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ).encode(images)
            image_embed = (
                WAN21CLIPVisionEncodeStage(self.bundle).encode(images)
                if getattr(self.bundle, "uses_clip_vision", False)
                else None
            )
            conds = dataclasses.replace(conds, image_latent=image_latent, image_embed=image_embed)
        return conds


__all__ = ["Wan22ReflDiffusionStep", "Wan22ReflDiffusionStage", "Wan22ReflPipeline"]
