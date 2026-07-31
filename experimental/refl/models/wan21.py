"""Recipe-local WAN 2.1 T2V step + stage + pipeline for REFL BPTT.

Mirrors ``experimental.refl.models.wan22`` but targets the WAN 2.1 T2V
single-DiT stack. The REFL-specific pieces live here:

- :class:`Wan21ReflDiffusionStep` — strict recipe-local single-branch
  CFG predictor used only by REFL BPTT.
- :class:`Wan21ReflDiffusionStage` — subclass of the mainline
  :class:`WAN21DiffusionStage` with an extra :meth:`diffuse_with_grad`
  method implementing single-branch CFG + BPTT windowing (mid/final
  timestep) + optional per-step KL against the LoRA-disabled reference.
- :class:`Wan21ReflPipeline` — subclass of the mainline
  :class:`WAN21Pipeline` that post-swaps ``self.diffusion`` for the REFL
  variant above, keeping every other stage (text/vae/scheduler) intact.

The mainline WAN 2.1 code path is unchanged. REFL-specific single-branch
CFG prediction lives in recipe-local :class:`Wan21ReflDiffusionStep`.
"""

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch

from experimental.refl.models.types import DiffuseWithGradResult
from unirl.models.wan21.bundle import WAN21Bundle
from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.diffusion import WAN21DiffusionStage, WAN21DiffusionStep
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from unirl.models.wan21.pipeline import WAN21Pipeline
from unirl.train.lora import adapters_disabled
from unirl.types.primitives import Images, Texts
from unirl.types.sampling import DiffusionSamplingParams

# Matches the mainline module-level constant in unirl/models/wan21/diffusion.py.
# Not imported directly to keep the recipe decoupled from the mainline's
# private surface.
_WAN_TIMESTEP_SCALE: float = 1000.0

# Inclusive max for torch.Generator.manual_seed and torch initial_seed conventions.
MAX_TORCH_SEED = (1 << 63) - 1


class Wan21ReflDiffusionStep(WAN21DiffusionStep):
    """Recipe-local WAN 2.1 single-branch denoising for REFL BPTT."""

    def predict_noise(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        model: WAN21Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: WAN21Conditions,
        *,
        branch: str,
    ) -> torch.Tensor:
        """Run exactly one CFG branch through the WAN 2.1 transformer."""
        if branch not in {"cond", "uncond"}:
            raise ValueError(f"Wan21ReflDiffusionStep.predict_noise: unknown branch={branch!r}")
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Wan21ReflDiffusionStep.predict_noise: conditions.text.embeds is None")

        prompt_embeds = conditions.text.embeds
        if branch == "cond":
            branch_embeds = prompt_embeds
        else:
            neg = conditions.negative_text
            branch_embeds = (
                neg.embeds if neg is not None and neg.embeds is not None else torch.zeros_like(prompt_embeds)
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
            hidden_states=sample_cat,
            encoder_hidden_states=branch_embeds,
            timestep=timestep,
            return_dict=False,
            **extra,
        )[0]


class Wan21ReflDiffusionStage(WAN21DiffusionStage):
    """WAN 2.1 T2V diffusion stage + REFL BPTT sampling override.

    Adds ``diffuse_with_grad`` that keeps the autograd graph alive on the
    returned final latent, honouring the REFL-specific BPTT knobs
    (``mid_timestep`` / ``final_timestep`` / ``kl_weight``) via
    ``params.sampler_kwargs``. Everything else is inherited from the
    mainline stage — ``diffuse`` / ``replay`` / ``predict_noise`` /
    ``trainable_module`` are untouched.
    """

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
        """Differentiable WAN 2.1 T2V sampling for REFL-style BPTT training.

        Returns :class:`DiffuseWithGradResult` with the live-grad
        ``z_final`` + per-sample ``kl_loss`` ``[B]``.

        BPTT knobs (read from ``params.sampler_kwargs``):

        - ``mid_timestep`` (int, default 0): step index at which the forward
          switches from ``torch.no_grad`` to grad-enabled — implements
          truncated BPTT (only the last ``T - mid_timestep`` steps
          participate in backward). DRaFT-1 sets ``mid_timestep = T - 1``.
        - ``final_timestep`` (int, default ``num_inference_steps - 1``):
          early stop. The loop breaks once ``i >= final_timestep``.
        - ``kl_weight`` (float, default 0.0): when non-zero, per-step KL
          ``(pred - ref_pred)**2 / (2 * sigma**2)`` is reduced per sample
          and accumulated into the ``[B]`` ``kl_loss``. Per-sample (not a
          scalar) so DP_SCATTER merge/re-shard round-trips each shard's own
          KL. The actor multiplies it by its ``kl_weight`` at the loss site.

        ``initial_latents`` follows the same contract as :meth:`diffuse`:
        when provided, used verbatim and the internal RNG path is
        bypassed.
        """
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Wan21ReflDiffusionStage.diffuse_with_grad: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        schedule = schedule.to(device)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(
                f"Wan21ReflDiffusionStage.diffuse_with_grad: schedule length {schedule.shape[0]} != T+1={T + 1}"
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
                    f"Wan21ReflDiffusionStage.diffuse_with_grad: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != tuple(latent_shape):
                raise ValueError(
                    f"Wan21ReflDiffusionStage.diffuse_with_grad: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {tuple(latent_shape)} "
                    f"for num_frames={int(params.num_frames)}, "
                    f"height={int(params.height)}, width={int(params.width)}."
                )
            latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            latents = self.generate_latents(
                batch_size=batch_size,
                latent_shape=latent_shape,
                device=device,
                dtype=self.trajectory_dtype,
                base_seed=int(params.seed),
            )

        # BPTT knobs.
        sk: Dict[str, Any] = dict(getattr(params, "sampler_kwargs", {}) or {})
        mid_timestep = int(sk.get("mid_timestep", 0))
        final_timestep = int(sk.get("final_timestep", T - 1))
        kl_weight = float(sk.get("kl_weight", 0.0))
        if not (0 <= mid_timestep <= final_timestep < T):
            raise ValueError(
                f"Wan21ReflDiffusionStage.diffuse_with_grad: require 0 <= mid_timestep <= "
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

        guidance_scale = float(params.guidance_scale)
        use_cfg = guidance_scale > 1.0
        step = self.step
        if not isinstance(step, Wan21ReflDiffusionStep):
            raise TypeError(
                f"Wan21ReflDiffusionStage.diffuse_with_grad requires Wan21ReflDiffusionStep, got {type(step).__name__}."
            )

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)
            grad_enabled = i >= mid_timestep

            if use_cfg:
                # REFL parity: the conditional branch follows the BPTT grad
                # window, while the uncond/negative branch is always
                # stop-grad. Batched CFG would incorrectly add a
                # ``(1 - guidance_scale) * d(uncond)/dθ`` term.
                cond_ctx = nullcontext() if grad_enabled else torch.no_grad()
                with cond_ctx, autocast_ctx:
                    noise_pred_cond = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="cond",
                    )
                with torch.no_grad(), autocast_ctx:
                    noise_pred_uncond = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="uncond",
                    )
                noise_pred_cond = noise_pred_cond.float()
                noise_pred_uncond = noise_pred_uncond.float()
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
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
                    )
                noise_pred = noise_pred.float()
                kl_pred = noise_pred

            # Per-step KL against the LoRA-disabled reference (REFL-style).
            # When CFG is enabled, this intentionally computes KL on the
            # conditional prediction before the stop-grad uncond CFG mix.
            if kl_weight != 0.0 and grad_enabled:
                with torch.no_grad(), autocast_ctx, adapters_disabled(transformer):
                    ref_pred = step.predict_noise(
                        self.model,
                        latents,
                        sigma,
                        conditions,
                        branch="cond",
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


class Wan21ReflPipeline(WAN21Pipeline):
    """WAN 2.1 T2V pipeline for the REFL recipe.

    Reuses the mainline :class:`WAN21Pipeline` construction (text encode /
    condition build / VAE decode / schedule) and post-swaps
    ``self.diffusion`` for the REFL-flavoured stage. The parent already
    validates every constructor argument and wires the strategy / step /
    precision policy through — we only need to rebuild the diffusion
    stage with the same underlying components.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        old = self.diffusion
        assert isinstance(old, WAN21DiffusionStage), (
            f"Wan21ReflPipeline expects parent to build WAN21DiffusionStage, got {type(old).__name__}"
        )
        self.diffusion = Wan21ReflDiffusionStage(
            model=old.model,
            step=Wan21ReflDiffusionStep(),
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

        Mirrors the condition assembly of :meth:`WAN21Pipeline.generate`
        (text via the public ``build_conditions``, image slots attached with
        the diffusion geometry from ``params``), plus the REFL convention
        that an explicit negative prompt rides
        ``params.sampler_kwargs['negative_prompt']``.
        """
        guidance = float(params.guidance_scale)
        negative_prompt = (params.sampler_kwargs or {}).get("negative_prompt")
        negatives = (
            Texts(texts=[str(negative_prompt)] * len(texts.texts)) if negative_prompt and guidance > 1.0 else None
        )
        conds = self.build_conditions(texts, negatives=negatives, guidance_scale=guidance)

        if images is not None:
            if images.pixels is None or int(images.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"Wan21ReflPipeline.build_refl_conditions: image count "
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


__all__ = ["Wan21ReflDiffusionStep", "Wan21ReflDiffusionStage", "Wan21ReflPipeline"]
