"""WAN 2.2 diffusion: dual-transformer per-step kernel + rollout-level stage.

WAN 2.2 introduces sigma-boundary-based routing between two transformer
copies:

- ``sigma >= boundary_ratio`` → ``high_noise`` branch (coarse structure)
- ``sigma <  boundary_ratio`` → ``low_noise`` branch (detail
  refinement); optionally with its own ``guidance_scale_2``

The routing is **per-step, per-sigma**, so it belongs in
:class:`WAN22DiffusionStep` (the kernel), not in the stage loop. The
stage layer is responsible for bookkeeping only and is otherwise
identical to WAN 2.1.

CFG batching follows the WAN 2.1 pattern (``[uncond, cond]`` along
batch dim, ``chunk(2)``, interpolate). The transformer call goes
through ``WAN22Bundle.transformer.forward(use_high_noise=..., ...)``
(the :class:`WanDualTransformer` composite) so branch routing stays
behind the stage abstraction. FSDPPolicy does not root-wrap the
composite; it discovers and fully-shards the ``WanTransformerBlock``
instances under both branches.

The replay path uses the same per-step kernel with ``prev_sample`` set;
this means each replay step also routes by sigma, mirroring how the
rollout was produced.

Math derived from ``models/wan22.py::forward_denoiser`` and
``samplers/fsdp/wan22_sampler.py`` (do NOT import legacy code).
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

import torch

from unirl.models.diffusion import SingleStreamDiffusionStep, SingleStreamLatentSpec, SingleStreamVideoDiffusionRunner
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import WAN22Bundle

_WAN_TIMESTEP_SCALE: float = 1000.0


class WAN22DiffusionStep(SingleStreamDiffusionStep[WAN22Bundle, WAN21Conditions]):
    """Per-step WAN 2.2 denoising kernel — stateless, dual-transformer routing.

    For each call, decides whether to route through the high- or
    low-noise sub-transformer based on the current sigma vs the
    bundle's ``boundary_ratio``. The low-noise branch optionally uses
    a separate guidance scale (``guidance_scale_2``); when ``None``,
    the same scale flows through both branches.
    """

    @staticmethod
    def _select_for_sigma(
        sigma: torch.Tensor,
        guidance_scale: float,
        guidance_scale_2: Optional[float],
        *,
        boundary_ratio: float,
    ) -> Tuple[bool, float]:
        """Decide which sub-transformer to use and which guidance to apply.

        Returns ``(use_high_noise, active_guidance)``.

        Boundary policy (matches ``models/wan22.py::_select_guidance_for_sigma``
        in spirit, but only branches on ``sigma`` directly — WAN 2.2's
        sigma schedule lives in ``[0, 1]`` and ``boundary_ratio`` is
        defined in that same domain):

        - ``sigma >= boundary_ratio`` → high_noise + ``guidance_scale``
        - ``sigma <  boundary_ratio`` → low_noise + ``guidance_scale_2``
          (falls back to ``guidance_scale`` when the per-stage scale
          is ``None``)

        Per-sample sigma policy: when ``sigma`` is a 1D tensor of per-
        sample values, we read ``sigma[0]`` to pick the branch for the
        whole batch. This is consistent with how both rollout and replay
        invoke the kernel — every call site here passes a single
        ``schedule[i]`` scalar (or broadcasts it), so all samples share
        the same sigma in any one ``predict_noise`` call. If a future
        consumer ever ships heterogeneous per-sample sigmas through this
        step, this assumption must be revisited (it would require
        per-sample routing — likely two forwards followed by per-sample
        gather).
        """
        sigma_val = float(sigma.item()) if sigma.dim() == 0 else float(sigma.flatten()[0].item())
        if sigma_val >= boundary_ratio:
            return True, float(guidance_scale)
        active = float(guidance_scale_2) if guidance_scale_2 is not None else float(guidance_scale)
        return False, active

    def predict_noise(
        self,
        model: WAN22Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: WAN21Conditions,
        *,
        guidance_scale: float,
        guidance_scale_2: Optional[float] = None,
    ) -> torch.Tensor:
        """Run dual-transformer noise prediction with optional CFG.

        Routes by sigma against ``model.boundary_ratio`` then applies CFG
        in the active branch. The call always goes through
        ``model.transformer.forward`` (the :class:`WanDualTransformer`
        composite) so sampling code depends on one stage-level routing
        surface rather than reaching into high/low sub-transformers.
        """
        if conditions.text is None:
            raise ValueError("WAN22DiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        if prompt_embeds is None:
            raise ValueError("WAN22DiffusionStep.predict_noise: conditions.text.embeds is None")

        use_high_noise, active_guidance = self._select_for_sigma(
            sigma,
            guidance_scale,
            guidance_scale_2,
            boundary_ratio=model.boundary_ratio,
        )

        batch_size = int(sample.shape[0])
        timestep = sigma * _WAN_TIMESTEP_SCALE
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif int(timestep.shape[0]) != batch_size:
            timestep = timestep.expand(batch_size)

        embeds_dtype = prompt_embeds.dtype
        sample_cast = sample.to(dtype=embeds_dtype)

        # I2V channel concat: when an image-condition latent is present,
        # prepend it on the channel axis (16 noise + 20 mask+image →
        # 36 transformer ``in_channels``). Identical across cond/uncond.
        image_latent = conditions.image_latent
        if image_latent is not None and image_latent.latents is not None:
            sample_cat = torch.cat(
                [sample_cast, image_latent.latents.to(device=sample_cast.device, dtype=embeds_dtype)],
                dim=1,
            )
        else:
            sample_cat = sample_cast

        # I2V CLIP-vision: forward ``encoder_hidden_states_image`` only
        # when the slot is populated. WAN 2.2's mainstream checkpoints
        # have ``image_dim == 0`` and never see this kwarg; the
        # composite :class:`WanDualTransformer` transparently routes
        # ``**kwargs`` to both ``high_noise`` and ``low_noise``.
        image_embed = conditions.image_embed
        image_embeds = image_embed.embeds if image_embed is not None and image_embed.embeds is not None else None
        extra: Dict[str, Any] = {}
        if image_embeds is not None:
            image_embeds = image_embeds.to(device=sample_cast.device, dtype=embeds_dtype)

        if active_guidance > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            if image_embeds is not None:
                extra["encoder_hidden_states_image"] = torch.cat([image_embeds, image_embeds], dim=0)

            noise_pred = model.transformer(
                use_high_noise=use_high_noise,
                hidden_states=torch.cat([sample_cat, sample_cat], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0),
                return_dict=False,
                **extra,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + active_guidance * (noise_pred_cond - noise_pred_uncond)

        if image_embeds is not None:
            extra["encoder_hidden_states_image"] = image_embeds

        return model.transformer(
            use_high_noise=use_high_noise,
            hidden_states=sample_cat,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            return_dict=False,
            **extra,
        )[0]

    # ---- Protocol surface ---------------------------------------------------

    def forward(
        self,
        *,
        strategy: StepStrategy,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one SDE transition given a precomputed ``noise_pred``."""
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=step_index,
        )

    def step(
        self,
        model: WAN22Bundle,
        conditions: WAN21Conditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        guidance_scale_2: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run dual-transformer forward + SDE transition. End-to-end one step.

        ``guidance_scale_2`` is the WAN 2.2 extension over the
        Protocol's ``step`` signature (Protocol accepts ``**kwargs``-style
        extension for model-specific knobs; here it's keyword-only with
        a default of ``None`` so we stay backwards compatible with
        callers that don't pass it).
        """
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
        )
        return self.forward(
            strategy=strategy,
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
        )

    def step_with_logp(
        self,
        model: WAN22Bundle,
        conditions: WAN21Conditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
        guidance_scale_2: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run dual-transformer forward + SDE transition (delegates to ``step``)."""
        return self.step(
            model,
            conditions,
            strategy=strategy,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            guidance_scale=guidance_scale,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
            guidance_scale_2=guidance_scale_2,
        )


class WAN22DiffusionStage(SingleStreamVideoDiffusionRunner[WAN22Bundle, WAN21Conditions]):
    """WAN 2.2 T2V rollout-level diffusion stage with dual-transformer routing.

    Owns the SDE ``strategy`` (stateful strategies require a stable
    instance across the loop) + bundle + kernel + precision policy.
    The kernel routes per-step between high- and low-noise transformers
    based on the bundle's ``boundary_ratio``; the stage loop is
    otherwise identical to WAN 2.1.

    ``replay`` also routes per-sigma, mirroring the rollout exactly.

    ``_no_split_modules`` provides the FSDPPolicy fallback for HF
    auto-discovery; WanTransformerBlock is shared by both
    sub-transformers in :class:`WanDualTransformer`.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("WanTransformerBlock",)

    _SPATIAL_DOWNSAMPLE: ClassVar[int] = 8
    _TEMPORAL_DOWNSAMPLE: ClassVar[int] = 4
    _DEFAULT_LATENT_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        *,
        model: WAN22Bundle,
        step: WAN22DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__(
            model=model,
            step=step,
            strategy=strategy,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
        )
        self.vae_scale_factor = self._SPATIAL_DOWNSAMPLE
        self.temporal_scale_factor = self._TEMPORAL_DOWNSAMPLE
        self.latent_channels = int(getattr(getattr(model.vae, "config", None), "z_dim", self._DEFAULT_LATENT_CHANNELS))

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _latent_shape(self, *, num_frames: int, height: int, width: int) -> Tuple[int, int, int, int]:
        if (int(num_frames) - 1) % self._TEMPORAL_DOWNSAMPLE != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample={self._TEMPORAL_DOWNSAMPLE} requires "
                f"(num_frames - 1) % {self._TEMPORAL_DOWNSAMPLE} == 0, got num_frames={num_frames}; "
                f"valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        latent_t = (int(num_frames) - 1) // self.temporal_scale_factor + 1
        latent_h = int(height) // self.vae_scale_factor
        latent_w = int(width) // self.vae_scale_factor
        return (self.latent_channels, latent_t, latent_h, latent_w)

    def _latent_spec(
        self,
        conditions: WAN21Conditions,
        params: DiffusionSamplingParams,
    ) -> SingleStreamLatentSpec:
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("WAN22DiffusionStage: conditions.text.embeds is None")
        embeds = conditions.text.embeds
        return SingleStreamLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=self._latent_shape(
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ),
        )

    def _step_kwargs(
        self,
        conditions: WAN21Conditions,
        params: DiffusionSamplingParams,
        *,
        sample: torch.Tensor,
        step_index: int,
        num_steps: int,
        mode: str,
        state: Any,
    ) -> Mapping[str, Any]:
        del conditions, sample, step_index, num_steps, state
        guidance_scale_2 = params.guidance_scale_2
        if mode != "predict" and guidance_scale_2 is None:
            guidance_scale_2 = self.model.guidance_scale_2
        return {"guidance_scale_2": guidance_scale_2}


__all__ = ["WAN22DiffusionStage", "WAN22DiffusionStep"]
