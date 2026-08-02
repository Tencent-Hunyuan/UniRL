"""HunyuanVideo-1.5 diffusion: typed params + per-step kernel + rollout-level stage.

Three classes mirror :mod:`unirl.models.wan21.diffusion`:

- :class:`HunyuanVideo15DiffusionParams` — typed request-shape knobs
  (steps / guidance / size / num_frames / seed / sde_indices / eta /
  init_same_noise / samples_per_prompt / noise_group_ids).
- :class:`HunyuanVideo15DiffusionStep` — stateless per-step kernel.
  :meth:`predict_noise` packs the latent stream with zero
  ``cond_latents`` and zero ``cond_mask`` along channel-dim ``1``
  (HunyuanVideo-1.5's T2V contract), batches the dual text streams for
  CFG (when ``guidance_scale > 1``), and forwards through the
  transformer; the protocol-matching ``forward`` / ``step`` /
  ``step_with_logp`` ride on top.
- :class:`HunyuanVideo15DiffusionStage` — implements
  ``DiffusionStage[HunyuanVideo15Conditions]``. Owns the SDE strategy,
  loop bookkeeping, latent shape derivation, and the constant
  ``vision_num_semantic_tokens`` / ``vision_states_dim`` /
  ``timestep_scale`` knobs that the step kernel reads via kwargs.

Latent geometry
---------------
Video latents are 5D: ``[B, C, T_lat, H_lat, W_lat]`` where
- ``T_lat = (num_frames - 1) // temporal_compression_ratio + 1``
- ``H_lat = height // spatial_compression_ratio``
- ``W_lat = width // spatial_compression_ratio``

The VAE downsamples 16× spatially and 4× temporally on the default
HunyuanVideo-1.5 checkpoint. Segment storage is therefore 6D
``[B, K, C, T_lat, H_lat, W_lat]`` (the ``K`` axis is the trajectory
position count).

Channel-dim packing
-------------------
The transformer's ``in_channels`` is ``2 * latent_channels + 1`` because
of the upstream packing
``cat([latents, cond_latents, cond_mask], dim=1)``. For T2V both extra
streams are zero (T2V doesn't condition on a reference image), but the
shape contract is fixed — the cat happens inside :meth:`predict_noise`
so the segment store and SDE math never see the packed shape.

CFG
---
Standard chunked CFG: stack ``[cond, uncond]`` along the batch dim,
single transformer forward, chunk back, then
``uncond + guidance_scale * (cond - uncond)``. **No norm-correction**
(that's a Qwen-Image specialty, not HunyuanVideo-1.5).

Timestep
--------
The transformer takes ``timestep = sigma * 1000`` (sigma ∈ [0, 1] →
timestep ∈ [0, 1000]); ``TIMESTEP_SCALE`` is exposed on the step
kernel so the test fake transformer can sanity-check it.

Math mirrors the original HunyuanVideo-1.5 sampler and denoiser
(PR #101). The new-design path does NOT import legacy code; spec sync
is via review / test.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional, Tuple

import torch

from unirl.models.diffusion import DiffusionLatentSpec, DiffusionStep, VideoDiffusionRunner
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import HunyuanVideo15Bundle
from .conditions import HunyuanVideo15Conditions


class HunyuanVideo15DiffusionStep(DiffusionStep[HunyuanVideo15Bundle, HunyuanVideo15Conditions]):
    """Per-step HunyuanVideo-1.5 denoising kernel — stateless.

    Extends the :class:`DiffusionStep` protocol with HunyuanVideo-1.5-
    specific per-call kwargs (``vision_num_semantic_tokens``,
    ``vision_states_dim``) on :meth:`predict_noise`, :meth:`step`, and
    :meth:`step_with_logp`. The protocol surface stays structurally
    compatible because Python protocols are non-strict on extra kwargs.
    """

    # Sigma → transformer timestep scale (sigma ∈ [0, 1] → t ∈ [0, 1000]).
    TIMESTEP_SCALE: ClassVar[float] = 1000.0

    def predict_noise(
        self,
        model: HunyuanVideo15Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: HunyuanVideo15Conditions,
        *,
        guidance_scale: float,
        vision_num_semantic_tokens: int,
        vision_states_dim: int,
    ) -> torch.Tensor:
        """Pack the latent stream, run the transformer, optionally apply CFG.

        For T2V (current scope), ``cond_latents`` and ``cond_mask`` are
        zero placeholders matching the sample shape. ``image_embeds`` is
        a zero placeholder of shape
        ``[B, vision_num_semantic_tokens, vision_states_dim]``.
        Returns noise prediction of the same shape as ``sample``
        (``[B, C, T_lat, H_lat, W_lat]``).
        """
        text_mllm = conditions.text_mllm
        text_glyph = conditions.text_glyph
        if text_mllm is None or text_mllm.embeds is None or text_mllm.attn_mask is None:
            raise ValueError(
                "HunyuanVideo15DiffusionStep.predict_noise: conditions.text_mllm must carry both embeds and attn_mask."
            )
        if text_glyph is None or text_glyph.embeds is None or text_glyph.attn_mask is None:
            raise ValueError(
                "HunyuanVideo15DiffusionStep.predict_noise: conditions.text_glyph must carry both embeds and attn_mask."
            )

        prompt_embeds = text_mllm.embeds
        prompt_embeds_mask = text_mllm.attn_mask
        prompt_embeds_2 = text_glyph.embeds
        prompt_embeds_mask_2 = text_glyph.attn_mask

        if sample.ndim != 5:
            raise ValueError(
                f"HunyuanVideo15DiffusionStep.predict_noise: expected 5D sample "
                f"[B, C, T, H, W], got {tuple(sample.shape)}"
            )
        batch_size, _, latent_t, latent_h, latent_w = sample.shape
        device = sample.device
        dtype = prompt_embeds.dtype

        # T2V channel-dim packing: zero cond_latents (same shape as
        # latents) + zero cond_mask (single channel). The transformer's
        # ``in_channels`` is ``2 * latent_channels + 1`` by contract.
        sample_cast = sample.to(dtype)
        cond_latents = torch.zeros_like(sample_cast)
        cond_mask = torch.zeros(batch_size, 1, latent_t, latent_h, latent_w, device=device, dtype=dtype)

        # T2V vision placeholder. The transformer cross-attends to it
        # but the zero content is a no-op (matches upstream behavior).
        image_embeds = torch.zeros(
            batch_size,
            int(vision_num_semantic_tokens),
            int(vision_states_dim),
            device=device,
            dtype=dtype,
        )

        # Sigma → timestep scaling. Always cast to a [B]-shape tensor on
        # the model's compute dtype.
        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size)
        else:
            timestep = sigma
        timestep = timestep.to(device=device, dtype=dtype) * self.TIMESTEP_SCALE

        latent_model_input = torch.cat([sample_cast, cond_latents, cond_mask], dim=1)

        if guidance_scale > 1.0 and conditions.negative_text_mllm is not None:
            neg_mllm = conditions.negative_text_mllm
            neg_glyph = conditions.negative_text_glyph
            if (
                neg_mllm.embeds is None
                or neg_mllm.attn_mask is None
                or neg_glyph is None
                or neg_glyph.embeds is None
                or neg_glyph.attn_mask is None
            ):
                raise ValueError(
                    "HunyuanVideo15DiffusionStep.predict_noise: CFG-on requires "
                    "both negative_text_mllm and negative_text_glyph with non-None "
                    "embeds + attn_mask."
                )

            # Stack [cond, uncond] along batch dim — a single transformer
            # forward halves wall-clock vs two separate calls.
            doubled_input = torch.cat([latent_model_input, latent_model_input], dim=0)
            doubled_timestep = torch.cat([timestep, timestep], dim=0)
            encoder_hs = torch.cat([prompt_embeds, neg_mllm.embeds.to(dtype)], dim=0)
            encoder_mask = torch.cat([prompt_embeds_mask, neg_mllm.attn_mask], dim=0)
            encoder_hs_2 = torch.cat([prompt_embeds_2, neg_glyph.embeds.to(dtype)], dim=0)
            encoder_mask_2 = torch.cat([prompt_embeds_mask_2, neg_glyph.attn_mask], dim=0)
            image_embeds_doubled = torch.cat([image_embeds, image_embeds], dim=0)

            noise_pred = model.transformer(
                hidden_states=doubled_input,
                timestep=doubled_timestep,
                encoder_hidden_states=encoder_hs,
                encoder_attention_mask=encoder_mask,
                encoder_hidden_states_2=encoder_hs_2,
                encoder_attention_mask_2=encoder_mask_2,
                image_embeds=image_embeds_doubled,
                return_dict=False,
            )[0]
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return model.transformer(
            hidden_states=latent_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_embeds_mask,
            encoder_hidden_states_2=prompt_embeds_2,
            encoder_attention_mask_2=prompt_embeds_mask_2,
            image_embeds=image_embeds,
            return_dict=False,
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
        model: HunyuanVideo15Bundle,
        conditions: HunyuanVideo15Conditions,
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
        vision_num_semantic_tokens: int = 729,
        vision_states_dim: int = 1152,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step."""
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
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
        model: HunyuanVideo15Bundle,
        conditions: HunyuanVideo15Conditions,
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
        vision_num_semantic_tokens: int = 729,
        vision_states_dim: int = 1152,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``. ``log_prob``
        and ``prev_sample_mean`` are ``None`` for deterministic strategies.
        """
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
            vision_num_semantic_tokens=vision_num_semantic_tokens,
            vision_states_dim=vision_states_dim,
        )


class HunyuanVideo15DiffusionStage(VideoDiffusionRunner[HunyuanVideo15Bundle, HunyuanVideo15Conditions]):
    """HunyuanVideo-1.5 rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel,
    the precision policy, and the vision-placeholder shape constants
    that the step kernel reads via kwargs.

    ``diffuse(conditions, *, schedule, params)`` runs the full sampling
    loop and returns a ``LatentSegment`` carrying the 6D trajectory
    ``[B, K, C, T_lat, H_lat, W_lat]`` plus per-SDE log probs
    (``sde_logp [N, S]`` + ``sde_indices [S]``).

    ``replay(conditions, *, segment, params, step_indices=None)``
    recomputes log-probs for the SDE transitions in a stored
    ``LatentSegment``. Returns a :class:`ReplayResult` with ``log_probs``
    of shape ``[B, S']`` aligned with ``segment.sde_logp`` (or a slice
    when ``step_indices`` selects a subset) and ``prev_sample_means``
    for KL-penalty consumption.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy
    when HF auto-discovery yields nothing — HunyuanVideo-1.5's
    transformer block class is ``HunyuanVideo15TransformerBlock``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = (
        "HunyuanVideo15TransformerBlock",
        "HunyuanVideo15PatchEmbed",
        "HunyuanVideo15TokenRefiner",
    )

    # VAE downsample defaults from upstream; overridden at construction
    # if the bundle's VAE exposes ``spatial_compression_ratio`` /
    # ``temporal_compression_ratio`` attributes (it does on the canonical
    # checkpoint). ``DEFAULT_LATENT_CHANNELS=32`` matches the diffusers
    # ``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v`` VAE
    # (32-channel; transformer ``in_channels=65=2*32+1``,
    # ``out_channels=32``). ``HunyuanVideo15Pipeline.latent_shape`` reads
    # ``model_config.latent_channels`` first (config-side override) and
    # falls back to this default; the stage init reads VAE config first
    # and falls back to the transformer's ``out_channels`` and then to
    # this constant — three layers of inference, with a runtime fail-fast
    # in ``diffuse(initial_latents=...)`` when driver and stage disagree.
    DEFAULT_SPATIAL_DOWNSAMPLE: ClassVar[int] = 16
    DEFAULT_TEMPORAL_DOWNSAMPLE: ClassVar[int] = 4
    DEFAULT_LATENT_CHANNELS: ClassVar[int] = 32

    def __init__(
        self,
        *,
        model: HunyuanVideo15Bundle,
        step: HunyuanVideo15DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vision_num_semantic_tokens: int = 729,
        vision_states_dim: int = 1152,
        spatial_compression_ratio: Optional[int] = None,
        temporal_compression_ratio: Optional[int] = None,
        latent_channels: Optional[int] = None,
    ) -> None:
        super().__init__(
            model=model,
            step=step,
            strategy=strategy,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
        )
        self.vision_num_semantic_tokens = int(vision_num_semantic_tokens)
        self.vision_states_dim = int(vision_states_dim)

        # VAE geometry: prefer attributes on the VAE itself, then the VAE
        # config, then the dataclass-level defaults.
        vae = model.vae
        if spatial_compression_ratio is None:
            spatial_compression_ratio = (
                int(getattr(vae, "spatial_compression_ratio", 0))
                or int(getattr(getattr(vae, "config", None), "spatial_compression_ratio", 0))
                or self.DEFAULT_SPATIAL_DOWNSAMPLE
            )
        if temporal_compression_ratio is None:
            temporal_compression_ratio = (
                int(getattr(vae, "temporal_compression_ratio", 0))
                or int(getattr(getattr(vae, "config", None), "temporal_compression_ratio", 0))
                or self.DEFAULT_TEMPORAL_DOWNSAMPLE
            )
        self.spatial_compression_ratio = int(spatial_compression_ratio)
        self.temporal_compression_ratio = int(temporal_compression_ratio)

        if latent_channels is None:
            cfg = getattr(vae, "config", None)
            ch = int(getattr(cfg, "latent_channels", 0)) if cfg is not None else 0
            if not ch:
                # Fall back to transformer's reported out_channels.
                tx_cfg = getattr(model.transformer, "config", None)
                ch = int(getattr(tx_cfg, "out_channels", self.DEFAULT_LATENT_CHANNELS))
            latent_channels = ch
        self.latent_channels = int(latent_channels)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _latent_shape(self, *, height: int, width: int, num_frames: int) -> Tuple[int, int, int]:
        latent_t = (int(num_frames) - 1) // self.temporal_compression_ratio + 1
        latent_h = max(1, int(height) // self.spatial_compression_ratio)
        latent_w = max(1, int(width) // self.spatial_compression_ratio)
        return latent_t, latent_h, latent_w

    def _latent_spec(
        self,
        conditions: HunyuanVideo15Conditions,
        params: DiffusionSamplingParams,
    ) -> DiffusionLatentSpec:
        if conditions.text_mllm is None or conditions.text_mllm.embeds is None:
            raise ValueError("HunyuanVideo15DiffusionStage: conditions.text_mllm.embeds is None")
        embeds = conditions.text_mllm.embeds
        latent_t, latent_h, latent_w = self._latent_shape(
            height=params.height, width=params.width, num_frames=params.num_frames
        )
        return DiffusionLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=(self.latent_channels, latent_t, latent_h, latent_w),
        )

    def _step_kwargs(
        self,
        conditions: HunyuanVideo15Conditions,
        params: DiffusionSamplingParams,
        *,
        sample: torch.Tensor,
        step_index: int,
        num_steps: int,
        mode: str,
        state: Any,
    ) -> Mapping[str, Any]:
        del conditions, params, sample, step_index, num_steps, mode, state
        return {
            "vision_num_semantic_tokens": self.vision_num_semantic_tokens,
            "vision_states_dim": self.vision_states_dim,
        }


__all__ = [
    "HunyuanVideo15DiffusionStage",
    "HunyuanVideo15DiffusionStep",
]
