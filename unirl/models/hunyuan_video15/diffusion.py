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

from contextlib import nullcontext
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

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


class HunyuanVideo15DiffusionStage(DiffusionStage[HunyuanVideo15Conditions]):
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
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
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

    def diffuse(
        self,
        conditions: HunyuanVideo15Conditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full HunyuanVideo-1.5 T2V sampling. Returns a ``LatentSegment``
        with 6D trajectory storage and ``modality=VIDEO``.

        ``initial_latents`` (optional) — driver-shipped x_T per
        ``req.request_conditions['initial_latents']``; see
        :class:`SD3DiffusionStage.diffuse` for the contract.
        """
        from unirl.sde.noise import generate_latents

        if conditions.text_mllm is None or conditions.text_mllm.embeds is None:
            raise ValueError("HunyuanVideo15DiffusionStage.diffuse: conditions.text_mllm.embeds is None")
        prompt_embeds = conditions.text_mllm.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(
                f"HunyuanVideo15DiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}"
            )
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        latent_t, latent_h, latent_w = self._latent_shape(
            height=params.height, width=params.width, num_frames=params.num_frames
        )
        expected_latent_shape = (self.latent_channels, latent_t, latent_h, latent_w)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"HunyuanVideo15DiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"HunyuanVideo15DiffusionStage.diffuse: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {expected_latent_shape} "
                    f"for num_frames={int(params.num_frames)}, "
                    f"height={int(params.height)}, width={int(params.width)}."
                )
            latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=expected_latent_shape,
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=int(params.seed),
            )

        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, latents.detach().clone()))
        sde_logp_list: List[torch.Tensor] = []

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        sigma_max = float(schedule[1].item()) if int(schedule.shape[0]) > 1 else 0.99

        step_kwargs: Dict = {
            "vision_num_semantic_tokens": self.vision_num_semantic_tokens,
            "vision_states_dim": self.vision_states_dim,
        }

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)
            step_eta = float(params.eta) if i in sde_set else 0.0

            with torch.no_grad(), autocast_ctx:
                new_latents, log_prob, _ = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=float(params.guidance_scale),
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=i,
                    **step_kwargs,
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if (i + 1) in needed:
                stored_pairs.append((i + 1, latents.detach().clone()))

            if log_prob is not None:
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        # 6D stacked storage: [B, K, C, T_lat, H_lat, W_lat].
        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)

        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None
        sde_indices_tensor = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices_tensor = torch.tensor(positions_collected, dtype=torch.long, device=device)

        # Stamp modality=VIDEO via the factory so downstream
        # ``segment.modality``-based routing doesn't mistake video latents
        # for image latents.
        return make_video_segment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices_tensor,
            sde_logp=sde_logp,
            sde_indices=sde_indices_tensor,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: HunyuanVideo15Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Caller is responsible for ``.train()`` mode + grad scope; this
        method only manages the autocast scope.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("HunyuanVideo15DiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("HunyuanVideo15DiffusionStage.replay: segment.sigmas missing")
        if segment.latents.ndim != 6:
            raise ValueError(
                f"HunyuanVideo15DiffusionStage.replay: expected latents "
                f"[B, K, C, T_lat, H_lat, W_lat], got {tuple(segment.latents.shape)}"
            )

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"HunyuanVideo15DiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        step_kwargs: Dict = {
            "vision_num_semantic_tokens": self.vision_num_semantic_tokens,
            "vision_states_dim": self.vision_states_dim,
        }

        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []
        with autocast_ctx:
            for step_idx in target:
                sigma = sigmas[step_idx].to(dtype=torch.float32)
                sigma_next = sigmas[step_idx + 1].to(dtype=torch.float32)
                sample = segment.latents_at(step_idx)
                prev_sample = segment.latents_at(step_idx + 1)
                _, log_prob, prev_mean = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=sample,
                    prev_sample=prev_sample,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=float(params.guidance_scale),
                    eta=float(params.eta),
                    sigma_max=sigma_max,
                    step_index=step_idx,
                    **step_kwargs,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"HunyuanVideo15DiffusionStage.replay: strategy returned "
                        f"None log-prob at step_index={step_idx} (deterministic mode); "
                        f"replay requires a stochastic SDE strategy."
                    )
                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    # ------------------------------------------------------------------
    # Single-step noise prediction (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def predict_noise_at_step(
        self,
        conditions: HunyuanVideo15Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        ``vision_num_semantic_tokens`` / ``vision_states_dim`` come from
        the Stage construction (model-architecture constants), not from
        ``params`` (per-request data).
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
            vision_num_semantic_tokens=self.vision_num_semantic_tokens,
            vision_states_dim=self.vision_states_dim,
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the FSDP wrap target — the bundle's transformer."""
        return self.model.transformer


__all__ = [
    "HunyuanVideo15DiffusionStage",
    "HunyuanVideo15DiffusionStep",
]
