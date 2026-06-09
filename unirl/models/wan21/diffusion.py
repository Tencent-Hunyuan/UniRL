"""WAN 2.1 diffusion: typed params + per-step kernel + rollout-level stage.

Three classes:

- ``WAN21DiffusionParams`` — typed request-shape knobs (steps / guidance /
  height / width / num_frames / seed / sde_indices / eta /
  init_same_noise / samples_per_prompt / noise_group_ids).
- ``WAN21DiffusionStep`` — stateless per-step kernel. ``step`` /
  ``step_with_logp`` take the model + conditions + strategy and run both
  CFG noise prediction and the SDE transition (via
  ``StepStrategy.denoise``). ``forward`` is a lower-level helper that
  takes a precomputed ``noise_pred``.
- ``WAN21DiffusionStage`` — implements ``DiffusionStage[WAN21Conditions]``.
  Owns the SDE ``strategy`` and the loop bookkeeping; delegates the
  per-step model+SDE work to the kernel. Also exposes ``replay`` for
  single-step log-prob replay during training.

CFG math derived from ``samplers/fsdp/wan_sampler.py`` and
``models/wan21.py::forward_denoiser`` (do NOT import legacy code).

WAN-specific deviations from SD3 v2:

- Hidden state is 5D ``[B, C, T_lat, H_lat, W_lat]`` (3D VAE temporal
  dim), not 4D. Latent shape is computed from ``num_frames`` /
  ``height`` / ``width`` with ``temporal_downsample=4`` /
  ``spatial_downsample=8``.
- ``WanTransformer3DModel`` takes ``encoder_hidden_states`` directly
  (no ``pooled_projections``).
- ``timestep`` is a 1D ``[B]`` tensor scaled by 1000 (matches WAN's
  training-time timestep convention).
- I2V channel concat: when ``conditions.image_latent`` is set the 20-
  channel mask+image payload is prepended on the channel axis before
  the transformer call (``in_channels`` jumps from 16 to 36).
- I2V CLIP-vision: when ``conditions.image_embed`` is set the patch
  embeddings are forwarded as ``encoder_hidden_states_image`` (batch-
  doubled to match the CFG ``[uncond, cond]`` stack).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import WAN21Bundle
from .conditions import WAN21Conditions

# WAN training-time timestep scale: sigma ∈ [0, 1] → timestep ∈ [0, 1000].
# Matches ``WAN21ModelBundle.TIMESTEP_SCALE`` in ``models/wan21.py``.
_WAN_TIMESTEP_SCALE: float = 1000.0


class WAN21DiffusionStep(DiffusionStep[WAN21Bundle, WAN21Conditions]):
    """Per-step WAN 2.1 denoising kernel — stateless.

    ``step`` / ``step_with_logp`` take the model + conditions + an SDE
    ``strategy`` per call, run CFG noise prediction internally, then
    apply the transition via ``strategy.denoise``. ``forward`` is the
    lower-level escape hatch that takes a precomputed ``noise_pred``.
    """

    def predict_noise(
        self,
        model: WAN21Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: WAN21Conditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run WAN transformer with CFG batched ``[uncond, cond]`` forward.

        Reads ``conditions.text.embeds`` for the conditional branch. For
        ``guidance_scale > 1`` reads ``conditions.negative_text.embeds``
        for the unconditional branch; falls back to zero embeddings when
        ``negative_text`` is ``None``.
        """
        if conditions.text is None:
            raise ValueError("WAN21DiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        if prompt_embeds is None:
            raise ValueError("WAN21DiffusionStep.predict_noise: conditions.text.embeds is None")

        batch_size = int(sample.shape[0])
        timestep = sigma * _WAN_TIMESTEP_SCALE
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif int(timestep.shape[0]) != batch_size:
            timestep = timestep.expand(batch_size)

        # WAN's transformer wants encoder_hidden_states in its own dtype;
        # latents are cast to match for the forward.
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

        # I2V CLIP-vision: when patch embeddings are present, forward
        # them as ``encoder_hidden_states_image``. Only emitted when the
        # WAN 2.1 transformer declares ``image_dim > 0`` (T2V never sets
        # this slot, so the kwarg is conditional to avoid leaking an
        # unknown kwarg to a T2V transformer signature).
        image_embed = conditions.image_embed
        image_embeds = image_embed.embeds if image_embed is not None and image_embed.embeds is not None else None
        extra: Dict[str, Any] = {}
        if image_embeds is not None:
            image_embeds = image_embeds.to(device=sample_cast.device, dtype=embeds_dtype)

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)

            if image_embeds is not None:
                extra["encoder_hidden_states_image"] = torch.cat([image_embeds, image_embeds], dim=0)

            noise_pred = model.transformer(
                hidden_states=torch.cat([sample_cat, sample_cat], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0),
                return_dict=False,
                **extra,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        if image_embeds is not None:
            extra["encoder_hidden_states_image"] = image_embeds

        return model.transformer(
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
        """Run one SDE transition given a precomputed ``noise_pred``.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``.
        ``prev_sample=None`` means sampling mode; otherwise log-prob
        replay. ``log_prob`` and ``prev_sample_mean`` are ``None`` for
        deterministic steps (``eta=0`` or DPM2-style ODE).
        """
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
        model: WAN21Bundle,
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``.
        """
        noise_pred = self.predict_noise(model, sample, sigma, conditions, guidance_scale=guidance_scale)
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
        model: WAN21Bundle,
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
        )


class WAN21DiffusionStage(DiffusionStage[WAN21Conditions]):
    """WAN 2.1 T2V rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies require a stable
    instance across the loop), the bundle, the kernel, and the precision
    policy. The kernel is stateless and is invoked per-step with the
    strategy passed in.

    ``diffuse(conditions, *, schedule, params)`` runs the full sampling
    loop and returns a ``LatentSegment`` carrying the trajectory plus
    per-SDE log probs (``sde_logp [N, S]`` + ``sde_indices [S]``).

    ``replay(conditions, *, segment, params, step_indices=None)``
    recomputes log-probs for the SDE transitions in a stored
    ``LatentSegment``. Returns a :class:`ReplayResult` for GRPO-style
    training (log_probs + per-step Gaussian mean μ_θ).

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy
    when HF auto-discovery yields nothing — diffusers'
    ``WanTransformer3DModel`` doesn't follow the HF transformers
    convention, so we declare it here.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("WanTransformerBlock",)

    # WAN VAE spatial/temporal downsampling factors. These are fixed for
    # the AutoencoderKLWan architecture (8× spatial, 4× temporal) and not
    # configurable per-request, so they live on the stage.
    _SPATIAL_DOWNSAMPLE: ClassVar[int] = 8
    _TEMPORAL_DOWNSAMPLE: ClassVar[int] = 4
    # Latent channel count fallback when ``vae.config.z_dim`` is absent.
    _DEFAULT_LATENT_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        *,
        model: WAN21Bundle,
        step: WAN21DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = self._SPATIAL_DOWNSAMPLE
        self.temporal_scale_factor = self._TEMPORAL_DOWNSAMPLE
        self.latent_channels = int(getattr(getattr(model.vae, "config", None), "z_dim", self._DEFAULT_LATENT_CHANNELS))

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _latent_shape(self, *, num_frames: int, height: int, width: int) -> Tuple[int, int, int, int]:
        """Return ``(C, T_lat, H_lat, W_lat)``.

        Pixel-space ``num_frames`` collapses to ``(num_frames - 1) //
        temporal_downsample + 1`` latent frames — matches WAN's reference
        implementation and the legacy sampler at
        ``samplers/fsdp/wan_sampler.py``.
        """
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: WAN21Conditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full WAN 2.1 T2V sampling. Returns a ``LatentSegment``.

        ``initial_latents`` (optional) — driver-shipped x_T per
        ``req.request_conditions['initial_latents']``. When provided,
        used verbatim and the internal ``generate_latents`` RNG path is
        bypassed. See :class:`SD3DiffusionStage.diffuse` for the contract.
        """
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("WAN21DiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"WAN21DiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        latent_shape = self._latent_shape(
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        )
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"WAN21DiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != tuple(latent_shape):
                raise ValueError(
                    f"WAN21DiffusionStage.diffuse: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {tuple(latent_shape)} "
                    f"for num_frames={int(params.num_frames)}, "
                    f"height={int(params.height)}, width={int(params.width)}."
                )
            latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=latent_shape,
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=int(params.seed),
            )

        # SDE indices: which steps record log probs.
        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        # Stored positions: SDE pairs ∪ {T} so VAE decode always has the clean latent.
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
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if (i + 1) in needed:
                stored_pairs.append((i + 1, latents.detach().clone()))

            if log_prob is not None:
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        # Pack into LatentSegment. WAN latents are 5D
        # [B, C, T_lat, H_lat, W_lat] so stacked is [B, K, C, T_lat, H_lat, W_lat].
        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)

        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None  # [B, S]
        sde_indices_tensor = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices_tensor = torch.tensor(positions_collected, dtype=torch.long, device=device)

        # Stamp ``modality=VIDEO`` via the factory helper. Plain
        # ``LatentSegment(...)`` would leave the ClassVar default
        # ``Modality.IMAGE`` in place, which is wrong for WAN T2V — any
        # downstream generic segment routing that branches on
        # ``segment.modality`` would mistake video latents for image
        # latents and (e.g.) try the image-only ``as_condition`` /
        # decode paths.
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
        conditions: WAN21Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Loops the per-step replay primitive (``step.step_with_logp`` with
        ``prev_sample`` set) over the segment's SDE indices (or the
        ``step_indices`` subset, which must be a subset of
        ``segment.sde_indices``). Returns a :class:`ReplayResult` with
        ``log_probs`` shape ``[B, len(target)]`` (cast to
        ``logprob_precision``) and ``prev_sample_means`` shape
        ``[B, len(target), C, T_lat, H_lat, W_lat]`` for KL penalty.

        Caller is responsible for ``.train()`` mode + grad scope; this
        method only manages the autocast scope.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("WAN21DiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("WAN21DiffusionStage.replay: segment.sigmas missing")

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"WAN21DiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
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
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"WAN21DiffusionStage.replay: strategy returned None log-prob "
                        f"at step_index={step_idx} (deterministic mode); replay "
                        f"requires a stochastic SDE strategy."
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
        conditions: WAN21Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Delegates to ``WAN21DiffusionStep.predict_noise``.
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on.

        For WAN 2.1, that's the bundle's transformer
        (``WanTransformer3DModel``) — the FSDP wrap target. Aux modules
        (VAE, text encoder) are siblings on the bundle, never under the
        transformer.
        """
        return self.model.transformer


__all__ = ["WAN21DiffusionStage", "WAN21DiffusionStep"]
