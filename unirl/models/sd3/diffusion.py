"""SD3 diffusion: typed params + per-step kernel + rollout-level stage.

Three classes:

- ``SD3DiffusionParams`` — typed request-shape knobs (steps / guidance /
  size / seed / sde_indices / eta / init_same_noise / samples_per_prompt /
  noise_group_ids / max_sequence_length).
- ``SD3DiffusionStep`` — stateless per-step kernel. ``step`` /
  ``step_with_logp`` take the model + conditions + strategy and run both
  CFG noise prediction and the SDE transition (via
  ``StepStrategy.denoise``). ``forward`` is a lower-level helper that
  takes a precomputed ``noise_pred``.
- ``SD3DiffusionStage`` — implements ``DiffusionStage[SD3Conditions]``.
  Owns the SDE ``strategy`` and the loop bookkeeping; delegates the
  per-step model+SDE work to the kernel. Also exposes ``replay`` for
  single-step log-prob replay during training.

CFG math copied from ``samplers/fsdp/sd3_sampler.py:158-207`` (do NOT
import legacy code).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ClassVar, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import SD3Bundle
from .conditions import SD3Conditions


class SD3DiffusionStep(DiffusionStep[SD3Bundle, SD3Conditions]):
    """Per-step SD3 denoising kernel — stateless.

    ``step`` / ``step_with_logp`` take the model + conditions + an SDE
    ``strategy`` per call, run CFG noise prediction internally, then
    apply the transition via ``strategy.denoise``. ``forward`` is the
    lower-level escape hatch that takes a precomputed ``noise_pred``.
    """

    def predict_noise(
        self,
        model: SD3Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: SD3Conditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run SD3 transformer with CFG batched ``[uncond, cond]`` forward.

        Reads ``conditions.text.embeds`` / ``.pooled`` for the conditional
        branch. For ``guidance_scale > 1`` reads
        ``conditions.negative_text.embeds`` / ``.pooled`` for the
        unconditional branch; falls back to zero embeddings if
        ``negative_text`` is ``None``.
        """
        if conditions.text is None:
            raise ValueError("SD3DiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        if text.embeds is None:
            raise ValueError("SD3DiffusionStep.predict_noise: conditions.text.embeds is None")
        # Pin every model input to the transformer's device. Dedicated-engine
        # (vLLM-Omni) replay hands sample/conditions back on CPU; the trainside
        # engine already has them on GPU (these ``.to`` calls are then no-ops).
        dev = model.device
        sample = sample.to(dev)
        sigma = sigma.to(dev)
        prompt_embeds = text.embeds.to(dev)
        pooled_prompt_embeds = text.pooled.to(dev) if text.pooled is not None else None

        # Cast latent/embeds to the transformer's param dtype before the bf16
        # pos_embed conv — autocast doesn't reliably catch the first conv input
        # under FSDP2 wrap (the DiffusionNFT forward-process path hits this; GRPO/FlowDPPO
        # replay feeds already-bf16 latents). Idempotent when dtype matches.
        try:
            model_dtype = next(model.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = sample.dtype
        sample = sample.to(dtype=model_dtype)
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=model_dtype)

        batch_size = sample.shape[0]
        timestep = sigma * 1000.0
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.shape[0] != batch_size:
            timestep = timestep.expand(batch_size)

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds.to(dev)
                negative_pooled_prompt_embeds = neg.pooled.to(dev) if neg.pooled is not None else None
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = (
                    torch.zeros_like(pooled_prompt_embeds) if pooled_prompt_embeds is not None else None
                )

            if pooled_prompt_embeds is not None:
                pooled_batched = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            else:
                pooled_batched = None

            noise_pred = model.transformer(
                hidden_states=torch.cat([sample, sample], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0),
                pooled_projections=pooled_batched,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return model.transformer(
            hidden_states=sample,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
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
        model: SD3Bundle,
        conditions: SD3Conditions,
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
        model: SD3Bundle,
        conditions: SD3Conditions,
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


class SD3DiffusionStage(DiffusionStage[SD3Conditions]):
    """SD3 rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel,
    and the precision policy. The kernel is stateless and is invoked
    per-step with the strategy passed in.

    ``diffuse(conditions, *, schedule, params)`` runs the full sampling
    loop and returns a ``LatentSegment`` carrying the trajectory plus
    per-SDE log probs (``sde_logp [N, S]`` + ``sde_indices [S]``).

    ``replay(conditions, *, segment, params, step_indices=None)``
    recomputes log-probs for the SDE transitions in a stored
    ``LatentSegment``. Returns a :class:`ReplayResult` with ``log_probs``
    of shape ``[B, S']`` aligned with ``segment.sde_logp`` (or a slice
    when ``step_indices`` selects a subset) and ``prev_sample_means``
    for KL-penalty consumption. Used by GRPO-style training.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy
    when HF auto-discovery (`type(trainable_root).__mro__._no_split_modules`)
    yields nothing — diffusers' ``SD3Transformer2DModel`` doesn't
    follow the HF transformers convention, so we declare it here.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("JointTransformerBlock",)

    def __init__(
        self,
        *,
        model: SD3Bundle,
        step: SD3DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 8,
        latent_channels: int = 16,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = vae_scale_factor
        self.latent_channels = latent_channels

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: SD3Conditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full SD3 sampling. Returns a ``LatentSegment``.

        ``initial_latents`` (optional) — the per-sample x_T tensor pre-shipped
        by the driver via ``req.request_conditions['initial_latents']``. When
        provided, used verbatim and the internal ``generate_latents`` RNG
        path is bypassed (driver owns reproducibility / group-sharing /
        cross-rollout variation). When ``None``, the legacy internal path
        runs with ``params.seed``-keyed Gaussian noise (used by tests and
        by engines that don't pre-ship noise).
        """
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("SD3DiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"SD3DiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        latent_h = int(params.height) // int(self.vae_scale_factor)
        latent_w = int(params.width) // int(self.vae_scale_factor)
        expected_latent_shape = (int(self.latent_channels), latent_h, latent_w)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"SD3DiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"SD3DiffusionStage.diffuse: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {expected_latent_shape} "
                    f"for height={int(params.height)}, width={int(params.width)}."
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
        sigma_max = schedule[1].float() if int(schedule.shape[0]) > 1 else torch.tensor(0.99)

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

        # Pack into LatentSegment.
        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)  # [B, K, C, H, W]

        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None  # [B, S]
        sde_indices_tensor = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices_tensor = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
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
        conditions: SD3Conditions,
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
        ``[B, len(target), C, H, W]`` for KL penalty.

        Caller is responsible for ``.train()`` mode + grad scope; this
        method only manages the autocast scope.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("SD3DiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("SD3DiffusionStage.replay: segment.sigmas missing")

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"SD3DiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        # Dedicated-engine (vLLM-Omni) rollouts hand the trajectory back on
        # CPU; pin replay to the model's (CUDA) device so the forward and the
        # autocast context match the transformer weights. Trainside segments
        # are already on this device.
        device = torch.device(self.model.device)
        sigmas = segment.sigmas.to(device)
        sigma_max = sigmas[1].float() if int(sigmas.shape[0]) > 1 else torch.tensor(0.99)

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
                sample = segment.latents_at(step_idx).to(device)
                prev_sample = segment.latents_at(step_idx + 1).to(device)
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
                        f"SD3DiffusionStage.replay: strategy returned None log-prob "
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
        conditions: SD3Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Delegates to ``SD3DiffusionStep.predict_noise`` (the same kernel
        both ``diffuse`` and ``replay`` call internally), so CFG batching
        + guidance scale handling stay identical to the sampling path.
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

        For SD3, that's the bundle's transformer (``SD3Transformer2DModel``)
        — the FSDP wrap target. Aux modules (VAE, text encoders) are
        siblings on the bundle, never under the transformer.
        """
        return self.model.transformer


__all__ = ["SD3DiffusionStage", "SD3DiffusionStep"]
