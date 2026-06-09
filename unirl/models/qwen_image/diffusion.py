"""Qwen-Image diffusion: typed params + per-step kernel + rollout-level stage.

Three classes mirror :mod:`unirl.models.sd3.diffusion`:

- :class:`QwenImageDiffusionParams` — typed request-shape knobs (steps /
  guidance / size / seed / sde_indices / eta / init_same_noise /
  samples_per_prompt / noise_group_ids /
  ``distilled_guidance_scale``).
- :class:`QwenImageDiffusionStep` — stateless per-step kernel. Wraps
  :meth:`predict_noise` (which packs latents into the
  ``[B, S, C*4]`` patch layout the Qwen-Image transformer expects,
  builds ``img_shapes``, runs CFG with **norm correction**, then unpacks
  the noise prediction back to ``[B, C, H, W]``) around
  ``StepStrategy.denoise``. The protocol-matching ``forward`` /
  ``step`` / ``step_with_logp`` ride on top.
- :class:`QwenImageDiffusionStage` — implements
  ``DiffusionStage[QwenImageConditions]``. Owns the SDE strategy and
  loop bookkeeping; segment latents stay in spatial ``[B, C, H, W]``
  shape so :class:`QwenImageVAEDecodeStage` can read them directly.

CFG math
--------
The Qwen-Image pipeline does **not** use the standard
``uncond + scale * (cond - uncond)`` form; it applies the combined
prediction, then rescales it to preserve the per-token L2 norm of the
conditional prediction. This is what the legacy
``models/qwen_image.py::forward_denoiser`` (PR #104 lines 506-511)
does, and it ships as the official Qwen-Image inference recipe::

    comb = neg + scale * (cond - neg)
    cond_norm = ||cond||_{dim=-1, keepdim=True}
    comb_norm = ||comb||_{dim=-1, keepdim=True}
    noise_pred = comb * (cond_norm / comb_norm)

The CFG batching is per-branch (two separate transformer forwards),
not the SD3-style ``[uncond, cond]`` chunked forward, because Qwen-VL
prompts have variable-length sequences with attention masks that
don't match between branches.

Latent packing
--------------
The Qwen-Image transformer operates on patchified latents
``[B, (H/2)*(W/2), C*4]`` (2×2 patches in the spatial plane). The
SDE loop, segment storage, and noise generation all use the
**unpacked** ``[B, C, H, W]`` shape; only :meth:`predict_noise`
packs/unpacks at the transformer boundary. This keeps
``LatentSegment.latents`` in the same ``[B, K, C, H, W]`` shape SD3
and Wan use, so :class:`QwenImageVAEDecodeStage` follows the SD3 decode
protocol without per-shape special-casing.

Math mirrors PR #104's ``qwen_image_sampler.py`` / ``forward_denoiser``.
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

from .bundle import QwenImageBundle
from .conditions import QwenImageConditions

# --------------------------------------------------------------------------
# Pack / unpack helpers — module-level so unit tests can import them
# without constructing the stage.
# --------------------------------------------------------------------------


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``.

    Reshapes the spatial grid into 2×2 patches and flattens. Mirrors
    ``samplers/fsdp/qwen_image_sampler.py::_pack_latents``.
    """
    if latents.ndim != 4:
        raise ValueError(f"_pack_latents: expected [B, C, H, W], got {tuple(latents.shape)}")
    batch_size, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"_pack_latents: H ({height}) and W ({width}) must be divisible by 2")
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def _unpack_latents(latents: torch.Tensor, *, latent_h: int, latent_w: int) -> torch.Tensor:
    """``[B, S, C*4]`` → ``[B, C, H, W]`` (inverse of :func:`_pack_latents`).

    Requires ``S == (H/2)*(W/2)``; ``H = latent_h``, ``W = latent_w``.
    """
    if latents.ndim != 3:
        raise ValueError(f"_unpack_latents: expected [B, S, C*4], got {tuple(latents.shape)}")
    batch_size, seq, packed_channels = latents.shape
    expected_seq = (latent_h // 2) * (latent_w // 2)
    if seq != expected_seq:
        raise ValueError(
            f"_unpack_latents: seq ({seq}) does not match "
            f"(latent_h/2)*(latent_w/2) = {expected_seq} for "
            f"latent_h={latent_h}, latent_w={latent_w}"
        )
    if packed_channels % 4 != 0:
        raise ValueError(f"_unpack_latents: packed channels ({packed_channels}) must be divisible by 4")
    channels = packed_channels // 4
    latents = latents.view(batch_size, latent_h // 2, latent_w // 2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels, latent_h, latent_w)


class QwenImageDiffusionStep(DiffusionStep[QwenImageBundle, QwenImageConditions]):
    """Per-step Qwen-Image denoising kernel — stateless.

    Extends the :class:`DiffusionStep` protocol with Qwen-Image-specific
    per-call kwargs (``latent_h`` / ``latent_w`` /
    ``distilled_guidance_scale``) on :meth:`predict_noise`,
    :meth:`step`, and :meth:`step_with_logp`. The protocol surface stays
    structurally compatible because Python protocols are non-strict on
    extra kwargs.
    """

    def predict_noise(
        self,
        model: QwenImageBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: QwenImageConditions,
        *,
        guidance_scale: float,
        latent_h: int,
        latent_w: int,
        distilled_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the Qwen-Image transformer with combined-CFG + norm correction.

        Packs ``sample`` ``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``,
        runs the transformer for the conditional branch (and, when
        ``guidance_scale > 1`` and ``conditions.negative_text`` is set,
        a second forward for the unconditional branch), applies the
        norm-corrected CFG blend, then unpacks the result back to
        ``[B, C, H, W]``.
        """
        if conditions.text is None:
            raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        prompt_embeds_mask = text.attn_mask
        if prompt_embeds is None:
            raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.text.embeds is None")
        if prompt_embeds_mask is None:
            raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.text.attn_mask is None")

        batch_size = sample.shape[0]
        device = sample.device
        dtype = prompt_embeds.dtype
        packed = _pack_latents(sample).to(dtype=dtype)

        # Qwen-Image's transformer takes raw sigma as the timestep
        # input (not sigma * 1000 like SD3).
        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size).to(device, dtype=dtype)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size).to(device, dtype=dtype)
        else:
            timestep = sigma.to(device, dtype=dtype)

        # The transformer needs the per-sample latent grid shape so it
        # can rebuild positional embeddings; format is
        # ``[[(frames, H/(vae_scale_factor*2), W/(vae_scale_factor*2))]] * B``.
        # Here ``latent_h`` / ``latent_w`` ARE already in the post-VAE
        # spatial grid, so the patchify divisor is just 2.
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]] * batch_size

        # Distilled-guidance scalar — embedded by the transformer when
        # ``guidance_embeds=True`` is set on its config (set by some
        # Qwen-Image variants only). Independent of CFG guidance_scale.
        guidance = None
        if getattr(model.transformer.config, "guidance_embeds", False):
            guidance_value = guidance_scale if distilled_guidance_scale is None else float(distilled_guidance_scale)
            guidance = torch.tensor([guidance_value], device=device, dtype=torch.float32).expand(batch_size)

        noise_pred_packed = model.transformer(
            hidden_states=packed,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
                negative_prompt_embeds_mask = neg.attn_mask
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                # Combined-CFG with norm correction. Spec: keep the per-token
                # L2 norm of the conditional prediction after CFG blending.
                comb = negative_noise_pred_packed + guidance_scale * (noise_pred_packed - negative_noise_pred_packed)
                cond_norm = torch.norm(noise_pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred_packed = comb * (cond_norm / comb_norm)

        return _unpack_latents(noise_pred_packed, latent_h=latent_h, latent_w=latent_w)

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

        Returns ``(prev_sample, log_prob, prev_sample_mean)``. Operates
        on unpacked ``[B, C, H, W]`` tensors (the strategy is shape-
        agnostic).
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
        model: QwenImageBundle,
        conditions: QwenImageConditions,
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
        latent_h: int = 0,
        latent_w: int = 0,
        distilled_guidance_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step."""
        if latent_h <= 0 or latent_w <= 0:
            # Recover from sample shape — diffuse/replay always pass both
            # explicitly, but defaulting here keeps unit tests that hand-
            # roll ``[B, C, H, W]`` simple.
            latent_h = int(sample.shape[-2])
            latent_w = int(sample.shape[-1])
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            latent_h=latent_h,
            latent_w=latent_w,
            distilled_guidance_scale=distilled_guidance_scale,
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
        model: QwenImageBundle,
        conditions: QwenImageConditions,
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
        latent_h: int = 0,
        latent_w: int = 0,
        distilled_guidance_scale: Optional[float] = None,
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
            latent_h=latent_h,
            latent_w=latent_w,
            distilled_guidance_scale=distilled_guidance_scale,
        )


class QwenImageDiffusionStage(DiffusionStage[QwenImageConditions]):
    """Qwen-Image rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel,
    and the precision policy. The kernel is stateless and is invoked
    per-step with the strategy + the per-call ``latent_h`` / ``latent_w``
    that pin the packed-latent geometry.

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
    when HF auto-discovery yields nothing — diffusers'
    ``QwenImageTransformer2DModel`` block class is
    ``QwenImageTransformerBlock``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("QwenImageTransformerBlock",)

    # Qwen-Image t2i uses an 8× VAE downsample and 16 latent channels
    # in the post-VAE grid. The model bundle's ``transformer.config``
    # carries the authoritative count via ``in_channels // 4`` (the
    # packed-latent format multiplies by 4); we default to that and let
    # callers override via the stage constructor.
    DEFAULT_VAE_SCALE_FACTOR: ClassVar[int] = 8

    def __init__(
        self,
        *,
        model: QwenImageBundle,
        step: QwenImageDiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 8,
        latent_channels: Optional[int] = None,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = vae_scale_factor
        if latent_channels is None:
            # Read from the transformer config: in_channels is the
            # packed-input dim (C * 4), so the post-VAE channel count is
            # in_channels // 4. Falls back to 16 if the attr is missing.
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 64) if tx_cfg is not None else 64
            latent_channels = int(in_channels) // 4
        self.latent_channels = int(latent_channels)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: QwenImageConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full Qwen-Image sampling. Returns a ``LatentSegment``.

        The SDE loop and the segment store stay in spatial ``[B, C, H, W]``
        shape; :meth:`QwenImageDiffusionStep.predict_noise` packs /
        unpacks at the transformer boundary so the VAE decode stage can
        read ``segment.latents[:, -1]`` without per-shape handling.

        ``initial_latents`` (optional) — driver-shipped x_T per
        ``req.request_conditions['initial_latents']``; see
        :class:`SD3DiffusionStage.diffuse` for the contract.
        """
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("QwenImageDiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"QwenImageDiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        # Latent grid follows the diffusers QwenImagePipeline convention:
        # latent_h = 2 * (H // (vae_scale_factor * 2)). The doubled-mod
        # rounding makes sure (latent_h % 2 == 0), which the 2×2 patch
        # pack requires.
        latent_h = 2 * (int(params.height) // (int(self.vae_scale_factor) * 2))
        latent_w = 2 * (int(params.width) // (int(self.vae_scale_factor) * 2))
        expected_latent_shape = (int(self.latent_channels), latent_h, latent_w)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"QwenImageDiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"QwenImageDiffusionStage.diffuse: initial_latents.shape[1:]="
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
                    latent_h=latent_h,
                    latent_w=latent_w,
                    distilled_guidance_scale=params.distilled_guidance_scale,
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if (i + 1) in needed:
                stored_pairs.append((i + 1, latents.detach().clone()))

            if log_prob is not None:
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)  # [B, K, C, H, W]

        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None
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
        conditions: QwenImageConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Recovers ``latent_h`` / ``latent_w`` from the stored segment's
        latent shape (``[B, K, C, H, W]``) so the kernel can rebuild the
        packed-latent geometry per step.

        Caller is responsible for ``.train()`` mode + grad scope; this
        method only manages the autocast scope.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("QwenImageDiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("QwenImageDiffusionStage.replay: segment.sigmas missing")
        if segment.latents.ndim != 5:
            raise ValueError(
                f"QwenImageDiffusionStage.replay: expected latents [B, K, C, H, W], got {tuple(segment.latents.shape)}"
            )
        latent_h = int(segment.latents.shape[-2])
        latent_w = int(segment.latents.shape[-1])

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"QwenImageDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
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
                    latent_h=latent_h,
                    latent_w=latent_w,
                    distilled_guidance_scale=params.distilled_guidance_scale,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"QwenImageDiffusionStage.replay: strategy returned None "
                        f"log-prob at step_index={step_idx} (deterministic mode); "
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
        conditions: QwenImageConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Latent ``H × W`` are taken directly from ``sample.shape[-2:]`` (no
        VAE round-trip). ``distilled_guidance_scale`` is read from
        ``params`` when set (Qwen-Image checkpoints with
        ``guidance_embeds=True`` use it; otherwise it's ignored).
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
            latent_h=int(sample.shape[-2]),
            latent_w=int(sample.shape[-1]),
            distilled_guidance_scale=getattr(params, "distilled_guidance_scale", None),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on.

        For Qwen-Image, that's the bundle's transformer
        (``QwenImageTransformer2DModel``) — the FSDP wrap target. Aux
        modules (VAE, text encoder) are siblings on the bundle, never
        under the transformer.
        """
        return self.model.transformer


__all__ = [
    "QwenImageDiffusionStage",
    "QwenImageDiffusionStep",
    "_pack_latents",
    "_unpack_latents",
]
