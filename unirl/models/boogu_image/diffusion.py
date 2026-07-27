"""Boogu-Image diffusion: per-step kernel + rollout-level stage.

Two classes mirror :mod:`unirl.models.z_image.diffusion`:

- :class:`BooguImageDiffusionStep` — stateless per-step kernel. Wraps
  :meth:`predict_noise` (which adapts the vendored
  ``BooguImageTransformer2DModel`` forward to the framework's batched
  ``[B, C, H, W]`` SDE math) around ``StepStrategy.denoise``. The
  protocol-matching ``forward`` / ``step`` / ``step_with_logp`` ride on top.
- :class:`BooguImageDiffusionStage` — implements
  ``DiffusionStage[BooguImageConditions]``. Owns the SDE strategy, the loop
  bookkeeping, and the cached rotary tables; segment latents stay in spatial
  ``[B, C, H, W]`` so :class:`BooguImageVAEDecodeStage` reads them directly.

Transformer adapter
-------------------
The vendored DiT consumes batched 4D latents directly (it lifts to a
per-sample list internally for variable-length packing). :meth:`predict_noise`:

1. passes ``t = 1 - sigma`` as the timestep in **model dtype** (the reference
   ``predict`` does ``t.expand(B).to(latents.dtype)``; the transformer scales
   by ``timestep_scale=1000`` internally);
2. forwards positionally per the reference
   (``transformer(latents, timestep, instruction_embeds, freqs_cis,
   instruction_attention_mask, ref_image_hidden_states=None)``) — the
   default ``return_dict=False`` returns a **bare tensor**, not a tuple
   (indexing ``[0]`` would take sample 0 of the batch);
3. applies CFG with Boogu's convention — gate ``guidance_scale > 1.0``
   (1.0 == off), a **second sequential forward** on the negative branch
   (the reference runs branches sequentially; positive/negative embeds are
   padded to different lengths), plain linear combine
   ``out + (g - 1) * (out - neg_out)`` with no norm correction;
4. **negates** the result: Boogu integrates ``x += (t_next - t)·v`` in the
   t-convention (t = 1 - σ, toward data), so the σ-convention FlowMatch
   velocity ``FlowSDEStrategy`` expects is ``-v`` (z_image precedent).

Rotary tables (``freqs_cis``) are the reference pipeline's per-call input:
built once from ``(axes_dim_rope, axes_lens, theta=10000)`` — resolution
independent — and cached per device on the stage.

CFG range
---------
The reference gates guidance by step fraction (``cfg_range``,
pipeline_boogu.py:3367-3381; default ``(0, 1)`` == always on). The stage
collapses that into a per-step *effective* guidance scale
(:meth:`BooguImageDiffusionStage._effective_guidance_scale`); scale 1.0 makes
the kernel skip the negative forward exactly like the reference's skipped
branch. ``cfg_range`` rides in ``DiffusionSamplingParams.sampler_kwargs``.

Math mirrors the reference ``BooguImagePipeline.processing`` T2I denoising
loop (pipeline_boogu.py:3243-3688).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ClassVar, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import BooguImageBundle
from .conditions import BooguImageConditions
from .vendor.rope import BooguImageRotaryPosEmbed


def build_freqs_cis(transformer_config, device: torch.device) -> List[torch.Tensor]:
    """Build the reference pipeline's rotary tables for the vendored DiT.

    Depends only on ``(axes_dim_rope, axes_lens, theta)`` — resolution
    independent; the per-(H, W) position gather happens inside the
    transformer's ``rope_embedder``. Mirrors the reference ``__call__``
    (pipeline_boogu.py:2896-2900).
    """
    tables = BooguImageRotaryPosEmbed.get_freqs_cis(
        list(transformer_config.axes_dim_rope),
        list(transformer_config.axes_lens),
        theta=10000,
    )
    return [t.to(device) for t in tables]


class BooguImageDiffusionStep(DiffusionStep[BooguImageBundle, BooguImageConditions]):
    """Per-step Boogu-Image denoising kernel — stateless."""

    def predict_noise(
        self,
        model: BooguImageBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: BooguImageConditions,
        *,
        guidance_scale: float,
        freqs_cis: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run the vendored DiT and return the FlowMatch velocity
        ``[B, C, H, W]`` (negated model output, with Boogu's text CFG applied
        when ``guidance_scale > 1.0``)."""
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("BooguImageDiffusionStep.predict_noise: conditions.text is None")

        dev = model.device
        try:
            model_dtype = next(model.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = sample.dtype
        sample = sample.to(device=dev, dtype=model_dtype)
        sigma = sigma.to(dev)

        batch_size = int(sample.shape[0])
        # Boogu timestep input: t = 1 - sigma, in MODEL dtype (the reference
        # predict() does t.expand(B).to(latents.dtype); timestep_scale=1000
        # is applied inside the transformer).
        if sigma.dim() == 0 or sigma.shape[0] != batch_size:
            timestep = (1.0 - sigma).expand(batch_size)
        else:
            timestep = 1.0 - sigma
        timestep = timestep.to(device=dev, dtype=model_dtype)

        if freqs_cis is None:
            freqs_cis = build_freqs_cis(model.transformer.config, dev)

        embeds = conditions.text.embeds.to(device=dev, dtype=model_dtype)
        mask = conditions.text.attn_mask
        if mask is None:
            raise ValueError("BooguImageDiffusionStep.predict_noise: conditions.text.attn_mask is None")
        mask = mask.to(dev)

        # Bare-tensor return: the vendored forward defaults return_dict=False
        # and returns [B, C, H, W] directly — no `[0]` indexing.
        out = model.transformer(
            sample,
            timestep,
            embeds,
            freqs_cis,
            mask,
            ref_image_hidden_states=None,
        )

        # Boogu CFG gate: 1.0 == off (unlike z_image's 0.0). Sequential
        # negative forward — positive/negative embeds are padded to
        # different lengths, and the reference runs branches sequentially.
        use_cfg = guidance_scale > 1.0 and conditions.negative_text is not None
        if use_cfg:
            neg = conditions.negative_text
            if neg.embeds is None or neg.attn_mask is None:
                raise ValueError("BooguImageDiffusionStep.predict_noise: negative_text embeds/attn_mask is None")
            neg_out = model.transformer(
                sample,
                timestep,
                neg.embeds.to(device=dev, dtype=model_dtype),
                freqs_cis,
                neg.attn_mask.to(dev),
                ref_image_hidden_states=None,
            )
            # Reference combine (pipeline_boogu.py:3646-3649), plain linear:
            # model_pred + (g - 1) * (model_pred - model_pred_drop_all).
            out = out + (guidance_scale - 1.0) * (out - neg_out)
        elif guidance_scale > 1.0:
            raise ValueError(
                "BooguImageDiffusionStep.predict_noise: guidance_scale > 1.0 "
                "but conditions.negative_text is None — the pipeline should "
                "have built ''-negatives"
            )

        # t-convention velocity -> sigma-convention FlowMatch velocity.
        return -out

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
        model: BooguImageBundle,
        conditions: BooguImageConditions,
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
        freqs_cis: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step."""
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            freqs_cis=freqs_cis,
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
        model: BooguImageBundle,
        conditions: BooguImageConditions,
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
        freqs_cis: Optional[List[torch.Tensor]] = None,
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
            freqs_cis=freqs_cis,
        )


class BooguImageDiffusionStage(DiffusionStage[BooguImageConditions]):
    """Boogu-Image rollout-level diffusion stage.

    Owns the SDE ``strategy``, the bundle, the stateless kernel, the
    precision policy, and the per-device rotary-table cache.

    Segment latents stay in spatial ``[B, C, H, W]`` (standard 2D FLUX
    ``AutoencoderKL``), so :class:`BooguImageVAEDecodeStage` reads
    ``segment.latents[:, -1]`` without per-shape handling.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy when
    HF auto-discovery yields nothing. NOTE: FSDP wrap matching is by exact
    concrete class name — recipes must list the five INSTANTIATED block
    classes (the bare ``BooguImageTransformerBlock`` is never instantiated,
    and ``BooguImageDoubleStreamTransformerBlock`` is a separate class, not
    a subclass).
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = (
        "BooguImageTransformerBlock",
        "BooguImageNoiseRefinerTransformerBlock",
        "BooguImageRefImgRefinerTransformerBlock",
        "BooguImageContextRefinerTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
    )

    def __init__(
        self,
        *,
        model: BooguImageBundle,
        step: BooguImageDiffusionStep,
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
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 16) if tx_cfg is not None else 16
            latent_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self._freqs_cis: Optional[List[torch.Tensor]] = None
        self._freqs_cis_device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_freqs_cis(self, device: torch.device) -> List[torch.Tensor]:
        """Per-device cache of the resolution-independent rotary tables."""
        if self._freqs_cis is None or self._freqs_cis_device != device:
            self._freqs_cis = build_freqs_cis(self.model.transformer.config, device)
            self._freqs_cis_device = device
        return self._freqs_cis

    @staticmethod
    def _effective_guidance_scale(step_index: int, num_steps: int, params: DiffusionSamplingParams) -> float:
        """Collapse the reference's ``cfg_range`` step-fraction gate into a
        per-step scale (pipeline_boogu.py:3367-3381, inclusive bounds; the
        out-of-range value is 1.0 == CFG off in Boogu's convention).

        ``cfg_range`` rides in ``params.sampler_kwargs``; the default
        ``(0.0, 1.0)`` keeps guidance on at every step. Deterministic in
        ``(step_index, num_steps)``, so replay reproduces rollout exactly.
        """
        lo, hi = params.sampler_kwargs.get("cfg_range", (0.0, 1.0))
        fraction = step_index / num_steps if num_steps > 0 else 0.0
        if float(lo) <= fraction <= float(hi):
            return float(params.guidance_scale)
        return 1.0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: BooguImageConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full Boogu-Image sampling. Returns a ``LatentSegment``.

        ``initial_latents`` (optional) — x_T resolved from the request
        ``Sample``'s diffusion generation Part.
        """
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("BooguImageDiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"BooguImageDiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        # Latent grid: 2 * (H // (vae_scale_factor * 2)) — equals the
        # reference's H // 8 after its ×16 pixel-size floor
        # (pipeline_boogu.py:2994-2997, 863-867) and guarantees the patch-2
        # divisibility flat_and_pad_to_seq needs.
        vsf = int(self.vae_scale_factor)
        latent_h = 2 * (int(params.height) // (vsf * 2))
        latent_w = 2 * (int(params.width) // (vsf * 2))
        expected_latent_shape = (int(self.latent_channels), latent_h, latent_w)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"BooguImageDiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"BooguImageDiffusionStage.diffuse: initial_latents.shape[1:]="
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

        freqs_cis = self._get_freqs_cis(device)

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
                    guidance_scale=self._effective_guidance_scale(i, T, params),
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=i,
                    freqs_cis=freqs_cis,
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
        conditions: BooguImageConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Caller is responsible for ``.train()`` mode + grad scope; this method
        only manages the autocast scope.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("BooguImageDiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("BooguImageDiffusionStage.replay: segment.sigmas missing")

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"BooguImageDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = torch.device(self.model.device)
        sigmas = segment.sigmas.to(device)
        # The cfg_range fraction denominator is the rollout's step count,
        # recovered from the stored schedule (T + 1 sigmas).
        num_steps = int(sigmas.shape[0]) - 1
        sigma_max = sigmas[1].float() if int(sigmas.shape[0]) > 1 else torch.tensor(0.99)

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        freqs_cis = self._get_freqs_cis(device)
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
                    guidance_scale=self._effective_guidance_scale(step_idx, num_steps, params),
                    eta=float(params.eta),
                    sigma_max=sigma_max,
                    step_index=step_idx,
                    freqs_cis=freqs_cis,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"BooguImageDiffusionStage.replay: strategy returned None log-prob "
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
        conditions: BooguImageConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Delegates to ``BooguImageDiffusionStep.predict_noise`` so CFG and
        guidance handling stay identical to the sampling path. Forward-process
        algorithms carry no loop index, so the ``cfg_range`` fraction gate is
        not applied here (the default ``(0, 1)`` makes this moot).
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
            freqs_cis=self._get_freqs_cis(torch.device(self.model.device)),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on — the
        bundle's vendored ``BooguImageTransformer2DModel`` (the FSDP wrap
        target)."""
        return self.model.transformer


__all__ = ["BooguImageDiffusionStage", "BooguImageDiffusionStep", "build_freqs_cis"]
