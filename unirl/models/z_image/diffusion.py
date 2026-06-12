"""Z-Image diffusion: per-step kernel + rollout-level stage.

Two classes mirror :mod:`unirl.models.sd3.diffusion`:

- :class:`ZImageDiffusionStep` — stateless per-step kernel. Wraps
  :meth:`predict_noise` (which adapts the single-stream
  ``ZImageTransformer2DModel``'s list-based forward to the framework's
  batched ``[B, C, H, W]`` SDE math) around ``StepStrategy.denoise``. The
  protocol-matching ``forward`` / ``step`` / ``step_with_logp`` ride on
  top.
- :class:`ZImageDiffusionStage` — implements
  ``DiffusionStage[ZImageConditions]``. Owns the SDE strategy and loop
  bookkeeping; segment latents stay in spatial ``[B, C, H, W]`` shape so
  :class:`ZImageVAEDecodeStage` can read them directly.

Transformer adapter
-------------------
Z-Image's S3-DiT consumes **lists**: a list of per-sample latents
``[C, F=1, H, W]`` and a list of per-sample caption embeddings
``[t_i, D]`` (variable length). It returns a list of per-sample velocity
predictions ``[C, F=1, H, W]``. :meth:`predict_noise`:

1. lifts the batched latent ``[B, C, H, W]`` → list of ``[C, 1, H, W]``;
2. rebuilds the per-prompt caption list from the padded
   ``conditions.text.embeds`` + ``attn_mask``;
3. passes ``t = 1 - sigma`` as the timestep (the diffusers reference
   feeds ``(1000 - sigma*1000)/1000``);
4. **negates** the model output (the reference does ``noise_pred =
   -model_out`` before the scheduler step) so the result is the
   FlowMatch velocity ``FlowSDEStrategy`` expects;
5. stacks the list back to ``[B, C, H, W]``.

CFG math
--------
Z-Image's CFG is ``pred = pos + scale * (pos - neg)`` (gated on
``guidance_scale > 0``), batched as a single ``[pos; neg]`` forward.
Z-Image-Turbo is distilled to run **without** CFG (``guidance_scale = 0``),
which is the RL-friendly setting; the CFG branch supports the undistilled
base model.

Math mirrors diffusers ``ZImagePipeline.__call__`` denoising loop.
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

from .bundle import ZImageBundle
from .conditions import ZImageConditions


def _caption_list(text, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """Rebuild the per-prompt variable-length caption list from a padded
    ``TextEmbedCondition`` (``embeds [B, T, D]`` + ``attn_mask [B, T]``).

    Dedicated-engine replay can hand conditions back on CPU; pin both the
    embeds and mask to the transformer's device before splitting.
    """
    if text is None or text.embeds is None:
        raise ValueError("ZImage predict_noise: conditions text/embeds is None")
    embeds = text.embeds.to(device=device, dtype=dtype)
    mask = text.attn_mask
    bsz = int(embeds.shape[0])
    if mask is None:
        return [embeds[i] for i in range(bsz)]
    bool_mask = mask.to(device).bool()
    return [embeds[i][bool_mask[i]] for i in range(bsz)]


class ZImageDiffusionStep(DiffusionStep[ZImageBundle, ZImageConditions]):
    """Per-step Z-Image denoising kernel — stateless."""

    def predict_noise(
        self,
        model: ZImageBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: ZImageConditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the single-stream Z-Image transformer and return the
        FlowMatch velocity ``[B, C, H, W]`` (negated model output, with CFG
        applied when ``guidance_scale > 0`` and a negative is present)."""
        if conditions.text is None:
            raise ValueError("ZImageDiffusionStep.predict_noise: conditions.text is None")

        dev = model.device
        sample = sample.to(dev)
        sigma = sigma.to(dev)
        try:
            model_dtype = next(model.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = sample.dtype
        sample = sample.to(dtype=model_dtype)

        batch_size = int(sample.shape[0])
        # Z-Image timestep input: (1000 - sigma*1000)/1000 == 1 - sigma.
        if sigma.dim() == 0:
            timestep = (1.0 - sigma).expand(batch_size)
        elif sigma.shape[0] != batch_size:
            timestep = (1.0 - sigma).expand(batch_size)
        else:
            timestep = 1.0 - sigma
        timestep = timestep.to(device=dev, dtype=torch.float32)

        cap_list = _caption_list(conditions.text, model_dtype, dev)

        # Lift batched latent [B, C, H, W] -> list of [C, 1, H, W].
        x_5d = sample.unsqueeze(2)  # [B, C, 1, H, W]

        use_cfg = guidance_scale > 0.0 and conditions.negative_text is not None
        if use_cfg:
            neg_list = _caption_list(conditions.negative_text, model_dtype, dev)
            x_list = list(torch.cat([x_5d, x_5d], dim=0).unbind(dim=0))
            cap_all = cap_list + neg_list
            timestep_all = timestep.repeat(2)
            out_list = model.transformer(x_list, timestep_all, cap_all, return_dict=False)[0]
            pos = torch.stack(out_list[:batch_size], dim=0)
            neg = torch.stack(out_list[batch_size:], dim=0)
            combined = pos + guidance_scale * (pos - neg)
            noise_pred = -combined
        else:
            x_list = list(x_5d.unbind(dim=0))
            out_list = model.transformer(x_list, timestep, cap_list, return_dict=False)[0]
            noise_pred = -torch.stack(out_list, dim=0)

        # Drop the temporal dim (Z-Image t2i uses F=1).
        return noise_pred.squeeze(2)

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
        model: ZImageBundle,
        conditions: ZImageConditions,
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
        """Run model forward + SDE transition. End-to-end one diffusion step."""
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
        model: ZImageBundle,
        conditions: ZImageConditions,
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


class ZImageDiffusionStage(DiffusionStage[ZImageConditions]):
    """Z-Image rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel, and
    the precision policy. The kernel is stateless and invoked per-step.

    Segment latents stay in spatial ``[B, C, H, W]`` shape (Z-Image's VAE
    is the standard 2D ``AutoencoderKL``), so :class:`ZImageVAEDecodeStage`
    reads ``segment.latents[:, -1]`` without per-shape handling.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy when
    HF auto-discovery yields nothing — diffusers'
    ``ZImageTransformer2DModel`` block class is ``ZImageTransformerBlock``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("ZImageTransformerBlock",)

    def __init__(
        self,
        *,
        model: ZImageBundle,
        step: ZImageDiffusionStep,
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

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def diffuse(
        self,
        conditions: ZImageConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full Z-Image sampling. Returns a ``LatentSegment``.

        ``initial_latents`` (optional) — driver-shipped x_T per
        ``req.request_conditions['initial_latents']``; see
        :class:`SD3DiffusionStage.diffuse` for the contract.
        """
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("ZImageDiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"ZImageDiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        # Latent grid follows the diffusers ZImagePipeline convention:
        # latent_h = 2 * (H // (vae_scale_factor * 2)). Equals H // vae_scale_factor
        # when H is a multiple of vae_scale_factor*2 (the pipeline enforces this).
        vsf = int(self.vae_scale_factor)
        latent_h = 2 * (int(params.height) // (vsf * 2))
        latent_w = 2 * (int(params.width) // (vsf * 2))
        expected_latent_shape = (int(self.latent_channels), latent_h, latent_w)
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"ZImageDiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"ZImageDiffusionStage.diffuse: initial_latents.shape[1:]="
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
        conditions: ZImageConditions,
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
            raise ValueError("ZImageDiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("ZImageDiffusionStage.replay: segment.sigmas missing")

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"ZImageDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

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
                        f"ZImageDiffusionStage.replay: strategy returned None log-prob "
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
        conditions: ZImageConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        Delegates to ``ZImageDiffusionStep.predict_noise`` so CFG batching
        and guidance handling stay identical to the sampling path.
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
        """Return the module the diffusion forward operates on — the
        bundle's ``ZImageTransformer2DModel`` (the FSDP wrap target)."""
        return self.model.transformer


__all__ = ["ZImageDiffusionStage", "ZImageDiffusionStep"]
