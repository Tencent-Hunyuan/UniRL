"""Qwen-Image diffusion: typed params + per-step kernel + rollout-level stage."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import torch

from unirl.models.types.batched_replay import BatchedStepReplayMixin
from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import SDEStrategy, StepStrategy
from unirl.types.conditions import TextEmbedCondition
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import QwenImageBundle
from .conditions import QwenImageConditions


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``."""
    if latents.ndim != 4:
        raise ValueError(f"_pack_latents: expected [B, C, H, W], got {tuple(latents.shape)}")
    batch_size, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"_pack_latents: H ({height}) and W ({width}) must be divisible by 2")
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def _unpack_latents(latents: torch.Tensor, *, latent_h: int, latent_w: int) -> torch.Tensor:
    """``[B, S, C*4]`` → ``[B, C, H, W]`` (inverse of :func:`_pack_latents`)."""
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
    """Per-step Qwen-Image denoising kernel — stateless."""

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
        """Run the Qwen-Image transformer with combined-CFG + norm correction."""
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

        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size).to(device, dtype=dtype)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size).to(device, dtype=dtype)
        else:
            timestep = sigma.to(device, dtype=dtype)

        img_shapes = [[(1, latent_h // 2, latent_w // 2)]] * batch_size

        guidance = None
        if getattr(model.transformer.config, "guidance_embeds", False):
            guidance_value = guidance_scale if distilled_guidance_scale is None else float(distilled_guidance_scale)
            guidance = torch.tensor([guidance_value], device=device, dtype=torch.float32).expand(batch_size)

        # Drop the all-pad tail columns; diffusers takes the RoPE text width
        # from the trimmed tensor and masks the remaining pad tokens out.
        max_true = int(prompt_embeds_mask.sum(dim=1).max().item())
        if prompt_embeds.shape[1] > max_true:
            prompt_embeds = prompt_embeds[:, :max_true]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_true]

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
                if negative_prompt_embeds_mask is None:
                    raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.negative_text.attn_mask is None")
                neg_max = int(negative_prompt_embeds_mask.sum(dim=1).max().item())
                if negative_prompt_embeds.shape[1] > neg_max:
                    negative_prompt_embeds = negative_prompt_embeds[:, :neg_max]
                    negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :neg_max]
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]
                comb = negative_noise_pred_packed + guidance_scale * (noise_pred_packed - negative_noise_pred_packed)
                cond_norm = torch.norm(noise_pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred_packed = comb * (cond_norm / comb_norm)

        return _unpack_latents(noise_pred_packed, latent_h=latent_h, latent_w=latent_w)

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
        """Run model forward + SDE transition."""
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


class QwenImageDiffusionStage(BatchedStepReplayMixin, DiffusionStage[QwenImageConditions]):
    """Qwen-Image rollout-level diffusion stage."""

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("QwenImageTransformerBlock",)

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
        batch_replay_steps: bool = False,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = vae_scale_factor
        self.batch_replay_steps = bool(batch_replay_steps)
        if latent_channels is None:
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 64) if tx_cfg is not None else 64
            latent_channels = int(in_channels) // 4
        self.latent_channels = int(latent_channels)

    def diffuse(
        self,
        conditions: QwenImageConditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full Qwen-Image sampling. Returns a ``LatentSegment``."""
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
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)

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

    def replay(
        self,
        conditions: QwenImageConditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions."""
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

        if self.batch_replay_steps and len(target) > 1 and isinstance(self.strategy, SDEStrategy):
            with autocast_ctx:
                return self._replay_batched_steps(
                    conditions,
                    segment=segment,
                    params=params,
                    target=target,
                    sigmas=sigmas,
                    sigma_max=sigma_max,
                    device=device,
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

    def _batched_step_kwargs(self, segment: LatentSegment, params: DiffusionSamplingParams) -> Dict[str, Any]:
        """Supply latent geometry and distilled guidance."""
        return {
            "latent_h": int(segment.latents.shape[-2]),
            "latent_w": int(segment.latents.shape[-1]),
            "distilled_guidance_scale": params.distilled_guidance_scale,
        }

    @staticmethod
    def _tile_conditions(conditions: QwenImageConditions, repeats: int) -> QwenImageConditions:
        def _rep(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.repeat(repeats, *([1] * (t.dim() - 1))) if t is not None else None

        def _tile(cond: Optional[TextEmbedCondition]) -> Optional[TextEmbedCondition]:
            if cond is None:
                return None
            return TextEmbedCondition(
                embeds=_rep(cond.embeds), pooled=_rep(cond.pooled), attn_mask=_rep(cond.attn_mask)
            )

        return QwenImageConditions(text=_tile(conditions.text), negative_text=_tile(conditions.negative_text))

    def predict_noise_at_step(
        self,
        conditions: QwenImageConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration."""
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

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on."""
        return self.model.transformer


__all__ = [
    "QwenImageDiffusionStage",
    "QwenImageDiffusionStep",
    "_pack_latents",
    "_unpack_latents",
]
