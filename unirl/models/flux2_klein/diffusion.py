"""FLUX.2-klein diffusion — the SDE loop runs in patchified ``[B, 128, H_pix/16, W_pix/16]`` space."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Set, Tuple

import torch

from unirl.models.types.batched_replay import BatchedStepReplayMixin
from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import SDEStrategy, StepStrategy
from unirl.types.conditions import TextEmbedCondition
from unirl.types.sampling import compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Flux2KleinBundle
from .conditions import Flux2KleinConditions
from .flux2_klein_utils import (
    pack_latents,
    prepare_latent_ids,
    prepare_text_ids,
    unpack_latents,
)


@dataclass
class Flux2KleinDiffusionParams:
    """Per-request sampling knobs for FLUX.2-klein diffusion."""

    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    seed: int = 42
    sde_indices: Optional[List[int]] = None
    eta: float = 0.7
    init_same_noise: bool = False
    samples_per_prompt: int = 1
    noise_group_ids: Optional[List[str]] = None


class Flux2KleinDiffusionStep(DiffusionStep[Flux2KleinBundle, Flux2KleinConditions]):
    """Per-step Klein kernel — stateless; packs ``[B, 128, H_pat, W_pat]`` to ``[B, H_pat*W_pat, 128]`` and back."""

    def predict_noise(
        self,
        model: Flux2KleinBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Flux2KleinConditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the Klein transformer forward on the patchified latent ``[B, 128, H_pat, W_pat]``."""
        if conditions.text is None:
            raise ValueError("Flux2KleinDiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        if prompt_embeds is None:
            raise ValueError("Flux2KleinDiffusionStep.predict_noise: conditions.text.embeds is None")

        batch_size = sample.shape[0]
        device = sample.device
        dtype = prompt_embeds.dtype

        packed = pack_latents(sample).to(dtype=dtype)
        noise_seq_len = packed.shape[1]

        if sigma.dim() == 0:
            timestep = sigma.float().expand(batch_size).to(device)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.float().expand(batch_size).to(device)
        else:
            timestep = sigma.float().to(device)

        guidance = torch.zeros(batch_size, device=device, dtype=dtype)
        txt_ids = prepare_text_ids(prompt_embeds).to(device=device)
        img_ids = prepare_latent_ids(sample).to(device=device)

        cond_tokens = conditions.image_latent
        if cond_tokens is not None:
            cond_tokens = cond_tokens.to(device=device, dtype=dtype)
            packed = torch.cat([packed, cond_tokens], dim=1)
            cond_ids = conditions.image_latent_ids
            if cond_ids is None:
                raise ValueError(
                    "Flux2KleinDiffusionStep.predict_noise: conditions.image_latent set "
                    "but image_latent_ids is None; both are required for the edit path."
                )
            img_ids = torch.cat([img_ids, cond_ids.to(device=device)], dim=1)

        noise_pred_packed = model.transformer(
            hidden_states=packed,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            guidance=guidance,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        noise_pred_packed = noise_pred_packed[:, :noise_seq_len]

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
                neg_txt_ids = prepare_text_ids(negative_prompt_embeds).to(device=device)
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    guidance=guidance,
                    txt_ids=neg_txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0][:, :noise_seq_len]
                noise_pred_packed = negative_noise_pred_packed + guidance_scale * (
                    noise_pred_packed - negative_noise_pred_packed
                )

        latent_h = int(sample.shape[-2])
        latent_w = int(sample.shape[-1])
        return unpack_latents(noise_pred_packed, latent_h, latent_w)

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
        model: Flux2KleinBundle,
        conditions: Flux2KleinConditions,
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
        model: Flux2KleinBundle,
        conditions: Flux2KleinConditions,
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


class Flux2KleinDiffusionStage(BatchedStepReplayMixin, DiffusionStage[Flux2KleinConditions]):
    """FLUX.2-klein rollout-level diffusion stage."""

    _no_split_modules: ClassVar[Tuple[str, ...]] = (
        "Flux2TransformerBlock",
        "Flux2SingleTransformerBlock",
    )

    DEFAULT_VAE_SCALE_FACTOR: ClassVar[int] = 8
    DEFAULT_PATCHIFY_FACTOR: ClassVar[int] = 2

    def __init__(
        self,
        *,
        model: Flux2KleinBundle,
        step: Flux2KleinDiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 8,
        patchify_factor: int = 2,
        latent_channels: Optional[int] = None,
        batch_replay_steps: bool = False,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.vae_scale_factor = int(vae_scale_factor)
        self.patchify_factor = int(patchify_factor)
        self.batch_replay_steps = bool(batch_replay_steps)
        if latent_channels is None:
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 128) if tx_cfg is not None else 128
            latent_channels = int(in_channels)
        self.latent_channels = int(latent_channels)

    def _patchified_shape(self, height: int, width: int) -> Tuple[int, int, int]:
        """Compute the patchified ``(C, H_pat, W_pat)`` for ``(height, width)`` pixels."""
        downsample = self.vae_scale_factor * self.patchify_factor
        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(
                f"Flux2KleinDiffusionStage: height ({height}) and width ({width}) "
                f"must be divisible by VAE×patchify downsample ({downsample})."
            )
        h_pat = height // downsample
        w_pat = width // downsample
        return (self.latent_channels, h_pat, w_pat)

    def diffuse(
        self,
        conditions: Flux2KleinConditions,
        *,
        schedule: torch.Tensor,
        params: Flux2KleinDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full FLUX.2-klein sampling; the segment stores patchified ``[B, K, 128, H_pat, W_pat]``."""
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Flux2KleinDiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"Flux2KleinDiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        expected_latent_shape = self._patchified_shape(int(params.height), int(params.width))
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"Flux2KleinDiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"Flux2KleinDiffusionStage.diffuse: initial_latents.shape[1:]="
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

        self.model.transformer.eval()

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
        conditions: Flux2KleinConditions,
        *,
        segment: LatentSegment,
        params: Flux2KleinDiffusionParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions."""
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("Flux2KleinDiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("Flux2KleinDiffusionStage.replay: segment.sigmas missing")
        if segment.latents.ndim != 5:
            raise ValueError(
                f"Flux2KleinDiffusionStage.replay: expected latents [B, K, C, H_pat, W_pat], "
                f"got {tuple(segment.latents.shape)}"
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
                f"Flux2KleinDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = sigmas[1].float() if int(sigmas.shape[0]) > 1 else torch.tensor(0.99)

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        prior_training = self.model.transformer.training
        self.model.transformer.eval()
        try:
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
                    )
                    if log_prob is None:
                        raise RuntimeError(
                            f"Flux2KleinDiffusionStage.replay: strategy returned None "
                            f"log-prob at step_index={step_idx} (deterministic mode); "
                            f"replay requires a stochastic SDE strategy."
                        )
                    log_probs.append(log_prob)
                    if prev_mean is not None:
                        prev_sample_means.append(prev_mean)
        finally:
            if prior_training:
                self.model.transformer.train()

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    @staticmethod
    def _tile_conditions(conditions: Flux2KleinConditions, repeats: int) -> Flux2KleinConditions:
        def _rep(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.repeat(repeats, *([1] * (t.dim() - 1))) if t is not None else None

        def _tile(cond: Optional[TextEmbedCondition]) -> Optional[TextEmbedCondition]:
            if cond is None:
                return None
            return TextEmbedCondition(
                embeds=_rep(cond.embeds), pooled=_rep(cond.pooled), attn_mask=_rep(cond.attn_mask)
            )

        return Flux2KleinConditions(
            text=_tile(conditions.text),
            negative_text=_tile(conditions.negative_text),
            image_latent=_rep(conditions.image_latent),
            image_latent_ids=_rep(conditions.image_latent_ids),
        )

    def predict_noise_at_step(
        self,
        conditions: Flux2KleinConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: Flux2KleinDiffusionParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration."""
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
        )

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on."""
        return self.model.transformer


__all__ = [
    "Flux2KleinDiffusionParams",
    "Flux2KleinDiffusionStage",
    "Flux2KleinDiffusionStep",
]
