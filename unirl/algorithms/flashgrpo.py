"""FlashGRPO: FlowGRPO with Flash-GRPO temporal-gradient rectification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import numpy as np
import torch

from unirl.sde.index_schedule import AllSDEScheduler
from unirl.sde.kernels import FlashSDEStrategy
from unirl.train.lora import adapters_disabled
from unirl.types.conditions import Condition
from unirl.types.sample import Part
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    gaussian_kl_div,
    grpo_clip_loss,
    resolve_clip_range_from_schedule,
    transition_sigma,
    typed_conditions,
)
from .flowgrpo import FlowGRPO

if TYPE_CHECKING:
    from unirl.models.types.replay_result import ReplayResult


@dataclass
class FlashGRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    clip_range: float = 1e-3
    clip_schedule: str = "constant"
    beta: float = 0.0
    old_logp_source: str = "rollout"
    adv_clip_max: float = 5.0
    params: Any = dc_field(default=None)
    rectification_indices: List[int] = dc_field(default_factory=list)
    batch_normalize_tgr: bool = False
    engine_per_sample_sde: bool = False


class FlashGRPO(FlowGRPO):
    """FlowGRPO with TGR-weighted policy loss and optional unrectified reference KL."""

    requires_per_sample_sde_index = True  # One stratified SDE step per prompt; see trainer/README.md.

    def __init__(
        self,
        *,
        params: Any,
        rectification_indices: Sequence[int] = (),
        batch_normalize_tgr: bool = FlashGRPOConfig.batch_normalize_tgr,
        engine_per_sample_sde: bool = FlashGRPOConfig.engine_per_sample_sde,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-3,
        clip_schedule: str = "constant",
        beta: float = 0.0,
        old_logp_source: str = "rollout",
        adv_clip_max: float = 5.0,
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__(
            params=params,
            stage=stage,
            pipeline=pipeline,
            stage_attr=stage_attr,
            clip_range=clip_range,
            clip_schedule=clip_schedule,
            beta=beta,
            old_logp_source=old_logp_source,
            backend=backend,
            conditions_cls=conditions_cls,
        )
        self.batch_normalize_tgr = bool(batch_normalize_tgr)
        if engine_per_sample_sde and not getattr(self.stage, "supports_order_independent_sde_noise", False):
            raise ValueError(
                "FlashGRPO engine_per_sample_sde requires sample/step-keyed SDE noise; "
                f"{type(self.stage).__name__} does not declare it (Wan uses global RNG). "
                "Disable engine_per_sample_sde or use a stage with request-local generators."
            )
        self.backend = backend
        self._batch_tgr_metrics: Dict[str, float] = {}
        if self.batch_normalize_tgr:
            if not rectification_indices:
                rectification_indices = params.scheduler.sde_candidate_pool()
            self.supports_multi_update = False
            if backend is None:
                raise ValueError("FlashGRPO batch_normalize_tgr requires the training backend, including on one rank.")
        self.rectification_indices = [int(i) for i in rectification_indices]
        self.adv_clip_max = float(adv_clip_max)
        if not self.adv_clip_max > 0.0:
            raise ValueError(f"FlashGRPO: adv_clip_max must be > 0; got {adv_clip_max!r}.")
        # Every rollout segment carries sde_index_per_sample; a replay that ignores it scores all rows at one sigma.
        if not getattr(self.stage, "supports_per_sample_sde_index", False):
            raise ValueError(
                f"FlashGRPO requires a stage that honours segment.sde_index_per_sample; "
                f"{type(self.stage).__name__} does not. Use FlowGRPO, or implement the "
                "per-sample replay on that stage and declare supports_per_sample_sde_index."
            )
        # _rectification_coefficients requires FlashSDEStrategy.transition_std.
        strategy = self.stage.strategy
        if not isinstance(strategy, FlashSDEStrategy):
            raise ValueError(
                f"FlashGRPO requires the stage strategy to be FlashSDEStrategy; got {type(strategy).__name__}."
            )

    @staticmethod
    def engine_groups_sde_steps(
        algorithm_cfg: Mapping[str, Any], *, num_updates_per_batch: int, accumulate_rollouts: int
    ) -> bool:
        """Reject per-sample rollout config before model loading; return ``engine_per_sample_sde``."""
        engine_groups = algorithm_cfg.get("engine_per_sample_sde", FlashGRPOConfig.engine_per_sample_sde)
        batch_normalize_tgr = algorithm_cfg.get("batch_normalize_tgr", FlashGRPOConfig.batch_normalize_tgr)
        if engine_groups and not batch_normalize_tgr:
            raise ValueError("engine_per_sample_sde requires FlashGRPO with algorithm.batch_normalize_tgr=true.")
        if batch_normalize_tgr and (num_updates_per_batch != 1 or accumulate_rollouts != 1):
            raise ValueError(
                "FlashGRPO batch_normalize_tgr normalizes one rollout/update; set "
                "stack.num_updates_per_batch=1 and accumulate_rollouts=1."
            )
        return engine_groups

    @staticmethod
    def per_sample_sde_pool(algorithm_cfg: Mapping[str, Any], diffusion: Any) -> List[int]:
        """Validate the scheduler's per-prompt SDE candidate pool against ``rectification_indices``; return it."""
        scheduler = diffusion.scheduler
        if not isinstance(scheduler, AllSDEScheduler):
            raise ValueError(
                "FlashGRPO per-sample SDE-index path requires sampling_params['diffusion'] to carry "
                f"an AllSDEScheduler; got scheduler={type(scheduler).__name__ if scheduler is not None else None}."
            )
        pool = scheduler.sde_candidate_pool()
        if not pool:
            raise ValueError("FlashGRPO per-sample SDE-index path: the scheduler candidate pool is empty.")
        # The last denoising step stores two latents, not three, so its group cannot be concatenated.
        num_inference_steps = diffusion.num_inference_steps
        if max(pool) >= num_inference_steps - 1:
            raise ValueError(
                f"FlashGRPO per-sample SDE candidate pool reaches step {max(pool)}, but the last denoising "
                f"step (num_inference_steps-1 = {num_inference_steps - 1}) stores a different number of "
                "trajectory latents, so mixed-shape rollout groups cannot be concatenated. Narrow the "
                "scheduler timestep_fraction so its end maps below the final step."
            )
        sorted_pool = sorted(pool)
        if algorithm_cfg.get("batch_normalize_tgr", FlashGRPOConfig.batch_normalize_tgr) and not algorithm_cfg.get(
            "rectification_indices"
        ):
            return pool
        if "rectification_indices" not in algorithm_cfg or algorithm_cfg["rectification_indices"] is None:
            raise ValueError(
                "FlashGRPO per-sample path requires algorithm.rectification_indices set to the SDE "
                f"candidate pool {sorted_pool}; got None."
            )
        rectification_sorted = sorted(int(i) for i in algorithm_cfg["rectification_indices"])
        if rectification_sorted != sorted_pool:
            raise ValueError(
                f"FlashGRPO rectification_indices {rectification_sorted} must equal the SDE candidate "
                f"pool {sorted_pool}: each sample draws its step from the pool but is normalized over "
                "rectification_indices, so a mismatch silently mis-scales the loss."
            )
        return pool

    @staticmethod
    def per_prompt_sde_steps(pool: Sequence[int], *, num_prompts: int, rollout_id: int) -> List[int]:
        """One stratified step per prompt: a rollout-seeded pool permutation cycled by prompt position."""
        pool_size = len(pool)
        rng = np.random.default_rng(rollout_id)
        # Keep the permutation and position modulo identical in both dispatch modes.
        order = rng.permutation(pool_size)
        steps = [pool[int(s)] for s in order]
        return [steps[position % pool_size] for position in range(num_prompts)]

    def prepare_part(self, part: Part) -> Part:
        """Attach detached [B, 1] weights before micro slicing; see algorithms/README.md Gotchas."""
        if not self.batch_normalize_tgr:
            return part
        pool = self.rectification_indices
        statistics = [0.0] * (3 + len(pool))
        error = None
        try:
            segment = part.segment
            if segment is None or segment.sigmas is None or segment.sde_index_per_sample is None:
                raise ValueError("missing segment.sigmas or segment.sde_index_per_sample")
            idx = segment.sde_index_per_sample
            if idx.ndim != 1 or idx.dtype not in (torch.int32, torch.int64) or idx.numel() != part.batch_size:
                raise ValueError("sde_index_per_sample must be a 1-D integer tensor with one step per sample")
            sigmas = segment.sigmas.to(dtype=torch.float32)
            if sigmas.ndim != 1 or not torch.isfinite(sigmas).all():
                raise ValueError("sigmas must be a finite 1-D schedule")
            if sigmas.shape[0] < 2 or min(pool) < 0 or max(pool) >= sigmas.shape[0] - 1:
                raise ValueError("sigmas must cover every candidate step and its successor; fix the rollout schedule")
            device = sigmas.device
            candidates = torch.tensor(pool, device=device, dtype=torch.long)
            matches = idx.to(device=device)[:, None] == candidates[None, :]
            if not matches.any(dim=1).all():
                raise ValueError(f"each sample's SDE step must belong to candidate pool {pool}")
            self.stage.strategy.init_schedule(sigmas)
            coefficients = self._rectification_coefficients(sigmas=sigmas, steps=pool, device=device).detach()
            if not (torch.isfinite(coefficients) & (coefficients > 0)).all():
                raise ValueError("TGR coefficients must be finite and positive; check sigmas and eta")
            weights = coefficients[matches.long().argmax(dim=1)]
            statistics = [weights.double().sum().item(), float(weights.numel()), 0.0, *matches.sum(dim=0).tolist()]
        except ValueError as exc:
            error = exc
            statistics[2] = 1.0
        total, count, invalid, *counts = self.backend.all_reduce_loss_sums(statistics)
        if invalid:
            raise ValueError(
                "FlashGRPO batch_normalize_tgr: invalid input on a training rank; "
                f"{error or 'check the failing rank for its schedule/step error'}. "
                "Fix the rollout schedule and step pool."
            ) from error
        if count <= 0 or not math.isfinite(total / count) or total <= 0:
            raise ValueError("FlashGRPO batch_normalize_tgr: global coefficient sum/count must be finite and positive.")
        norm = (
            coefficients.detach().mean().clamp_min(torch.finfo(torch.float32).eps)
            if len(set(counts)) == 1
            else coefficients.new_tensor(total / count)
        )
        pool_weights = coefficients / norm
        if not (torch.isfinite(pool_weights) & (pool_weights > 0)).all():
            raise ValueError(
                "FlashGRPO batch_normalize_tgr: normalized weights must be finite and positive; check sigmas."
            )
        segment.policy_loss_weights = (weights / norm).reshape(-1, 1).detach()
        global_counts = torch.tensor(counts, dtype=torch.float64, device=device)
        active = pool_weights[global_counts > 0]
        self._batch_tgr_metrics = {
            "flash_tgr_global_mean": (pool_weights.double() * global_counts).sum().item() / count,
            "flash_tgr_global_min": active.min().item(),
            "flash_tgr_global_max": active.max().item(),
        }
        return part

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        # The rollout engine stamps every per-sample request; a missing stamp means the request carried no steps.
        if segment.sde_index_per_sample is None:
            raise ValueError(
                "FlashGRPO requires segment.sde_index_per_sample; the rollout was not stratified per prompt."
            )
        if segment.sde_indices is None or segment.sde_indices.numel() == 0:
            raise ValueError("FlashGRPO requires segment.sde_indices to be non-empty.")
        target_steps = segment.sde_indices.tolist()

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs
        # Each sample took its one stochastic step at its own index, so slot 0 holds its logp.
        old_logp = segment.sde_logp[:, :1].to(dtype=new_logp.dtype, device=new_logp.device)

        clip_range = resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_raw = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device)
        adv_clipped = adv_raw.clamp(-self.adv_clip_max, self.adv_clip_max)
        adv_b = adv_clipped.reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )

        if self.batch_normalize_tgr:
            if segment.policy_loss_weights is None:
                raise ValueError("FlashGRPO batch_normalize_tgr requires prepare_part before micro-batch slicing.")
            tgr = segment.policy_loss_weights.to(device=new_logp.device)
        else:
            tgr = self._rectification_weights_per_sample(segment=segment, device=new_logp.device)
        tgr = tgr.to(dtype=loss_per_elem.dtype)
        loss_per_elem = loss_per_elem * tgr
        policy_loss = loss_per_elem.mean()
        loss = policy_loss
        if self.beta > 0.0:
            kl_ref = self._reference_kl(
                conditions=typed_conds, segment=segment, target_steps=target_steps, replay=replay_result
            )
            kl_loss = self.beta * kl_ref
            loss = loss + kl_loss

        (loss * loss_scale).backward()
        metric_tensors = {
            "policy_loss": policy_loss.detach(),
            "flash_tgr_mean": tgr.detach().mean(),
            "flash_tgr_min": tgr.detach().min(),
            "flash_tgr_max": tgr.detach().max(),
            "adv_clip_fraction": (adv_raw.abs() > self.adv_clip_max).float().mean(),
            "adv_abs_max": adv_raw.abs().max(),
            **ratio_metrics,
        }
        if self.beta > 0.0:
            metric_tensors.update(kl_ref_mean=kl_ref.detach(), kl_loss=kl_loss.detach(), total_loss=loss.detach())
        metric_values = torch.stack([value.float() for value in metric_tensors.values()]).cpu().tolist()
        metrics: Dict[str, Any] = dict(zip(metric_tensors, metric_values))
        metrics["clip_range"] = clip_range
        if self.beta > 0.0:
            metrics["beta"] = self.beta
        if self.batch_normalize_tgr:
            metrics.update(self._batch_tgr_metrics)

        return AlgorithmStepResult(
            loss=metrics["total_loss"] if self.beta > 0.0 else metrics["policy_loss"],
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _reference_kl(
        self, *, conditions: Any, segment: LatentSegment, target_steps: List[int], replay: "ReplayResult"
    ) -> torch.Tensor:
        """Mean per-row dual-stream KL to the LoRA-disabled reference; see algorithms/README.md Gotchas."""
        new_means = replay.prev_sample_means
        if new_means is None:
            raise RuntimeError(
                "FlashGRPO: beta>0 requires stage.replay() to return prev_sample_means, "
                "but got None. Ensure the stage's replay method produces means."
            )
        with torch.no_grad(), adapters_disabled(self.ref_model):
            result = self.stage.replay(conditions, segment=segment, params=self.params, step_indices=target_steps)
        if result.prev_sample_means is None:
            raise RuntimeError(
                "FlashGRPO: stage.replay() returned prev_sample_means=None for the adapter-disabled reference forward."
            )
        ref_means = result.prev_sample_means.detach().to(dtype=new_means.dtype, device=new_means.device)
        if replay.video_transition_stds is not None:
            sigma_t = replay.video_transition_stds.detach()
        else:
            sigma_t = transition_sigma(
                self.stage,
                segment=segment,
                target_steps=segment.sde_index_per_sample,
                eta=self.params.eta,
                device=new_means.device,
                add_coefficient=True,
            ).reshape(-1, 1)  # [B, 1]
        kl_per_elem = gaussian_kl_div(new_means, ref_means, sigma_t)
        kl_per_step = kl_per_elem.mean(dim=tuple(range(2, kl_per_elem.ndim)))
        if replay.audio_in_policy:
            audio_means = replay.audio_prev_sample_means
            if audio_means is None or result.audio_prev_sample_means is None or replay.audio_transition_stds is None:
                raise RuntimeError(
                    "FlashGRPO: joint reference KL requires audio policy/reference means and transition stds."
                )
            ref_audio = result.audio_prev_sample_means.detach().to(audio_means)
            audio_kl = gaussian_kl_div(audio_means, ref_audio, replay.audio_transition_stds.detach())
            n_video, n_audio = new_means[0, 0].numel(), audio_means[0, 0].numel()
            kl_per_step = kl_per_step * (n_video / (n_video + n_audio)) + audio_kl.mean(
                dim=tuple(range(2, audio_kl.ndim))
            ) * (n_audio / (n_video + n_audio))
        return kl_per_step.mean()

    def _rectification_weights_per_sample(
        self,
        *,
        segment: LatentSegment,
        device: torch.device,
    ) -> torch.Tensor:
        """Per-sample rectification weights ``[N, 1]``, normalized over the required candidate pool."""
        if segment.sigmas is None:
            raise ValueError("FlashGRPO requires segment.sigmas to compute temporal rectification weights.")
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        weights = self._rectification_coefficients(sigmas=sigmas, steps=segment.sde_index_per_sample, device=device)
        norm = self._rectification_coefficients(sigmas=sigmas, steps=self.rectification_indices, device=device)
        weights = weights / norm.detach().mean().clamp_min(torch.finfo(torch.float32).eps)
        return weights.reshape(-1, 1)

    def _rectification_coefficients(
        self,
        *,
        sigmas: torch.Tensor,
        steps: Union[Sequence[int], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Reciprocal of the upstream per-step temporal-gradient denominator at ``steps``; a tensor stays on device."""
        if sigmas.shape[0] < 2:
            raise ValueError(f"FlashGRPO rectification requires sigmas length >= 2; got {sigmas.shape[0]}.")
        idx = (
            steps.to(device=device, dtype=torch.long)
            if isinstance(steps, torch.Tensor)
            else torch.tensor(steps, dtype=torch.long, device=device)
        )
        if idx.numel() == 0:
            raise ValueError("FlashGRPO rectification requires at least one timestep index.")
        eta = self.params.eta
        if eta <= 0.0:
            raise ValueError("FlashGRPO requires params.eta > 0 on trained SDE steps.")
        sigma = sigmas[:-1].index_select(0, idx)
        sigma_next = sigmas[1:].index_select(0, idx)
        transition_std = self.stage.strategy.transition_std(
            sigma=sigma, sigma_next=sigma_next, eta=eta, sigma_max=sigmas[1]
        ).clamp_min(torch.finfo(torch.float32).eps)
        sigma_safe = sigma.clamp_min(torch.finfo(torch.float32).eps)
        denom = (sigma - sigma_next) / transition_std + transition_std * (1.0 - sigma) / (2.0 * sigma_safe)
        return denom.reciprocal()


__all__ = ["FlashGRPO", "FlashGRPOConfig"]
