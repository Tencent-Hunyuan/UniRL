"""FlashGRPO: FlowGRPO with Flash-GRPO temporal-gradient rectification."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Type, Union

import torch

from unirl.sde.kernels import FlashSDEStrategy
from unirl.types.conditions import Condition
from unirl.types.sample import Part
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    grpo_clip_loss,
    reference_kl_loss,
    resolve_clip_range_from_schedule,
    typed_conditions,
)
from .flowgrpo import FlowGRPO


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

    # The trainer gates its per-sample SDE-index orchestration on this flag: stratify
    # one SDE step per prompt, stamp ``segment.sde_index_per_sample``, then merge the
    # groups into one training track. Without it every sample shares the one step
    # ``get_sde_indices`` draws and the optimizer step sees a single sigma.
    requires_per_sample_sde_index = True

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
        # The trainer stamps sde_index_per_sample for every FlashGRPO recipe, so a
        # stage whose replay ignores it would score every row at one group's sigma.
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
        except Exception as exc:
            error = exc
            statistics[2] = 1.0
        total, count, invalid, *counts = self.backend.all_reduce_loss_sums(statistics)
        if invalid:
            raise ValueError(
                "FlashGRPO batch_normalize_tgr: invalid input on a training rank; "
                f"{error or 'check the failing rank for its schedule/step error'}. Fix the rollout schedule and step pool."
            ) from error
        if count <= 0 or not torch.isfinite(torch.tensor(total / count)) or total <= 0:
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
        segment.flash_tgr_weights = (weights / norm).reshape(-1, 1).detach()
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
        # The trainer stamps every FlashGRPO rollout; a missing stamp means it never stratified.
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
            if segment.flash_tgr_weights is None:
                raise ValueError("FlashGRPO batch_normalize_tgr requires prepare_part before micro-batch slicing.")
            tgr = segment.flash_tgr_weights.to(device=new_logp.device)
        else:
            tgr = self._rectification_weights_per_sample(segment=segment, device=new_logp.device)
        tgr = tgr.to(dtype=loss_per_elem.dtype)
        loss_per_elem = loss_per_elem * tgr
        policy_loss = loss_per_elem.mean()
        loss = policy_loss
        if self.beta > 0.0:
            kl_ref = reference_kl_loss(
                self.stage,
                self.ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
                new_means=replay_result.prev_sample_means,
                algo="FlashGRPO",
                new_replay=replay_result,
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
            loss=metrics.get("total_loss", metrics["policy_loss"]),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

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
