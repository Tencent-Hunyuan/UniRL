"""PerSampleStepGRPO: FlowGRPO over rows that each trained one SDE step, scaled by a per-row weight."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Mapping, Optional, Type

import torch

from unirl.sde.kernels import StepStrategy
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import AlgorithmStepResult, _grpo_clip_loss, _resolve_clip_range_from_schedule, typed_conditions
from .flowgrpo import FlowGRPO


class PerSampleStepGRPO(FlowGRPO):
    """Shared loss of FlashGRPO and TempFlowGRPO; subclasses define ``_row_weights`` (README)."""

    # The trainer reads this to fan one rollout into per-step generates and stamp
    # ``segment.sde_index_per_sample`` before merging the groups into one track.
    per_sample_sde_layout: ClassVar[str] = ""  # "stratified" (FlashGRPO) or "branched" (TempFlowGRPO)
    required_strategy: ClassVar[Type[StepStrategy]] = StepStrategy
    weight_metric: ClassVar[str] = "row_weight"

    def __init__(
        self,
        *,
        params: Any,
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
        name = type(self).__name__
        # Checked before super() so a beta>0 recipe never loads the reference model.
        if float(beta) != 0.0:
            raise ValueError(
                f"{name}: reference-KL is not supported on the per-sample SDE-index path; got beta={beta!r}."
            )
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
        if not float(adv_clip_max) > 0.0:
            raise ValueError(f"{name}: adv_clip_max must be > 0; got {adv_clip_max!r}.")
        self.adv_clip_max = float(adv_clip_max)
        # The trainer stamps sde_index_per_sample, so a stage whose replay ignores it
        # would score every row at one group's sigma.
        if not getattr(self.stage, "supports_per_sample_sde_index", False):
            raise ValueError(
                f"{name} requires a stage that honours segment.sde_index_per_sample; "
                f"{type(self.stage).__name__} does not. Use FlowGRPO, or implement the "
                "per-sample replay on that stage and declare supports_per_sample_sde_index."
            )
        strategy = getattr(self.stage, "strategy", None)
        if not isinstance(strategy, self.required_strategy):
            raise ValueError(
                f"{name} requires the stage strategy to be {self.required_strategy.__name__}; "
                f"got {type(strategy).__name__}."
            )

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        name = type(self).__name__
        # The trainer stamps every per-sample rollout; a missing stamp means it never fanned out.
        if segment.sde_index_per_sample is None:
            raise ValueError(f"{name} requires segment.sde_index_per_sample; the rollout was not split per step.")
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

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

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_raw = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device)
        adv_clipped = adv_raw.clamp(-self.adv_clip_max, self.adv_clip_max)
        adv_b = adv_clipped.reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )

        weights = self._row_weights(segment=segment, device=new_logp.device)
        weights = weights.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
        loss_per_elem = loss_per_elem * weights
        loss = loss_per_elem.mean()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            f"{self.weight_metric}_mean": float(weights.detach().mean().item()),
            f"{self.weight_metric}_min": float(weights.detach().min().item()),
            f"{self.weight_metric}_max": float(weights.detach().max().item()),
            "adv_clip_fraction": float((adv_raw.abs() > self.adv_clip_max).float().mean().item()),
            "adv_abs_max": float(adv_raw.abs().max().item()),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }

        (loss * loss_scale).backward()

        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _row_weights(self, *, segment: LatentSegment, device: torch.device) -> torch.Tensor:
        """Per-row loss weight ``[N, 1]`` at each row's own SDE step."""
        raise NotImplementedError


__all__ = ["PerSampleStepGRPO"]
