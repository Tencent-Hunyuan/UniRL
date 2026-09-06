"""Stage-driven ``FlowGRPO`` over a ``LatentSegment``."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _reference_kl_loss,
    _reference_replay_means,
    _require_replay_anchor_for_batched_replay,
    _resolve_clip_range_from_schedule,
    _resolve_reference_model,
    _transition_sigma,
    gather_sde_field,
    rollout_replay_logp_absdiff,
    typed_conditions,
)

logger = logging.getLogger(__name__)


@dataclass
class FlowGRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"
    beta: float = 0.0
    old_logp_source: str = "rollout"
    max_rollout_replay_logp_absdiff: Optional[float] = None
    rollout_replay_parity_action: str = "raise"
    params: Any = dc_field(default=None)


class FlowGRPO(StageAlgorithm):
    """GRPO over a diffusion ``LatentSegment`` via ``DiffusionStage.replay``."""

    supports_multi_update = True
    requires_backend = True
    anchor_fields = ("sde_logp",)

    def recomputes_anchor(self) -> bool:
        return self.old_logp_source == "replay"

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        beta: float = 0.0,
        old_logp_source: str = "rollout",
        max_rollout_replay_logp_absdiff: Optional[float] = None,
        rollout_replay_parity_action: str = "raise",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("FlowGRPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.params = params
        self.clip_range = float(clip_range)
        self.clip_schedule = str(clip_schedule)
        self.beta = float(beta)
        self._ref_model = _resolve_reference_model(backend, beta=self.beta, algo="FlowGRPO")
        self.old_logp_source = str(old_logp_source).strip().lower()
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowGRPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        self.max_rollout_replay_logp_absdiff = (
            None if max_rollout_replay_logp_absdiff is None else float(max_rollout_replay_logp_absdiff)
        )
        require(
            self.max_rollout_replay_logp_absdiff is None or self.max_rollout_replay_logp_absdiff >= 0,
            f"FlowGRPO: max_rollout_replay_logp_absdiff must be non-negative; got {max_rollout_replay_logp_absdiff!r}",
        )
        self.rollout_replay_parity_action = str(rollout_replay_parity_action).strip().lower()
        require(
            self.rollout_replay_parity_action in ("raise", "warn"),
            f"FlowGRPO: rollout_replay_parity_action must be 'raise' or 'warn'; got {rollout_replay_parity_action!r}",
        )
        _require_replay_anchor_for_batched_replay(self.stage, self.old_logp_source, algo="FlowGRPO")
        self.conditions_cls = conditions_cls

    @staticmethod
    def _parity_metrics(
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        target_steps: List[int],
    ) -> Dict[str, float]:
        """Rollout-vs-replay |Δlogp|, overall and per denoising step."""
        metrics = rollout_replay_logp_absdiff(new_logp, old_logp)
        for column, step_index in enumerate(target_steps):
            step = rollout_replay_logp_absdiff(new_logp[:, column], old_logp[:, column])
            metrics[f"rollout_replay_logp_absdiff_step_{step_index}_mean"] = step["rollout_replay_logp_absdiff_mean"]
            metrics[f"rollout_replay_logp_absdiff_step_{step_index}_max"] = step["rollout_replay_logp_absdiff_max"]
        return metrics

    def _enforce_parity(self, drift_metrics: Dict[str, float], target_steps: List[int]) -> None:
        """Trip the configured gate when replay's log-probs diverge from the rollout's."""
        threshold = self.max_rollout_replay_logp_absdiff
        observed = drift_metrics["rollout_replay_logp_absdiff_max"]
        if threshold is None or observed <= threshold:
            return
        per_step = ", ".join(
            f"{step}:{drift_metrics[f'rollout_replay_logp_absdiff_step_{step}_mean']:.6g}"
            f"/{drift_metrics[f'rollout_replay_logp_absdiff_step_{step}_max']:.6g}"
            for step in target_steps
        )
        message = (
            f"FlowGRPO rollout/replay log-prob parity exceeded {threshold:.6g}: "
            f"max={observed:.6g} mean={drift_metrics['rollout_replay_logp_absdiff_mean']:.6g}; "
            f"per-step mean/max [{per_step}]"
        )
        if self.rollout_replay_parity_action == "warn":
            logger.warning(message)
            return
        raise RuntimeError(message)

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Establish the frozen π_old anchor (``segment.sde_logp``) before the ``num_updates_per_batch`` loop."""
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        if self.old_logp_source == "rollout":
            if segment.sde_logp is None:
                raise RuntimeError(
                    "FlowGRPO.prepare_segment: old_logp_source='rollout' but the "
                    "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                    "None). Pin a rollout build that emits trajectory log-probs, or set "
                    "old_logp_source='replay'."
                )
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        segment.sde_logp = result.log_probs.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
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
        new_means = replay_result.prev_sample_means

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )

        drift_metrics = self._parity_metrics(new_logp, old_logp, target_steps)
        self._enforce_parity(drift_metrics, target_steps)

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_b,
            clip_range=clip_range,
        )
        policy_loss = loss_per_elem.mean()
        loss = policy_loss
        metrics: Dict[str, Any] = {
            "policy_loss": float(policy_loss.detach().item()),
            "clip_range": float(clip_range),
            **drift_metrics,
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }

        if self.beta > 0.0:
            if new_means is None:
                raise RuntimeError(
                    "FlowGRPO: beta>0 requires stage.replay() to return prev_sample_means, "
                    "but got None. Ensure the stage's replay method produces means."
                )
            sigma_t = _transition_sigma(
                self.stage,
                segment=segment,
                target_steps=target_steps,
                eta=float(self.params.eta),
                device=new_logp.device,
                add_coefficient=True,
            )
            ref_means = _reference_replay_means(
                self.stage,
                self._ref_model,
                conditions=typed_conds,
                segment=segment,
                params=self.params,
                target_steps=target_steps,
            ).to(dtype=new_means.dtype, device=new_means.device)
            kl_ref = _reference_kl_loss(new_means, ref_means, sigma_t)
            loss = loss + self.beta * kl_ref
            metrics["beta"] = float(self.beta)
            metrics["kl_ref_mean"] = float(kl_ref.detach().item())

        (loss * loss_scale).backward()

        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment."""
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]


__all__ = ["FlowGRPO", "FlowGRPOConfig"]
