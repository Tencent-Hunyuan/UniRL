"""Experimental GRPO with collective-safe exact replay/rollout parity."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch

from experimental.train_inference_parity.gate import collective_error_barrier, exact_parity_gate
from unirl.algorithms.base import (
    AlgorithmStepResult,
    typed_conditions,
)
from unirl.algorithms.grpo import GRPO
from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment


class TrainInferenceParityGRPO(GRPO):
    """GRPO whose optimization anchor is the exact rollout log-probability."""

    supports_multi_update = False

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        if self.old_logp_source != "rollout":
            raise ValueError("exact parity requires old_logp_source='rollout'")
        if not isinstance(segment.log_probs, torch.Tensor):
            raise TypeError("TrainInferenceParityGRPO requires rollout log_probs")
        segment.rollout_log_probs = segment.log_probs.detach().cpu().clone()

    def _prepare_exact_replay(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> Any:
        if segment.tokens is None or segment.lengths is None:
            raise ValueError("TrainInferenceParityGRPO requires packed tokens and lengths on every rank")
        if not isinstance(segment.rollout_log_probs, torch.Tensor):
            raise TypeError("TrainInferenceParityGRPO requires segment.rollout_log_probs")
        segment.log_probs = segment.rollout_log_probs
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        context_factory = getattr(self.stage, "exact_context", None)
        if not callable(context_factory):
            raise RuntimeError("TrainInferenceParityGRPO requires stage.exact_context()")
        return typed_conds, context_factory

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        setup_error: Optional[str] = None
        typed_conds = None
        context_factory = None
        try:
            typed_conds, context_factory = self._prepare_exact_replay(
                conditions=conditions,
                segment=segment,
            )
        except Exception as error:
            setup_error = f"{type(error).__name__}: {error}"
        collective_error_barrier(setup_error, phase="pre-replay")

        with context_factory():
            replay_error: Optional[str] = None
            new_logp: Optional[torch.Tensor] = None
            rollout_anchor: Optional[torch.Tensor] = None
            try:
                new_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
                rollout_anchor = segment.rollout_log_probs
            except Exception as error:
                replay_error = f"{type(error).__name__}: {error}"
                new_logp = None
                rollout_anchor = None

            parity_metrics = exact_parity_gate(
                new_logp if isinstance(new_logp, torch.Tensor) else None,
                rollout_anchor,
                local_error=replay_error,
            )
            if new_logp is None or new_logp.numel() == 0:
                return AlgorithmStepResult(
                    loss=0.0,
                    metrics=parity_metrics,
                    num_steps_or_tokens=0,
                    has_backward=False,
                )
            loss, clip_range, ratio_metrics = self._build_policy_loss(
                segment=segment,
                new_logp=new_logp,
                advantages=advantages,
                training_progress=training_progress,
            )
            (loss * loss_scale).backward()

        metrics: Dict[str, Any] = dict(parity_metrics)
        metrics.update(
            {
                "policy_loss": float(loss.detach().item()),
                "clip_range": float(clip_range),
                **{key: float(value.item()) for key, value in ratio_metrics.items()},
            }
        )
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["TrainInferenceParityGRPO"]
