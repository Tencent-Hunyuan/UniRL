"""AROPD — on-policy distillation for autoregressive LMs (teacher-anchored)."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.train.lora import adapter_active, adapter_names
from unirl.types.conditions import Condition
from unirl.types.loss_agg import LossAggMode
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    StageAlgorithm,
    aggregate_token_losses,
    rollout_replay_k3,
    rollout_replay_logp_absdiff,
    select_active_tokens,
    typed_conditions,
)


class AROPD(StageAlgorithm):
    """Stop-gradient policy-gradient OPD over the student's own AR rollout."""

    # Teacher adapter lives on the trainable model; the trainer injects the backend.
    requires_backend = True
    # Supervision is teacher-driven; rewards (if any) are monitoring-only.
    requires_advantages = False
    # The teacher anchor is frozen once per rollout; re-anchored per-update is unvalidated.
    supports_multi_update = False
    # The teacher anchor must be replayed at the exact training-micro geometry (bf16
    # forwards are batch-shape sensitive), like every replay-sourced AR anchor.
    anchor_fields = ("teacher_log_probs",)
    recomputes_anchor = True
    # The surrogate is a flat masked token mean; no other mode is implemented. The
    # stack's token-share weighting (loss_scale.py) keys on this declaration.
    loss_agg_mode = LossAggMode.TOKEN_MEAN.value

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        backend: Any = None,
        teacher: str = "teacher",
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: float,
        gap_clamp: Optional[float] = 10.0,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("AROPD: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.conditions_cls = conditions_cls
        # Must equal sampling.temperature so the teacher replay's log_softmax(logits/T)
        # matches the student's tempered replay; required — a silent default would put
        # the anchor on the wrong temperature with no raise (see the algorithms README).
        self.sampling_temperature = float(sampling_temperature)
        # Elementwise |gap| bound: grouping-invariant (a per-token op) and it biases
        # the estimator only when it clips (the gap_clipped_tokens / gap_active_tokens
        # counters report how often).
        # None = raw gaps.
        if gap_clamp is not None and (not math.isfinite(gap_clamp) or gap_clamp <= 0):
            raise ValueError(f"AROPD: gap_clamp must be a finite positive bound or None; got {gap_clamp!r}.")
        self.gap_clamp = None if gap_clamp is None else float(gap_clamp)
        model = getattr(backend, "model", None) if backend is not None else None
        if model is None:
            raise ValueError(
                "AROPD: no `backend` was injected — the teacher adapter lives on the "
                "trainable model. ARTrainer injects it when the algorithm declares "
                "requires_backend=True."
            )
        self._model = model
        self.teacher = teacher
        present = adapter_names(model)
        if self.teacher not in present:
            raise ValueError(
                f"AROPD: teacher adapter {self.teacher!r} not found on the trainable model "
                f"(present: {sorted(present)}). Declare it under backend.lora_cfg.frozen_adapters "
                "as {name, path} entries."
            )

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the packed teacher log-prob anchor under the frozen adapter."""
        if int(segment.tokens.shape[0]) == 0:
            # Empty micro: still write the anchor so the multi-micro anchor path
            # does not raise "a micro produced None" (GRPO's anchors are pre-populated).
            segment.teacher_log_probs = segment.log_probs.new_zeros(0)
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad(), adapter_active(self._model, self.teacher):
            teacher_logp = self.stage.replay(
                typed_conds, segment=segment, temperature=self.sampling_temperature
            )  # [total_tokens]
        segment.teacher_log_probs = teacher_logp.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Gap-weighted student score: ``-sum(sg(teacher_logp - new_logp) * new_logp)``."""
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.teacher_log_probs is None:
            raise RuntimeError(
                "AROPD.compute_loss_and_backward: segment.teacher_log_probs is None — the "
                "anchor step was skipped; this is a stack wiring bug, not a config error."
            )

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens]
        teacher_logp = segment.teacher_log_probs.to(dtype=new_logp.dtype, device=new_logp.device)

        # The stop-gradient teacher-student gap is the per-token advantage; never
        # backprop into mean(student_logp - teacher_logp) directly (its gradient is
        # mean(grad log pi) — zero in expectation, and the gap's sign never enters).
        # The student side is new_logp.detach() — current weights at the teacher
        # anchor's micro geometry, so the gap carries no rollout-vs-replay drift;
        # segment.log_probs only feeds the drift metrics below. gap_clamp is an
        # elementwise, grouping-invariant bound that biases the estimator when it
        # clips; the gap_clipped_tokens / gap_active_tokens counters report how often.
        raw_gap = teacher_logp - new_logp.detach()
        gap = raw_gap if self.gap_clamp is None else raw_gap.clamp(-self.gap_clamp, self.gap_clamp)
        loss = aggregate_token_losses(
            -(gap * new_logp),
            lengths=segment.lengths,
            loss_mask=segment.loss_mask,
            loss_agg_mode=self.loss_agg_mode,
            horizon=0,
        )
        (loss * loss_scale).backward()

        rollout_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        metric_gap = select_active_tokens(segment.loss_mask, raw_gap)[0]
        metrics: Dict[str, Any] = {
            "distill_loss": float(loss.detach().item()),
            "teacher_gap_mean": float(metric_gap.mean().item()),
            "teacher_gap_abs_mean": float(metric_gap.abs().mean().item()),
            **rollout_replay_logp_absdiff(new_logp, rollout_logp, segment.loss_mask),
            **rollout_replay_k3(new_logp, rollout_logp, segment.loss_mask),
        }
        if self.gap_clamp is not None:
            # Additive counters: the metric aggregator plain-averages values, so a
            # per-micro fraction would be micro-weighted; the ratio of the two
            # averaged counters equals the true token-level clip fraction.
            metric_clipped = select_active_tokens(segment.loss_mask, raw_gap)[0]
            metrics["gap_clipped_tokens"] = int((metric_clipped.abs() > self.gap_clamp).sum().item())
            metrics["gap_active_tokens"] = int(metric_clipped.numel())
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["AROPD"]
