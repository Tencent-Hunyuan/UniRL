"""Shared skeleton for AR ratio-clip algorithms (GRPO / GSPO).

Both GRPO and GSPO are ``StageAlgorithm``s that run the same PPO-style
clipped surrogate over a ``TextSegment``: the teacher-forced forward and
per-token log-prob recompute is owned by ``stage.replay(...)``, the rollout
log-prob is the frozen ``old_logp`` anchor, and the loss is a clipped ratio
objective. They differ in exactly ONE place — the granularity of the
importance ratio (GRPO: per token; GSPO: per length-normalized sequence).

This module factors the identical part (construction, empty-segment guards,
replay, clip-range scheduling, backward, metric assembly, result packing)
into :class:`_ARClipStageAlgorithm` via the template-method pattern, leaving
subclasses to implement only :meth:`_policy_loss`.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    StageAlgorithm,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


class _ARClipStageAlgorithm(StageAlgorithm):
    """Template for AR clipped-ratio algorithms; subclasses supply the ratio.

    Skeleton owned here: ``stage.replay`` → frozen rollout ``old_logp`` →
    scheduled clip range → :meth:`_policy_loss` (subclass hook) → ``backward`` →
    metric assembly → :class:`AlgorithmStepResult`. Subclasses implement
    :meth:`_policy_loss`, the only part that differs between token-level (GRPO)
    and sequence-level (GSPO) importance ratios.
    """

    # old_logp is the rollout (SGLang) log-prob, frozen on the segment and
    # unchanged across mini-batch updates, so reusing it across
    # num_updates_per_batch>1 is the deliberate rollout-anchored PPO ratio
    # (verl bypass_mode=True parity); the ratio then absorbs the rollout-vs-train
    # engine gap on later mini-batches (accepted for parity).
    supports_multi_update = True

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        loss_agg_mode: str = "token-mean",
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError(f"{type(self).__name__}: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.clip_range = float(clip_range)
        self.clip_range_high = None if clip_range_high is None else float(clip_range_high)
        self.clip_schedule = str(clip_schedule)
        self.loss_agg_mode = str(loss_agg_mode)
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)

    def _policy_loss(
        self,
        *,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        segment: "TextSegment",
        clip_range: float,
        clip_range_high: Optional[float],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute the clipped policy loss and ratio metrics for one micro-batch.

        Args:
            new_logp: differentiable replay log-probs, packed-varlen.
            old_logp: frozen rollout log-probs, aligned with ``new_logp``.
            advantages: per-sample advantages.
            segment: the AR text segment (provides ``lengths`` / cu_seqlens).
            clip_range: schedule-resolved lower clip epsilon.
            clip_range_high: schedule-resolved upper clip epsilon, or ``None``.

        Returns:
            ``(loss_scalar, ratio_metrics)``. ``loss_scalar`` is ``None`` when
            there are no valid samples (caller then skips backward).
        """
        raise NotImplementedError

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.tokens is None or segment.lengths is None or segment.log_probs is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )

        loss, ratio_metrics = self._policy_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=advantages,
            segment=segment,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        if loss is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["_ARClipStageAlgorithm"]
