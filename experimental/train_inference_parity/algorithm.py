"""Experimental GRPO with collective-safe exact replay/rollout parity."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, Mapping, Optional

import torch

from experimental.train_inference_parity.gate import collective_error_barrier, exact_parity_gate
from unirl.algorithms.base import (
    AlgorithmStepResult,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
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
        super().prepare_segment(conditions=conditions, segment=segment)
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
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        context_factory = getattr(self.stage, "exact_context", None)
        if not callable(context_factory):
            raise RuntimeError("TrainInferenceParityGRPO requires stage.exact_context()")
        return typed_conds, context_factory

    def _build_policy_loss(
        self,
        *,
        segment: "TextSegment",
        new_logp: torch.Tensor,
        advantages: torch.Tensor,
        training_progress: float,
    ) -> tuple[torch.Tensor, float, Dict[str, torch.Tensor]]:
        old_logp = segment.rollout_log_probs.to(device=new_logp.device)
        adv_per_token = self._expand_advantages_to_tokens(
            advantages,
            segment.lengths,
            dtype=new_logp.dtype,
            device=new_logp.device,
        )
        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(
                self.clip_range_high,
                self.clip_schedule,
                training_progress,
            )
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        mask: Optional[torch.Tensor] = None
        if segment.loss_mask is not None:
            mask = segment.loss_mask.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
            loss_per_elem = loss_per_elem * mask

        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean"):
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            if mask is None:
                if self.loss_agg_mode == "seq-mean-token-sum-norm":
                    loss = torch.stack([part.sum() for part in parts]).mean() / float(self.horizon)
                else:
                    loss = torch.stack([part.mean() if part.numel() else part.new_zeros(()) for part in parts]).mean()
            else:
                mask_parts = torch.split(mask, segment.lengths.tolist())
                valid_parts = [
                    (part, float(mask_part.sum().item()))
                    for part, mask_part in zip(parts, mask_parts)
                    if bool(mask_part.any())
                ]
                if self.loss_agg_mode == "seq-mean-token-sum-norm":
                    per_seq = [part.sum() / float(self.horizon) for part, _ in valid_parts]
                else:
                    per_seq = [part.sum() / weight for part, weight in valid_parts]
                loss = torch.stack(per_seq).mean() if per_seq else loss_per_elem.sum() * 0.0
        elif mask is None:
            loss = loss_per_elem.mean()
        else:
            loss = loss_per_elem.sum() / mask.sum().clamp(min=1)
        return loss, clip_range, ratio_metrics

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
        # Replay and FSDP forwards have their own collectives. Every rank must
        # agree to enter them, or every rank must fail before they start.
        collective_error_barrier(setup_error, phase="pre-replay")

        # Keep exact mode active through backward. Non-reentrant activation
        # checkpointing reruns forward operators from inside ``backward()``.
        with ExitStack() as exact_stack:
            exact_stack.enter_context(context_factory())
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

            # No rank raises a local shape/dtype/finite/setup error before this
            # fixed collective. Every rank either continues or fails together.
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
