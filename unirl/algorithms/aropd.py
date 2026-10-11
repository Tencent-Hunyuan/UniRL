"""AROPD — on-policy distillation for autoregressive LMs (teacher-anchored)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Type

import torch

from unirl.train.lora import adapter_active, adapter_names
from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    StageAlgorithm,
    rollout_replay_k3,
    rollout_replay_logp_absdiff,
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

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        backend: Any = None,
        teacher: str = "teacher",
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
        normalize_gap: bool = True,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("AROPD: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.conditions_cls = conditions_cls
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        # Must equal sampling.temperature so the teacher replay's log_softmax(logits/T)
        # lives on the same tempered scale as the rollout's per-token log-probs.
        self.sampling_temperature = float(sampling_temperature)
        self.normalize_gap = bool(normalize_gap)
        model = getattr(backend, "model", None) if backend is not None else None
        if model is None:
            raise ValueError(
                "AROPD: no `backend` was injected — the teacher adapter lives on the "
                "trainable model. ARTrainer injects it when the algorithm declares "
                "requires_backend=True."
            )
        self._model = model
        self.teacher = str(teacher)
        present = adapter_names(model)
        if self.teacher not in present:
            raise ValueError(
                f"AROPD: teacher adapter {self.teacher!r} not found on the trainable model "
                f"(present: {sorted(present)}). Declare it under backend.lora_cfg.frozen_adapters "
                "as {name, path} entries."
            )
        # Token-share weighting keeps the masked token-mean loss invariant under micro-batching.
        self.loss_weighting = "token"

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the packed teacher log-prob anchor under the frozen adapter."""
        if segment.tokens is None or int(segment.tokens.shape[0]) == 0:
            return
        if segment.log_probs is None:
            raise RuntimeError(
                "AROPD.prepare_segment: segment.log_probs is None — the student side of the "
                "gap needs the rollout's per-token log-probs."
            )
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
        """Gap-weighted student score: ``-sum(sg(teacher_logp - old_logp) * new_logp)``."""
        if (
            segment.tokens is None
            or segment.lengths is None
            or segment.log_probs is None
            or segment.teacher_log_probs is None
        ):
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if int(segment.tokens.shape[0]) == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens]
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)
        teacher_logp = segment.teacher_log_probs.to(dtype=new_logp.dtype, device=new_logp.device)

        # The stop-gradient teacher-student gap is the per-token REINFORCE advantage;
        # never backprop into mean(student_logp - teacher_logp) directly (see the
        # algorithms README for the derivation). Batch-normalizing the gap is a
        # control variate — E[b·∇log π_θ] = 0 keeps the estimator unbiased — and
        # bounds the raw |gap| scale (a teacher that despises a sampled continuation
        # emits -15-nat gaps whose unnormalized gradient swamps the step; mirrors
        # the repo's GRPO normalize_adv_by_std and DiffusionOPD's sigma scaling).
        gap = (teacher_logp - old_logp).detach()
        if self.normalize_gap:
            gap = (gap - gap.mean()) / (gap.std() + 1e-6)
        loss_per_elem = -(gap * new_logp)
        if segment.loss_mask is not None:
            mask = segment.loss_mask.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
            loss_per_elem = loss_per_elem * mask
            loss = loss_per_elem.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        rollout_logp = (segment.rollout_log_probs if segment.rollout_log_probs is not None else old_logp).to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        metrics: Dict[str, Any] = {
            "distill_loss": float(loss.detach().item()),
            "teacher_gap_mean": float(gap.mean().item()),
            "teacher_gap_abs_mean": float(gap.abs().mean().item()),
            **rollout_replay_logp_absdiff(new_logp, rollout_logp),
            **rollout_replay_k3(new_logp, rollout_logp),
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["AROPD"]
