"""Stage-driven ``GSPO`` (Group Sequence Policy Optimization) over a ``TextSegment``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_k3,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


@dataclass
class GSPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 3e-4
    clip_schedule: str = "constant"
    old_logp_source: str = "rollout"


class GSPO(StageAlgorithm):
    """Sequence-level GSPO over an AR ``TextSegment`` via ``ARStage.replay``."""

    supports_multi_update = True
    anchor_fields = ("log_probs", "rollout_log_probs")

    def recomputes_anchor(self) -> bool:
        return self.old_logp_source == "replay"

    _MAX_LOG_RATIO = 10.0

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        clip_range: float = 3e-4,
        clip_schedule: str = "constant",
        clip_range_high: Optional[float] = None,
        loss_agg_mode: str = "seq-mean",
        conditions_cls: Optional[Type[Any]] = None,
        sampling_temperature: Optional[float] = None,
        old_logp_source: str = "rollout",
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("GSPO: either `stage` or `pipeline` must be provided")
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
        self.old_logp_source = str(old_logp_source).strip().lower()
        if self.old_logp_source not in ("rollout", "replay"):
            raise ValueError(f"GSPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}")

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the selected π_old anchor before optimizer updates."""
        if segment.tokens is None or segment.log_probs is None or int(segment.tokens.shape[0]) == 0:
            return
        if segment.rollout_log_probs is None:
            segment.rollout_log_probs = segment.log_probs.detach().cpu().clone()
        if self.old_logp_source == "rollout":
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            frozen = self.stage.replay(
                typed_conds,
                segment=segment,
                temperature=self.sampling_temperature,
            )
        segment.log_probs = frozen.detach().cpu()

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

        seq_new, seq_old, seq_adv = self._reduce_to_sequences(new_logp, old_logp, advantages, segment.lengths)
        if seq_new.numel() == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        log_ratio = (seq_new - seq_old).clamp(max=self._MAX_LOG_RATIO)
        loss_per_seq, ratio_metrics = _grpo_clip_loss(
            new_logp=log_ratio,
            old_logp=torch.zeros_like(log_ratio),
            advantages=seq_adv,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        loss = loss_per_seq.mean()
        (loss * loss_scale).backward()

        rollout_logp = (segment.rollout_log_probs if segment.rollout_log_probs is not None else segment.log_probs).to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, rollout_logp),
            **rollout_replay_k3(new_logp, rollout_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )

    @staticmethod
    def _reduce_to_sequences(
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce packed per-token log-probs to one length-normalized value per sequence via a segment-sum."""
        device = new_logp.device
        lengths = lengths.to(device)
        num_seqs = int(lengths.shape[0])
        if int(advantages.shape[0]) != num_seqs:
            raise ValueError(f"GSPO: advantages batch={int(advantages.shape[0])} != sequences={num_seqs}")

        seg_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), lengths)
        denom = lengths.to(new_logp.dtype).clamp(min=1)
        seq_new = new_logp.new_zeros(num_seqs).index_add(0, seg_ids, new_logp) / denom
        seq_old = old_logp.new_zeros(num_seqs).index_add(0, seg_ids, old_logp) / denom
        seq_adv = advantages.detach().to(dtype=new_logp.dtype, device=device)

        valid = lengths > 0
        if bool(valid.all()):
            return seq_new, seq_old, seq_adv
        return seq_new[valid], seq_old[valid], seq_adv[valid]


__all__ = ["GSPO", "GSPOConfig"]
