"""Stage-driven ``GSPO`` (Group Sequence Policy Optimization) over a ``TextSegment``.

Implements **GSPO**, introduced in Zheng et al. "Group Sequence Policy
Optimization" (arXiv:2507.18071). GSPO is the sequence-level sibling of
:class:`GRPO`: GRPO forms a **per-token** importance ratio
``exp(new_logp_t - old_logp_t)`` and clips each token; GSPO forms **one ratio per
sequence** from the length-normalized sequence log-ratio (paper Eq. 7-8)::

    s_i = (1 / |y_i|) * Σ_t (new_logp_{i,t} - old_logp_{i,t})
    ratio_i = exp(s_i)
    loss = mean_i  max( -A_i * ratio_i,  -A_i * clip(ratio_i, 1-ε, 1+ε) )

and applies the clipped surrogate at the sequence granularity. This removes the
per-token ratio variance that destabilizes MoE RL (the paper's motivation), so it
pairs naturally with the Qwen3-Omni thinker (a Qwen3-MoE decoder).

Provenance / relation to other code
------------------------------------
This is an **independent UniRL implementation**, not a port: it mirrors the
sibling :class:`unirl.algorithms.grpo.GRPO` (same ``StageAlgorithm`` contract,
``stage.replay`` owns the teacher-forced per-token ``new_logp`` recompute) and
reuses UniRL's shared ``_grpo_clip_loss`` clip math at sequence granularity. It
reduces per-token log-ratios to per-sequence ratios via ``segment.lengths``
(``torch.split``) rather than the token-mask vectorization other frameworks use;
only the algorithm's mathematical definition (the equations above) is shared with
those, and equations are not copyrightable. GSPO's clip range is much tighter than
GRPO's (the paper uses ε≈3e-4); set it in the recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.text import TextSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    rollout_replay_logp_absdiff,
    typed_conditions,
)


@dataclass
class GSPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    # GSPO's sequence-level ratio has much lower variance than GRPO's per-token
    # ratio, so the clip range is ~10-30x tighter (paper: ε≈3e-4).
    clip_range: float = 3e-4
    clip_schedule: str = "constant"


class GSPO(StageAlgorithm):
    """Sequence-level GSPO over an AR ``TextSegment`` via ``ARStage.replay``.

    Args mirror :class:`GRPO`; only the ratio granularity differs (sequence vs
    token). ``clip_range`` / ``clip_range_high`` are the sequence-ratio clip
    bounds (much smaller than GRPO's). ``loss_agg_mode`` is accepted for recipe
    symmetry but GSPO is inherently sequence-mean (one term per sequence), so it
    does not change the reduction.
    """

    # old_logp is the frozen rollout log-prob on the segment; reusing it across
    # num_updates_per_batch>1 is the deliberate rollout-anchored ratio (GRPO parity).
    supports_multi_update = True

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
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens], differentiable
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)  # [total_tokens]

        # Reduce per-token log-ratios to ONE length-normalized log-ratio per
        # sequence (GSPO's s_i). Split by segment.lengths (framework cu_seqlens).
        lengths = [int(n) for n in segment.lengths.tolist()]
        adv = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device)  # [B]
        if int(adv.shape[0]) != len(lengths):
            raise ValueError(f"GSPO: advantages batch={int(adv.shape[0])} != sequences={len(lengths)}")

        new_parts = torch.split(new_logp, lengths)
        old_parts = torch.split(old_logp, lengths)
        seq_new: List[torch.Tensor] = []
        seq_old: List[torch.Tensor] = []
        seq_adv: List[torch.Tensor] = []
        for k, n in enumerate(lengths):
            if n <= 0:
                continue
            # Length-normalized sequence log-prob (mean over tokens). The new side
            # is differentiable (grad flows through replay); the old side is frozen.
            seq_new.append(new_parts[k].mean())
            seq_old.append(old_parts[k].mean())
            seq_adv.append(adv[k])
        if not seq_new:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        seq_new_t = torch.stack(seq_new)  # [B'] differentiable
        seq_old_t = torch.stack(seq_old)  # [B']
        seq_adv_t = torch.stack(seq_adv)  # [B']

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        clip_high = (
            None
            if self.clip_range_high is None
            else _resolve_clip_range_from_schedule(self.clip_range_high, self.clip_schedule, training_progress)
        )
        # Reuse the shared PPO clip math at SEQUENCE granularity: one element per
        # sequence, so the ratio it forms is exp(s_new - s_old) = GSPO's ratio_i.
        loss_per_seq, ratio_metrics = _grpo_clip_loss(
            new_logp=seq_new_t,
            old_logp=seq_old_t,
            advantages=seq_adv_t,
            clip_range=clip_range,
            clip_range_high=clip_high,
        )
        loss = loss_per_seq.mean()  # sequence-mean (GSPO is inherently per-sequence)
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


__all__ = ["GSPO", "GSPOConfig"]
