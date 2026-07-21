"""Stage-driven ``GSPO`` (Group Sequence Policy Optimization) over a ``TextSegment``.

Implements **GSPO**, introduced in Zheng et al. "Group Sequence Policy
Optimization" (arXiv:2507.18071). GSPO is the sequence-level sibling of
:class:`~unirl.algorithms.grpo.GRPO`: GRPO forms a **per-token** importance
ratio and clips each token; GSPO forms **one ratio per sequence** from the
length-normalized sequence log-ratio (paper Eq. 7-8)::

    s_i = (1 / |y_i|) * Σ_t (new_logp_{i,t} - old_logp_{i,t})
    ratio_i = exp(s_i)
    loss = mean_i  max( -A_i * ratio_i,  -A_i * clip(ratio_i, 1-ε, 1+ε) )

and applies the clipped surrogate at the sequence granularity. This removes the
per-token ratio variance that destabilizes MoE RL (the paper's motivation), so it
pairs naturally with the Qwen3-Omni thinker (a Qwen3-MoE decoder).

This is a self-contained :class:`StageAlgorithm` mirroring the construction of
:class:`~unirl.algorithms.grpo.GRPO` / :class:`~unirl.algorithms.cppo.CPPO` /
:class:`~unirl.algorithms.drpo.DRPO` (same ``compute_loss_and_backward`` skeleton:
empty-segment guards → ``stage.replay`` → frozen rollout ``old_logp`` anchor →
clip-range schedule → loss → ``backward`` → metric assembly). The teacher-forced
forward and per-token log-prob recompute are owned by ``stage.replay(...)``; the
algorithm only adds the sequence-level reduction. It reuses the shared
``unirl.algorithms.base._grpo_clip_loss`` helper at sequence granularity so the
ratio it forms is ``exp(s_new - s_old) = exp(s_i)``.

Provenance / relation to other code
------------------------------------
This is an **independent UniRL implementation**, not a port of any external
code. The per-token → per-sequence reduction uses a segment-sum over
``segment.lengths`` (the framework's cu_seqlens), NOT a token-mask
vectorization. Only the algorithm's mathematical definition (the equations
above) is shared with other GSPO implementations; equations are not
copyrightable and the paper is public. No third-party source was copied.

GSPO's clip range is much tighter than GRPO's (the paper uses ε≈3e-4); set it
in the recipe.
"""

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
    # GSPO's sequence-level ratio has much lower variance than GRPO's per-token
    # ratio, so the clip range is ~10-30x tighter (paper: ε≈3e-4).
    clip_range: float = 3e-4
    clip_schedule: str = "constant"


class GSPO(StageAlgorithm):
    """Sequence-level GSPO over an AR ``TextSegment`` via ``ARStage.replay``.

    Forms ONE importance ratio per sequence from the length-normalized sequence
    log-ratio and runs the PPO clipped surrogate at sequence granularity. The
    teacher-forced forward and per-token log-prob recompute is owned by
    :meth:`ARStage.replay`; this class reduces the packed per-token log-probs
    to one length-normalized value per sequence and feeds that through the
    shared PPO clip math.

    Args:
        stage: The :class:`ARStage` whose ``replay`` produces packed-varlen
            new log-probs aligned with ``segment.log_probs``.
        clip_range: PPO clip range epsilon (sequence-level; much smaller than
            GRPO's per-token range).
        clip_schedule: ``"constant"``, ``"linear_decay"``, or
            ``"cosine_decay"``.
        clip_range_high: DAPO "clip-higher" upper epsilon; ``None`` ⇒ symmetric.
        loss_agg_mode: accepted for recipe symmetry; GSPO is inherently
            sequence-mean (one term per sequence), so it does not change the
            reduction.
        conditions_cls: Stage-typed conditions container with
            ``from_dict(Mapping[str, Condition])``.
        sampling_temperature: AR rollout temperature, applied as a
            ``logits / T`` scaling inside :meth:`ARStage.replay` so replay's
            log-softmax matches SGLang's sampling distribution
            (``log_softmax(logits / T)``).
    """

    # old_logp is the rollout (SGLang) log-prob, frozen on the segment and
    # unchanged across mini-batch updates, so reusing it across
    # num_updates_per_batch>1 is the deliberate rollout-anchored PPO ratio
    # (verl bypass_mode=True parity), matching GRPO/DRPO; the ratio then absorbs
    # the rollout-vs-train engine gap on later mini-batches (accepted for parity).
    supports_multi_update = True

    # Upper bound on the per-sequence log-ratio before exp(), guarding against
    # overflow to inf when the sequence is far off-policy (early training).
    # Mirrors verl's clamp(log_seq_importance_ratio, max=10.0).
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
        )  # [total_tokens]
        # old_logp = the rollout log-prob, frozen on the segment — the deliberate
        # rollout-anchored ratio across num_updates_per_batch steps (see the
        # supports_multi_update class comment; verl bypass_mode=True parity).
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

        # ratio_i = exp(s_i) with s_i = mean_t(new) - mean_t(old). Clamp the
        # log-ratio before _grpo_clip_loss exponentiates it (numerical stability).
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

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            # Rollout↔replay drift on the raw PER-TOKEN log-probs (before the
            # sequence reduction) — the direct autoregress-vs-replay correctness
            # gauge. k3 is the calibrated KL surrogate (p=replay new, q=rollout
            # old); on-policy it is ~0 and rises if any misaligns between rollout
            # and replay.
            **rollout_replay_k3(new_logp, old_logp),
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
        """Reduce packed per-token log-probs to one length-normalized value per
        sequence via a vectorized segment-sum over cu_seqlens (no Python loop).

        Returns ``(seq_new, seq_old, seq_adv)`` for sequences with length > 0.
        ``seq_new`` stays differentiable (grad flows through replay); ``seq_old``
        is frozen.
        """
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
