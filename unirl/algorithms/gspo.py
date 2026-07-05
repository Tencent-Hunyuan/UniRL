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

Provenance / relation to other code
------------------------------------
This is an **independent UniRL implementation**, not a port. The construction,
replay, clip scheduling, backward, and metric plumbing are shared with GRPO via
:class:`unirl.algorithms._ar_clip._ARClipStageAlgorithm`; this class implements
only the sequence-level reduction. The per-token → per-sequence reduction uses a
segment-sum over ``segment.lengths`` (the framework's cu_seqlens), and the shared
``_grpo_clip_loss`` is reused at sequence granularity so the ratio it forms is
``exp(s_new - s_old) = exp(s_i)``. Only the algorithm's mathematical definition
(the equations above) is shared with other GSPO implementations; equations are
not copyrightable. GSPO's clip range is much tighter than GRPO's (the paper uses
ε≈3e-4); set it in the recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ._ar_clip import _ARClipStageAlgorithm
from .base import BaseAlgorithmConfig, _grpo_clip_loss


@dataclass
class GSPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    # GSPO's sequence-level ratio has much lower variance than GRPO's per-token
    # ratio, so the clip range is ~10-30x tighter (paper: ε≈3e-4).
    clip_range: float = 3e-4
    clip_schedule: str = "constant"


class GSPO(_ARClipStageAlgorithm):
    """Sequence-level GSPO over an AR ``TextSegment`` via ``ARStage.replay``.

    Args mirror the shared base; only the ratio granularity differs (sequence vs
    token). ``clip_range`` / ``clip_range_high`` are the sequence-ratio clip
    bounds (much smaller than GRPO's). ``loss_agg_mode`` is accepted for recipe
    symmetry but GSPO is inherently sequence-mean (one term per sequence), so it
    does not change the reduction.
    """

    # Upper bound on the per-sequence log-ratio before exp(), guarding against
    # overflow to inf when the sequence is far off-policy (early training).
    # Mirrors verl's clamp(log_seq_importance_ratio, max=10.0).
    _MAX_LOG_RATIO = 10.0

    def __init__(
        self,
        *,
        clip_range: float = 3e-4,
        loss_agg_mode: str = "seq-mean",
        **kwargs,
    ) -> None:
        super().__init__(clip_range=clip_range, loss_agg_mode=loss_agg_mode, **kwargs)

    def _policy_loss(
        self,
        *,
        new_logp: torch.Tensor,
        old_logp: torch.Tensor,
        advantages: torch.Tensor,
        segment,
        clip_range: float,
        clip_range_high: Optional[float],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        seq_new, seq_old, seq_adv = self._reduce_to_sequences(new_logp, old_logp, advantages, segment.lengths)
        if seq_new.numel() == 0:
            return None, {}

        # ratio_i = exp(s_i) with s_i = mean_t(new) - mean_t(old). Clamp the
        # log-ratio before _grpo_clip_loss exponentiates it (numerical stability).
        log_ratio = (seq_new - seq_old).clamp(max=self._MAX_LOG_RATIO)
        loss_per_seq, ratio_metrics = _grpo_clip_loss(
            new_logp=log_ratio,
            old_logp=torch.zeros_like(log_ratio),
            advantages=seq_adv,
            clip_range=clip_range,
            clip_range_high=clip_range_high,
        )
        return loss_per_seq.mean(), ratio_metrics

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
