"""Stage-driven ``GRPO`` over a ``TextSegment``.

GRPO forms a **per-token** importance ratio ``exp(new_logp_t - old_logp_t)``
and runs the PPO clipped surrogate at token granularity. The construction,
replay, clip scheduling, backward, and metric plumbing are shared with
:class:`~unirl.algorithms.gspo.GSPO` via
:class:`unirl.algorithms._ar_clip._ARClipStageAlgorithm`; this class implements
only the token-level ratio and the ``loss_agg_mode`` reduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from ._ar_clip import _ARClipStageAlgorithm
from .base import BaseAlgorithmConfig, _grpo_clip_loss


@dataclass
class GRPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "ar"
    conditions_cls: str = ""
    clip_range: float = 1e-4
    clip_schedule: str = "constant"


class GRPO(_ARClipStageAlgorithm):
    """GRPO over an AR ``TextSegment`` via ``ARStage.replay``.

    Per-sample advantages are expanded to per-token via ``cu_seqlens`` and fed
    through the shared PPO clip math at token granularity, then reduced by
    ``loss_agg_mode``:

    - ``"seq-mean-token-sum-norm"`` (Dr.GRPO/DAPO): per-seq token-SUM / horizon,
      then mean over sequences (length-UNbiased).
    - ``"seq-mean-token-mean"`` (ORIGINAL GRPO): per-seq token-MEAN, then mean
      over sequences (length-normalized, the standard-GRPO length bias).
    - ``"token-mean"`` (default): flat mean over all tokens.

    Args mirror the shared base; ``horizon`` is GRPO-only (the token-sum-norm
    denominator). See :meth:`ARStage.replay` for the ``sampling_temperature``
    contract (``logits / T`` so replay's log-softmax matches SGLang sampling).
    """

    def __init__(
        self,
        *,
        clip_range: float = 1e-4,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        **kwargs,
    ) -> None:
        super().__init__(clip_range=clip_range, loss_agg_mode=loss_agg_mode, **kwargs)
        self.horizon = int(horizon)

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
        adv_per_token = self._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            clip_range=clip_range,
            clip_range_high=clip_range_high,
        )

        if self.loss_agg_mode in ("seq-mean-token-sum-norm", "seq-mean-token-mean") and segment.lengths is not None:
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            if self.loss_agg_mode == "seq-mean-token-sum-norm":
                loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
            else:  # seq-mean-token-mean — guard 0-length responses (mean of empty = NaN)
                loss = torch.stack([p.mean() if p.numel() else p.new_zeros(()) for p in parts]).mean()
        else:
            loss = loss_per_elem.mean()
        return loss, ratio_metrics

    @staticmethod
    def _expand_advantages_to_tokens(
        advantages: torch.Tensor,
        lengths: torch.Tensor,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand per-sample ``advantages`` to per-token by repeating each
        sample's advantage across its ``lengths``-defined token span.
        """
        bs = int(advantages.shape[0])
        if int(lengths.shape[0]) != bs:
            raise ValueError(f"GRPO advantage expansion: advantages batch={bs} != lengths={int(lengths.shape[0])}")
        chunks: List[torch.Tensor] = []
        adv_cast = advantages.detach().to(dtype=dtype, device=device)
        for k in range(bs):
            n = int(lengths[k].item())
            if n > 0:
                chunks.append(adv_cast[k].expand(n))
        if not chunks:
            return torch.zeros(0, dtype=dtype, device=device)
        return torch.cat(chunks, dim=0)


__all__ = ["GRPO", "GRPOConfig"]
