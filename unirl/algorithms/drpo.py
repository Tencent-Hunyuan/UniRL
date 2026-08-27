"""DRPO (AR): Divergence Regularized Policy Optimization for token-level RL."""

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
    rollout_replay_logp_absdiff,
    typed_conditions,
)
from .grpo import GRPO


@dataclass
class DRPOConfig(BaseAlgorithmConfig):
    """Config for :class:`DRPO` (the paper's DRPO method, §3)."""

    stage_attr: str = "ar"
    conditions_cls: str = ""
    drpo_epsilon: float = 12.5
    penalty_mu_weighted: bool = True
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: Optional[float] = None
    old_logp_source: str = "rollout"


def _drpo_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    epsilon: float,
    mu_weighted: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DRPO per-token loss over packed-varlen ``[total_tokens]`` (paper §3; gradient Eq 9, weight Table 1)."""
    log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    old_prob = torch.exp(old_logp).detach()

    ratio_delta = ratio - 1.0
    if mu_weighted:
        penalty_weight = old_prob
        adaptive_eps = torch.where(old_prob > 0.0, epsilon / old_prob, torch.full_like(old_prob, float("inf")))
    else:
        penalty_weight = torch.ones_like(old_prob)
        adaptive_eps = torch.full_like(old_prob, epsilon)
    quadratic_penalty = adv.abs() * penalty_weight * ratio_delta.square() / (2.0 * epsilon)
    pg_losses = -adv * ratio + quadratic_penalty
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": ((ratio - 1.0) - log_diff).mean().detach(),
        "drpo_penalty_mean": quadratic_penalty.mean().detach(),
        "clipfrac_upper": (ratio > (1.0 + adaptive_eps)).float().mean().detach(),
        "clipfrac_lower": (ratio < (1.0 - adaptive_eps)).float().mean().detach(),
    }
    return pg_losses, metrics


class DRPO(StageAlgorithm):
    """DRPO for AR token-level policies — the paper's proposed method (§3)."""

    def __init__(
        self,
        *,
        pipeline: Any = None,
        stage_attr: str = "ar",
        drpo_epsilon: float = 12.5,
        penalty_mu_weighted: bool = True,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        sampling_temperature: Optional[float] = None,
        old_logp_source: str = "rollout",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if pipeline is None:
            raise ValueError("DRPO: `pipeline` must be provided (the v2 trainer injects it)")
        self.stage = getattr(pipeline, stage_attr)
        self.drpo_epsilon = float(drpo_epsilon)
        self.penalty_mu_weighted = bool(penalty_mu_weighted)
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.conditions_cls = conditions_cls
        self.old_logp_source = str(old_logp_source).strip().lower()
        if self.old_logp_source not in ("rollout", "replay"):
            raise ValueError(f"DRPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}")
        self.supports_multi_update = True

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the π_old anchor (``segment.log_probs``) before the ``num_updates_per_batch`` loop."""
        if self.old_logp_source != "replay":
            return
        if segment.tokens is None or segment.log_probs is None or int(segment.tokens.shape[0]) == 0:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            frozen = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
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

        adv_per_token = GRPO._expand_advantages_to_tokens(
            advantages, segment.lengths, dtype=new_logp.dtype, device=new_logp.device
        )

        loss_per_elem, ratio_metrics = _drpo_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            epsilon=self.drpo_epsilon,
            mu_weighted=self.penalty_mu_weighted,
        )

        if segment.loss_mask is not None:
            mask = segment.loss_mask.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
            loss_per_elem = loss_per_elem * mask

        if self.loss_agg_mode == "seq-mean-token-sum-norm" and segment.lengths is not None:
            parts = torch.split(loss_per_elem, segment.lengths.tolist())
            loss = torch.stack([p.sum() for p in parts]).mean() / float(self.horizon)
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "drpo_epsilon": self.drpo_epsilon,
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["DRPO", "DRPOConfig"]
