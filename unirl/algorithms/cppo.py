"""CPPO (AR): Cumulative Prefix-divergence Policy Optimization for token-level RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

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
class CPPOConfig(BaseAlgorithmConfig):
    """Config for :class:`CPPO` (the paper's CPPO Binary-TV method)."""

    stage_attr: str = "ar"
    conditions_cls: str = ""
    cppo_delta: float = 0.2
    cppo_w_min: float = 0.8
    cppo_delta_b: float = 0.02
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: Optional[float] = None
    old_logp_source: str = "rollout"


def _cppo_mask(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    lengths: torch.Tensor,
    delta: float,
    w_min: float,
    delta_b: float,
) -> torch.Tensor:
    """CPPO Binary-TV keep-mask over packed-varlen ``[total_tokens]`` tensors."""
    prob = torch.exp(new_logp.float())
    old_prob = torch.exp(old_logp.float())
    D_all = (prob - old_prob).abs()  # Binary-TV divergence D_t
    toward_mu = (advantages * (ratio - 1.0)) <= 0.0

    keep_parts: List[torch.Tensor] = []
    for D_t, toward_t in zip(
        torch.split(D_all, lengths.tolist()),
        torch.split(toward_mu, lengths.tolist()),
    ):
        T = int(D_t.shape[0])
        if T == 0:
            keep_parts.append(D_t.new_zeros(0, dtype=torch.bool))
            continue

        pos = torch.arange(1, T + 1, device=D_t.device, dtype=D_t.dtype)
        w_t = 1.0 - (1.0 - w_min) * (pos - 1) / max(T - 1, 1)
        Z_t = w_t * D_t

        S_prev = torch.cat([Z_t.new_zeros(1), torch.cumsum(Z_t, dim=0)[:-1]])
        W_prev = torch.cat([w_t.new_zeros(1), torch.cumsum(w_t, dim=0)[:-1]])

        p90 = torch.quantile(D_t, q=0.9)
        delta_b_seq = p90.clamp(min=delta_b, max=2.0 * delta_b)

        c_t = torch.minimum(
            torch.full_like(Z_t, delta),
            delta + delta_b_seq * W_prev - S_prev,
        )
        feasible = Z_t <= c_t
        keep_parts.append(toward_t | feasible)

    keep = torch.cat(keep_parts) if keep_parts else toward_mu
    return keep.detach().to(dtype=new_logp.dtype)


def _cppo_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    lengths: torch.Tensor,
    delta: float,
    w_min: float,
    delta_b: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """CPPO per-token loss over packed-varlen ``[total_tokens]`` (paper Eq. 8-11; Binary-TV hard mask)."""
    log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)
    adv = advantages.detach()

    with torch.no_grad():
        keep = _cppo_mask(
            new_logp=new_logp.detach(),
            old_logp=old_logp,
            advantages=adv,
            ratio=ratio.detach(),
            lengths=lengths,
            delta=delta,
            w_min=w_min,
            delta_b=delta_b,
        )

    pg_losses = -adv * ratio * keep
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": ((ratio - 1.0) - log_diff).mean().detach(),
        "masked_fraction": (1.0 - keep).mean().detach(),
    }
    return pg_losses, metrics


class CPPO(StageAlgorithm):
    """CPPO (Binary-TV) for AR token-level policies — the paper's proposed method."""

    supports_multi_update = True

    def __init__(
        self,
        *,
        pipeline: Any = None,
        stage: Any = None,
        stage_attr: str = "ar",
        cppo_delta: float = 0.2,
        cppo_w_min: float = 0.8,
        cppo_delta_b: float = 0.02,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        sampling_temperature: Optional[float] = None,
        old_logp_source: str = "rollout",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("CPPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.cppo_delta = float(cppo_delta)
        self.cppo_w_min = float(cppo_w_min)
        self.cppo_delta_b = float(cppo_delta_b)
        if not 0.0 < self.cppo_w_min <= 1.0:
            raise ValueError(f"CPPO: cppo_w_min must be in (0, 1]; got {self.cppo_w_min}")
        if self.cppo_delta_b < 0.0:
            raise ValueError(f"CPPO: cppo_delta_b must be >= 0; got {self.cppo_delta_b}")
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.conditions_cls = conditions_cls
        self.old_logp_source = str(old_logp_source).strip().lower()
        if self.old_logp_source not in ("rollout", "replay"):
            raise ValueError(f"CPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}")

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the ``pi_old`` / ``mu`` anchor before the ``num_updates_per_batch`` loop, per ``old_logp_source``."""
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

        lengths = segment.lengths.to(device=new_logp.device)
        loss_per_elem, ratio_metrics = _cppo_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            lengths=lengths,
            delta=self.cppo_delta,
            w_min=self.cppo_w_min,
            delta_b=self.cppo_delta_b,
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
            "cppo_delta": self.cppo_delta,
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["CPPO", "CPPOConfig"]
