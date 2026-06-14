"""CPPO (AR): Cumulative Prefix-divergence Policy Optimization for token-level RL.

Implements **CPPO** (Binary-TV variant), the proposed method of "Beyond Uniform
Token-Level Trust Region in LLM Reinforcement Learning" (arXiv:2606.10968), for
autoregressive (token-level, discrete) policies.

CPPO keeps DPPO's token-level ratio-advantage surrogate but replaces DPPO's
*uniform* per-token Binary-TV threshold with a **position-weighted token-level
threshold and a cumulative prefix budget** (paper Eq. 8-11 + Algorithm 1). Only
the trust-region *mask* changes — the loss term is the DPPO one
(``L = E[sum_t keep_t * rho_t * A_t]``), so CPPO is the **hard-mask** sibling of
the same DPPO Binary-TV lineage that :class:`~unirl.algorithms.drpo.DRPO`
smooths into a quadratic regularizer.

Per response of length ``T`` (``t`` is the 1-based token position; in UniRL's
packed-varlen layout ``T`` is each sequence's own length — there is no padding,
so this matches the paper's per-sequence ``T`` in Eq. 9)::

    D_t = |pi(y_t|s_t) - mu(y_t|s_t)|              (Binary-TV per-token divergence)
    w_t = w_min + (1 - w_min) * (T - t) / (T - 1)  (decreasing position weight in [w_min, 1])
    Z_t = w_t * D_t
    S_t = sum_{j<=t} Z_j ,  W_t = sum_{j<=t} w_j    (prefix sums; S_0 = W_0 = 0)
    c_t = min(delta, delta + delta_b^seq * W_{t-1} - S_{t-1})   (effective threshold, Eq. 8)
    keep token t  iff  A_t * (rho_t - 1) <= 0   OR   Z_t <= c_t      (Eq. 10)

The first clause always keeps updates that move ``pi`` back toward ``mu``; the
budget only restricts updates that move ``pi`` farther from ``mu``. The
prefix-average budget ``delta_b`` is calibrated per sequence from its own
divergence statistics (paper Eq. 22, Base-model warm-up calibration)::

    delta_b^seq = clamp(P90(D_{1:T}), delta_b_min, 2 * delta_b_min)

where ``delta_b_min`` is ``cppo_delta_b``.

``mu`` is the rollout-policy token probability: ``old_logp`` is the SGLang
rollout log-prob frozen on the segment (``old_logp_source='rollout'``), exactly
the behavior policy CPPO's trust region anchors on (paper Sec. 4) — the same
rollout-anchored ``old_logp`` semantics as :class:`GRPO` / :class:`DRPO`.

(AR-only by design; CPPO targets discrete token-level policies.)
"""

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CPPOConfig(BaseAlgorithmConfig):
    """Config for :class:`CPPO` (the paper's CPPO Binary-TV method).

    Attributes:
        stage_attr: Which stage slot to bind to (``"ar"``).
        conditions_cls: Dotted path to the stage-typed conditions class.
        cppo_delta: Token-level Binary-TV threshold ``delta`` (paper Sec. 4 /
            Table 3: 0.15 for dense models, 0.20 for the 30B-A3B MoE model). This
            is DPPO's per-token trust-region scale; CPPO additionally tightens it
            with the cumulative prefix budget below.
        cppo_w_min: Floor ``w_min`` of the linear position-weight schedule
            ``w_t = w_min + (1 - w_min) * (T - t) / (T - 1)`` (paper default 0.8).
            Earlier tokens get weight 1, late tokens get ``w_min``.
        cppo_delta_b: Floor ``delta_b_min`` of the per-sequence dynamic
            prefix-average budget (paper Eq. 22; default 0.02).
        loss_agg_mode: ``"token-mean"`` or ``"seq-mean-token-sum-norm"``
            (per-seq token-SUM / horizon, then mean over sequences).
        horizon: Fixed length normalizer for ``seq-mean-token-sum-norm``
            (= max response length).
        sampling_temperature: Rollout sampling temperature; replay rescales
            logits by it so ``log_softmax(logits / T)`` matches the sampling
            distribution. MUST equal ``sampling.temperature``. Falls back to the
            :class:`ARSamplingParams` default when None.
        old_logp_source: ``"rollout"`` (default, the canonical CPPO mode) anchors
            the ratio and the Binary-TV divergence ``mu`` on the rollout engine's
            emitted logprobs for ALL ``num_updates_per_batch`` steps; ``"replay"``
            freezes a train-side ``pi_old`` in :meth:`prepare_segment` instead.
    """

    stage_attr: str = "ar"
    conditions_cls: str = ""
    # Paper Sec. 4 / Table 3: delta = clip_ratio (0.15 dense, 0.20 for 30B-A3B).
    cppo_delta: float = 0.2
    # Position-weight floor (paper default 0.8) and dynamic prefix-budget floor
    # (paper Eq. 22 default 0.02).
    cppo_w_min: float = 0.8
    cppo_delta_b: float = 0.02
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: Optional[float] = None
    # "rollout" (default) keeps the rollout engine's emitted logprobs as mu for
    # ALL num_updates_per_batch steps (verl bypass-mode parity, = the behavior
    # policy CPPO anchors on); "replay" freezes a train-side pi_old at pre-update
    # weights in prepare_segment instead.
    old_logp_source: str = "rollout"


# ---------------------------------------------------------------------------
# Loss helper — AR (token-level)
# ---------------------------------------------------------------------------


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
    """CPPO Binary-TV keep-mask over packed-varlen ``[total_tokens]`` tensors.

    Built entirely under ``torch.no_grad`` by the caller — the mask is a
    trust-region gate, not part of the differentiable loss. Each packed sequence
    is processed independently (the prefix sums must NOT bleed across packed
    boundaries), so the per-sequence position weight keys on that sequence's own
    length ``T`` (no padding in this layout, unlike the 2D right-padded reference).

    Args:
        new_logp: New-policy (pi) log-probs at current weights. ``[total_tokens]``.
        old_logp: Rollout-policy (mu) log-probs, frozen. ``[total_tokens]``.
        advantages: Per-token advantages (expanded from per-sample). ``[total_tokens]``.
        ratio: ``exp(new_logp - old_logp)``, the importance ratio. ``[total_tokens]``.
        lengths: Per-sequence token counts; sums to ``total_tokens``.
        delta: Token-level Binary-TV threshold (paper Sec. 4).
        w_min: Position-weight floor (paper default 0.8).
        delta_b: Dynamic prefix-budget floor (paper Eq. 22).

    Returns:
        ``keep`` mask ``[total_tokens]`` (float 0/1), detached.
    """
    # The mask is a no-grad gate, so compute the divergence/prefix arithmetic in
    # fp32 regardless of the logprob dtype: torch.quantile rejects bf16/fp16, and
    # a bf16 cumsum over a long response (up to max_new_tokens) loses the prefix
    # sum to rounding. The recipe pins logprob_precision=fp32, but this keeps CPPO
    # correct under a bf16-logprob config too (where DRPO has no cumsum/quantile).
    prob = torch.exp(new_logp.float())
    old_prob = torch.exp(old_logp.float())
    D_all = (prob - old_prob).abs()  # Binary-TV divergence D_t
    # Eq. 10 first clause: always keep updates that move pi back toward mu.
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

        # Decreasing position weight w_t in [w_min, 1] over this sequence's own
        # length T (paper Eq. 9); t is 1-based, so w_1 = 1 and w_T = w_min. Use the
        # paper's (t - 1) numerator so a single-token response (T == 1) keeps its
        # lone first token at w_1 = 1; for T >= 2 this equals w_min + (1-w_min)(T-t)/(T-1).
        pos = torch.arange(1, T + 1, device=D_t.device, dtype=D_t.dtype)
        w_t = 1.0 - (1.0 - w_min) * (pos - 1) / max(T - 1, 1)
        Z_t = w_t * D_t

        # Prefix sums with a one-token right shift so the decision at t uses only
        # the preceding prefix (S_{t-1}, W_{t-1}); S_0 = W_0 = 0 (Appendix D).
        S_prev = torch.cat([Z_t.new_zeros(1), torch.cumsum(Z_t, dim=0)[:-1]])
        W_prev = torch.cat([w_t.new_zeros(1), torch.cumsum(w_t, dim=0)[:-1]])

        # Per-sequence dynamic prefix budget (Eq. 22):
        #   delta_b^seq = clamp(P90(D_{1:T}), delta_b, 2 * delta_b).
        p90 = torch.quantile(D_t, q=0.9)
        delta_b_seq = p90.clamp(min=delta_b, max=2.0 * delta_b)

        # Effective threshold c_t = min(delta, delta + delta_b^seq * W_{t-1} - S_{t-1}) (Eq. 8).
        c_t = torch.minimum(
            torch.full_like(Z_t, delta),
            delta + delta_b_seq * W_prev - S_prev,
        )
        feasible = Z_t <= c_t  # budget feasibility
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
    """CPPO per-token loss (paper Eq. 8-11; Binary-TV hard mask).

    Operates on packed-varlen ``[total_tokens]`` tensors. Per kept token::

        L_t = -A_t * r_t        (DPPO ratio-advantage surrogate)

    with ``r_t = pi/mu`` kept differentiable (no ``.detach()``) so the gradient
    flows through the ratio, matching :func:`~unirl.algorithms.base._grpo_clip_loss`
    and :func:`~unirl.algorithms.drpo._drpo_loss`. The CPPO mask zeroes the loss on
    tokens rejected by the position-weighted cumulative-prefix budget (Eq. 10).

    Args:
        new_logp: New-policy (pi) log-probs at current weights. ``[total_tokens]``.
        old_logp: Rollout-policy (mu) log-probs, frozen. ``[total_tokens]``.
        advantages: Per-token advantages (expanded from per-sample).
        lengths: Per-sequence token counts; sums to ``total_tokens``.
        delta: Token-level Binary-TV threshold (paper Sec. 4).
        w_min: Position-weight floor (paper default 0.8).
        delta_b: Dynamic prefix-budget floor (paper Eq. 22).

    Returns:
        ``(loss_per_element, metrics_dict)``. Reduction is the caller's job.
    """
    log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)  # r_t = pi/mu (differentiable through new_logp)
    adv = advantages.detach()

    # Trust-region gate (Eq. 10): no grad, it only decides which tokens train.
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
        "approx_kl": ((ratio - 1.0) - log_diff).mean().detach(),  # k3 estimator
        # Fraction of tokens rejected by the CPPO budget mask (paper Fig. 7
        # budget-mask share; analogous to DPPO/verl pg_clipfrac).
        "masked_fraction": (1.0 - keep).mean().detach(),
    }
    return pg_losses, metrics


# ---------------------------------------------------------------------------
# Algorithm class — AR (token-level)
# ---------------------------------------------------------------------------


class CPPO(StageAlgorithm):
    """CPPO (Binary-TV) for AR token-level policies — the paper's proposed method.

    CPPO (Cumulative Prefix-divergence Policy Optimization) keeps DPPO's
    token-level ratio-advantage surrogate but replaces DPPO's uniform Binary-TV
    threshold with a **position-weighted threshold + cumulative prefix budget**
    (paper Eq. 8-11). Only the trust-region mask changes; the loss term is DPPO's
    ``L = E[sum_t keep_t * (-A_t * r_t)]``.

    Args:
        pipeline: The trainer-injected pipeline; the stage is resolved from it via
            ``getattr(pipeline, stage_attr)``.
        stage_attr: Which pipeline attribute holds the AR stage (``"ar"``).
        cppo_delta: Token-level Binary-TV threshold ``delta`` (0.15 dense /
            0.20 for 30B-A3B; paper Table 3).
        cppo_w_min: Position-weight floor (paper default 0.8).
        cppo_delta_b: Dynamic prefix-budget floor (paper Eq. 22; default 0.02).
        loss_agg_mode: ``"token-mean"`` or ``"seq-mean-token-sum-norm"``.
        horizon: Fixed length normalizer for ``seq-mean-token-sum-norm``.
        sampling_temperature: Rollout sampling temperature, passed to
            ``stage.replay`` so its log-softmax matches the sampling distribution.
            MUST equal ``sampling.temperature``.
        old_logp_source: ``"rollout"`` (default) or ``"replay"``.
        conditions_cls: Stage-typed conditions container.
    """

    # old_logp (the ratio denominator and the Binary-TV mu) is the rollout
    # (SGLang) log-prob, frozen on the segment and unchanged across mini-batch
    # updates — so reusing it across num_updates_per_batch>1 is the deliberate
    # rollout-anchored trust region (verl bypass_mode=True parity), matching
    # GRPO / DRPO. The ratio then also absorbs the rollout-vs-train engine gap on
    # later mini-batches (accepted for parity).
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
        # replay rescales logits by this temperature so its log-softmax matches
        # the rollout sampling distribution (log_softmax(logits / T)); MUST equal
        # sampling.temperature. Mirrors GRPO / DRPO. Falls back to the
        # ARSamplingParams default when unset.
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.conditions_cls = conditions_cls
        # pi_old / mu source. "rollout" = the rollout engine's emitted
        # segment.log_probs, kept as the anchor for ALL num_updates_per_batch
        # steps (= the behavior policy CPPO's trust region anchors on; verl
        # bypass-mode parity). "replay" = prepare_segment recomputes a frozen
        # train-side pi_old at pre-update weights. Both anchors are frozen for the
        # whole rollout batch, so multi-update is supported in either mode.
        self.old_logp_source = str(old_logp_source).strip().lower()
        if self.old_logp_source not in ("rollout", "replay"):
            raise ValueError(f"CPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}")

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> None:
        """Freeze the ``pi_old`` / ``mu`` anchor (``segment.log_probs``) before the
        ``num_updates_per_batch`` loop, per ``old_logp_source``.

        - ``"rollout"`` (default): keep the rollout engine's emitted
          ``segment.log_probs`` as the anchor for ALL N updates (the behavior
          policy CPPO anchors on; verl bypass-mode parity).
        - ``"replay"``: recompute ``pi_old`` via a ``torch.no_grad``
          ``stage.replay`` at the **pre-update** weights and **overwrite**
          ``segment.log_probs``. Mirrors :meth:`DRPO.prepare_segment` replay mode.
        """
        if self.old_logp_source != "replay":
            return
        if segment.tokens is None or segment.log_probs is None or int(segment.tokens.shape[0]) == 0:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            frozen = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        # Keep the replay's native (fp32) precision so the anchor stays as close
        # as possible to new_logp's fp32 replay (mirrors DRPO / FlowGRPO).
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
        new_logp = self.stage.replay(
            typed_conds, segment=segment, temperature=self.sampling_temperature
        )  # [total_tokens]
        # old_logp = the frozen pi_old / mu anchor (segment.log_probs). "rollout"
        # (default) keeps the rollout engine's logp; "replay" means
        # prepare_segment already overwrote it with a frozen train-side replay.
        old_logp = segment.log_probs.to(dtype=new_logp.dtype, device=new_logp.device)

        # Expand per-sample advantages to per-token.
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

        # Apply loss_mask if present (token-level masking for padding/eos).
        if segment.loss_mask is not None:
            mask = segment.loss_mask.to(dtype=loss_per_elem.dtype, device=loss_per_elem.device)
            loss_per_elem = loss_per_elem * mask

        if self.loss_agg_mode == "seq-mean-token-sum-norm" and segment.lengths is not None:
            # Reference seq-mean-token-sum-norm: per-sequence token-SUM / horizon,
            # then mean over the micro-batch's sequences. The stack's
            # loss_scale=1/num_micros then averages across micro-batches.
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
