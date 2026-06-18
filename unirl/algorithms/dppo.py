"""DPPO (AR): Divergence-based Proximal Policy Optimization for token-level RL.

Implements **DPPO**, the method of Qi et al. "Rethinking the Trust Region in LLM
Reinforcement Learning" (arXiv:2602.04879), for autoregressive (token-level,
discrete) policies. DPPO is the **foundational** trust region that both shipped
team algorithms extend: :class:`~unirl.algorithms.cppo.CPPO` makes its threshold
position-weighted and cumulative (the **hard-mask** sibling), and
:class:`~unirl.algorithms.drpo.DRPO` smooths it into an advantage-weighted
quadratic regularizer (the **smooth-mask** sibling).

DPPO keeps GRPO's token-level ratio-advantage surrogate but replaces PPO's ratio
clipping with a **uniform per-token Binary-TV mask** (paper §3): the trust region
is ``|pi(y_t|s_t) - mu(y_t|s_t)| <= delta`` on the absolute probability shift of
the sampled token, which is better behaved than the importance ratio for a
long-tailed vocabulary (a rare token gives a huge ratio after a tiny probability
change). The threshold ``delta`` is the same for every token — unlike CPPO's
position-weighted, cumulative threshold — so the keep decision is purely
token-local (no prefix sums).

Per sampled token (``t`` is the token position within its sequence)::

    D_t = |pi(y_t|s_t) - mu(y_t|s_t)|                  (Binary-TV per-token divergence)
    r_t = pi(y_t|s_t) / mu(y_t|s_t)                    (importance ratio)
    keep token t  iff  A_t * (r_t - 1) <= 0   OR   D_t <= delta
    L = E[ sum_t keep_t * (-A_t * r_t) ]

The first clause always keeps updates that move ``pi`` back toward ``mu`` (i.e.
``A_t(r_t - 1) <= 0``); the threshold only restricts updates that move ``pi``
*farther* from ``mu``. This is exactly CPPO's keep rule (Eq. 10) with the
effective threshold collapsed to the constant ``delta`` (``w_min = 1`` and no
prefix budget), so DPPO is the natural reduction of CPPO.

``mu`` is the rollout-policy token probability: ``old_logp`` is the SGLang rollout
log-prob frozen on the segment (``old_logp_source='rollout'``), exactly the
behavior policy DPPO's trust region anchors on — the same rollout-anchored
``old_logp`` semantics as :class:`GRPO` / :class:`DRPO` / :class:`CPPO`.

(AR-only by design; DPPO targets discrete token-level policies. The diffusion/flow
analogue is :class:`~unirl.algorithms.flowdppo.FlowDPPO`.)
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
    rollout_replay_logp_absdiff,
    typed_conditions,
)
from .grpo import GRPO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DPPOConfig(BaseAlgorithmConfig):
    """Config for :class:`DPPO` (the paper's uniform Binary-TV trust region).

    Attributes:
        stage_attr: Which stage slot to bind to (``"ar"``).
        conditions_cls: Dotted path to the stage-typed conditions class.
        dppo_delta: Token-level Binary-TV threshold ``delta`` (paper §4 /
            CPPO Table 3: 0.15 for dense models). This is the uniform per-token
            trust-region scale — the same value for every token (CPPO's
            ``cppo_delta`` with the position weight and prefix budget removed).
        loss_agg_mode: ``"token-mean"`` or ``"seq-mean-token-sum-norm"``
            (per-seq token-SUM / horizon, then mean over sequences).
        horizon: Fixed length normalizer for ``seq-mean-token-sum-norm``
            (= max response length).
        sampling_temperature: Rollout sampling temperature; replay rescales
            logits by it so ``log_softmax(logits / T)`` matches the sampling
            distribution. MUST equal ``sampling.temperature``. Falls back to the
            :class:`ARSamplingParams` default when None.
        old_logp_source: ``"rollout"`` (default, the canonical DPPO mode) anchors
            the ratio and the Binary-TV divergence ``mu`` on the rollout engine's
            emitted logprobs for ALL ``num_updates_per_batch`` steps; ``"replay"``
            freezes a train-side ``pi_old`` in :meth:`prepare_segment` instead.
    """

    stage_attr: str = "ar"
    conditions_cls: str = ""
    # Paper §4 / CPPO Table 3: delta = clip_ratio (0.15 for dense models). Uniform
    # across tokens — DPPO's defining simplification vs CPPO's position-weighted,
    # cumulative-prefix threshold.
    dppo_delta: float = 0.15
    loss_agg_mode: str = "token-mean"
    horizon: int = 8192
    sampling_temperature: Optional[float] = None
    # "rollout" (default) keeps the rollout engine's emitted logprobs as mu for
    # ALL num_updates_per_batch steps (verl bypass-mode parity, = the behavior
    # policy DPPO anchors on); "replay" freezes a train-side pi_old at pre-update
    # weights in prepare_segment instead.
    old_logp_source: str = "rollout"


# ---------------------------------------------------------------------------
# Loss helper — AR (token-level)
# ---------------------------------------------------------------------------


def _dppo_mask(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """DPPO uniform Binary-TV keep-mask over packed-varlen ``[total_tokens]``.

    Built entirely under ``torch.no_grad`` by the caller — the mask is a
    trust-region gate, not part of the differentiable loss. Unlike CPPO, the
    decision is purely token-local (uniform threshold, no prefix sums), so there
    is no per-sequence loop: every token is kept iff its update moves ``pi`` back
    toward ``mu`` OR its absolute probability shift stays within ``delta``.

    Args:
        new_logp: New-policy (pi) log-probs at current weights. ``[total_tokens]``.
        old_logp: Rollout-policy (mu) log-probs, frozen. ``[total_tokens]``.
        advantages: Per-token advantages (expanded from per-sample). ``[total_tokens]``.
        ratio: ``exp(new_logp - old_logp)``, the importance ratio. ``[total_tokens]``.
        delta: Uniform token-level Binary-TV threshold (paper §3).

    Returns:
        ``keep`` mask ``[total_tokens]`` (float 0/1), detached.
    """
    # Compute the divergence in fp32 regardless of the logprob dtype: a bf16
    # subtraction of two near-equal probabilities loses the small shift to
    # rounding. The recipe pins logprob_precision=fp32, but this keeps DPPO
    # correct under a bf16-logprob config too.
    prob = torch.exp(new_logp.float())
    old_prob = torch.exp(old_logp.float())
    D_t = (prob - old_prob).abs()  # Binary-TV divergence D_t

    # Keep rule (= CPPO Eq. 10 with c_t collapsed to the constant delta):
    #  1. always keep updates that move pi back toward mu, and
    #  2. keep diverging updates only while D_t stays within the threshold.
    toward_mu = (advantages * (ratio - 1.0)) <= 0.0
    feasible = D_t <= delta
    keep = toward_mu | feasible
    return keep.detach().to(dtype=new_logp.dtype)


def _dppo_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    delta: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """DPPO per-token loss (paper §3; uniform Binary-TV hard mask).

    Operates on packed-varlen ``[total_tokens]`` tensors. Per kept token::

        L_t = -A_t * r_t        (ratio-advantage surrogate)

    with ``r_t = pi/mu`` kept differentiable (no ``.detach()``) so the gradient
    flows through the ratio, matching :func:`~unirl.algorithms.cppo._cppo_loss`
    and :func:`~unirl.algorithms.drpo._drpo_loss`. The DPPO mask zeroes the loss
    on tokens whose absolute probability shift exceeds the uniform threshold.

    Args:
        new_logp: New-policy (pi) log-probs at current weights. ``[total_tokens]``.
        old_logp: Rollout-policy (mu) log-probs, frozen. ``[total_tokens]``.
        advantages: Per-token advantages (expanded from per-sample).
        delta: Uniform token-level Binary-TV threshold (paper §3).

    Returns:
        ``(loss_per_element, metrics_dict)``. Reduction is the caller's job.
    """
    log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
    ratio = torch.exp(log_diff)  # r_t = pi/mu (differentiable through new_logp)
    adv = advantages.detach()

    # Trust-region gate (paper §3): no grad, it only decides which tokens train.
    with torch.no_grad():
        keep = _dppo_mask(
            new_logp=new_logp.detach(),
            old_logp=old_logp,
            advantages=adv,
            ratio=ratio.detach(),
            delta=delta,
        )

    pg_losses = -adv * ratio * keep
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": ((ratio - 1.0) - log_diff).mean().detach(),  # k3 estimator
        # Fraction of tokens zeroed by the Binary-TV mask (analogous to
        # DPPO/verl pg_clipfrac; the CPPO sibling's masked_fraction).
        "masked_fraction": (1.0 - keep).mean().detach(),
    }
    return pg_losses, metrics


# ---------------------------------------------------------------------------
# Algorithm class — AR (token-level)
# ---------------------------------------------------------------------------


class DPPO(StageAlgorithm):
    """DPPO for AR token-level policies — the foundational Binary-TV trust region.

    DPPO (Divergence-based Proximal Policy Optimization) keeps GRPO's token-level
    ratio-advantage surrogate but replaces PPO's ratio clipping with a **uniform
    per-token Binary-TV mask** (paper §3). Per kept token::

        L_t = -A_t * r_t ,    keep_t = (A_t (r_t - 1) <= 0) OR (|pi - mu| <= delta)

    This is the trust region that :class:`CPPO` makes position-weighted /
    cumulative and that :class:`DRPO` smooths into a quadratic regularizer.

    Args:
        pipeline: The trainer-injected pipeline; the stage is resolved from it via
            ``getattr(pipeline, stage_attr)``.
        stage: Pre-resolved stage (tests); ``pipeline`` is the production path.
        stage_attr: Which pipeline attribute holds the AR stage (``"ar"``).
        dppo_delta: Uniform token-level Binary-TV threshold ``delta`` (0.15 dense;
            paper §4 / CPPO Table 3).
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
    # GRPO / DRPO / CPPO. The ratio then also absorbs the rollout-vs-train engine
    # gap on later mini-batches (accepted for parity).
    supports_multi_update = True

    def __init__(
        self,
        *,
        pipeline: Any = None,
        stage: Any = None,
        stage_attr: str = "ar",
        dppo_delta: float = 0.15,
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        sampling_temperature: Optional[float] = None,
        old_logp_source: str = "rollout",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("DPPO: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        self.stage = stage
        self.dppo_delta = float(dppo_delta)
        if self.dppo_delta <= 0.0:
            raise ValueError(f"DPPO: dppo_delta must be > 0; got {self.dppo_delta}")
        self.loss_agg_mode = str(loss_agg_mode)
        self.horizon = int(horizon)
        # replay rescales logits by this temperature so its log-softmax matches
        # the rollout sampling distribution (log_softmax(logits / T)); MUST equal
        # sampling.temperature. Mirrors GRPO / DRPO / CPPO. Falls back to the
        # ARSamplingParams default when unset.
        if sampling_temperature is None:
            from unirl.types.sampling import ARSamplingParams

            sampling_temperature = ARSamplingParams.__dataclass_fields__["temperature"].default
        self.sampling_temperature = float(sampling_temperature)
        self.conditions_cls = conditions_cls
        # pi_old / mu source. "rollout" = the rollout engine's emitted
        # segment.log_probs, kept as the anchor for ALL num_updates_per_batch
        # steps (= the behavior policy DPPO's trust region anchors on; verl
        # bypass-mode parity). "replay" = prepare_segment recomputes a frozen
        # train-side pi_old at pre-update weights. Both anchors are frozen for the
        # whole rollout batch, so multi-update is supported in either mode.
        self.old_logp_source = str(old_logp_source).strip().lower()
        if self.old_logp_source not in ("rollout", "replay"):
            raise ValueError(f"DPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}")

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
          policy DPPO anchors on; verl bypass-mode parity).
        - ``"replay"``: recompute ``pi_old`` via a ``torch.no_grad``
          ``stage.replay`` at the **pre-update** weights and **overwrite**
          ``segment.log_probs``. Mirrors :meth:`CPPO.prepare_segment` replay mode.
        """
        if self.old_logp_source != "replay":
            return
        if segment.tokens is None or segment.log_probs is None or int(segment.tokens.shape[0]) == 0:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            frozen = self.stage.replay(typed_conds, segment=segment, temperature=self.sampling_temperature)
        # Keep the replay's native (fp32) precision so the anchor stays as close
        # as possible to new_logp's fp32 replay (mirrors CPPO / DRPO / FlowGRPO).
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

        loss_per_elem, ratio_metrics = _dppo_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            advantages=adv_per_token,
            delta=self.dppo_delta,
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
            "dppo_delta": self.dppo_delta,
            **rollout_replay_logp_absdiff(new_logp, old_logp),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=int(new_logp.shape[0]),
            has_backward=True,
        )


__all__ = ["DPPO", "DPPOConfig"]
