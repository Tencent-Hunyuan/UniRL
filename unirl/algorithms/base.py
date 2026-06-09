"""Stage-driven algorithm base class.

The training-side contract for ``models`` pipelines: an algorithm
holds a stage (``DiffusionStage[C]`` or ``ARStage[C]``) and computes loss
over ``(conditions, segment, advantages)``. All model dispatch, CFG batching,
SDE math, autocast, and per-step iteration are owned by ``stage.replay(...)``;
the algorithm is pure ratio-clip math against the segment's stored log-probs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from unirl.distributed.group.remote import Remote

if TYPE_CHECKING:
    from unirl.types.conditions import Condition
    from unirl.types.segments.base import Segment


# ---------------------------------------------------------------------------
# Shared helpers used by FlowGRPO, FlowDPPO, DiffusionNFT
# ---------------------------------------------------------------------------


def typed_conditions(
    conditions: Mapping[str, "Condition"],
    conditions_cls: Optional[Type[Any]],
) -> Any:
    """Reconstruct the stage's typed conditions container from the dict shape.

    When ``conditions_cls`` is ``None`` (e.g. unit tests against a fake stage
    that accepts the dict directly), the dict is forwarded verbatim. Otherwise
    ``conditions_cls.from_dict(...)`` is invoked.
    """
    if conditions_cls is None:
        return conditions
    return conditions_cls.from_dict(dict(conditions))


def gather_sde_field(
    tensor: Optional[torch.Tensor],
    sde_indices: Optional[torch.Tensor],
    target_steps: List[int],
    *,
    field_name: str = "field",
) -> torch.Tensor:
    """Gather slices from a segment's SDE-aligned tensor by step index.

    Maps ``target_steps`` to positions in ``sde_indices`` via
    ``torch.searchsorted`` (O(S' log S)) and returns
    ``tensor[:, positions, ...]``.

    Used by GRPO (for ``sde_logp``) and FlowDPPO (for ``sde_logp`` + ``sde_means``).
    """
    if tensor is None or sde_indices is None:
        raise ValueError(
            f"gather_sde_field: {field_name} or sde_indices is None "
            f"(ensure prepare_segment ran before compute_loss_and_backward)."
        )
    target_t = torch.tensor(target_steps, dtype=sde_indices.dtype, device=sde_indices.device)
    # Ensure sde_indices is sorted (searchsorted requirement)
    sort_order = sde_indices.argsort()
    sde_indices = sde_indices[sort_order]
    tensor = tensor[:, sort_order.tolist()]
    positions = torch.searchsorted(sde_indices, target_t)
    # Clamp to valid range before validation (searchsorted can return len for out-of-range)
    positions = positions.clamp(max=sde_indices.shape[0] - 1)
    # Validate looked-up positions match
    if (sde_indices[positions] != target_t).any():
        bad = [int(t) for t, p in zip(target_steps, positions) if sde_indices[p] != t]
        raise ValueError(
            f"gather_sde_field({field_name}): target steps {bad} not in sde_indices={sde_indices.tolist()}"
        )
    return tensor[:, positions.tolist()]


def rollout_replay_logp_absdiff(new_logp: torch.Tensor, old_logp: torch.Tensor) -> Dict[str, float]:
    """Per-token |Δlogp| between rollout and replay — AR train-rollout drift gauge.

    ``old_logp`` is the rollout-time log-prob (SGLang / trainside autoregress)
    and ``new_logp`` is the teacher-forced replay at the current weights. On a
    single on-policy update the two differ only by the rollout-vs-replay *engine*
    gap (a temperature/logprob misconfig, a broken SGLang weight sync, or bf16
    KV-cache-vs-full-forward drift). ``mean|Δlogp|`` reports that gap directly and
    symmetrically — more legible than the exp-biased ``ratio_mean``. AR-only: the
    diffusion algorithms self-record or recompute ``old_logp`` with the same
    model, so their gap is ~0 by construction and they do not emit this metric.

    Assumes non-empty inputs, mirroring ``_grpo_clip_loss`` — the AR callers
    early-return on a zero-token segment before this runs.
    """
    with torch.no_grad():
        absdiff = (new_logp - old_logp).abs()
    return {
        "rollout_replay_logp_absdiff_mean": float(absdiff.mean()),
        "rollout_replay_logp_absdiff_max": float(absdiff.max()),
    }


def _resolve_clip_range_from_schedule(clip_range: float, schedule: str, progress: float) -> float:
    """Schedule-aware clip range. Mirrors ``GRPOAlgorithm.get_clip_range``."""
    if schedule == "linear_decay":
        return clip_range * (1.0 - 0.5 * float(progress))
    if schedule == "cosine_decay":
        return clip_range * (0.5 * (1.0 + math.cos(math.pi * float(progress))))
    return clip_range


def _grpo_clip_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    clip_range_high: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO-style clipped objective. Element-wise; reduction is the caller's job.

    ``clip_range`` is the lower clip ε⁻ (ratio floor ``1-clip_range``).
    ``clip_range_high`` (DAPO "clip-higher") is the upper ε⁺ (ratio ceil
    ``1+clip_range_high``); ``None`` ⇒ symmetric (= ``clip_range``), the prior
    behaviour, so ``FlowGRPO`` is unaffected.

    Returns ``(loss_per_element, ratio_metrics_dict)``. The metrics tensors are
    detached scalars suitable for logging.
    """
    high = clip_range if clip_range_high is None else clip_range_high
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + high)
    loss_per_elem = torch.maximum(unclipped, clipped)

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    gt = (ratio - 1.0 > high).float()
    lt = (1.0 - ratio > clip_range).float()
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "clip_fraction": torch.maximum(gt, lt).mean().detach(),
        "clipfrac_gt_one": gt.mean().detach(),
        "clipfrac_lt_one": lt.mean().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
    }
    return loss_per_elem, metrics


@dataclass(frozen=True)
class AlgorithmStepResult:
    """Result of one micro-step under the stage-driven contract.

    ``num_steps_or_tokens`` is the diffusion step count for diffusion
    algorithms or the trained-token count for AR algorithms.
    """

    loss: float
    metrics: Mapping[str, Any]
    num_steps_or_tokens: int
    has_backward: bool


class BaseAlgorithmConfig(ABC):
    """Marker base for all algorithm config dataclasses.

    Used as the type annotation / base class for the per-stage algorithm
    config dataclasses.
    """


class StageAlgorithm(Remote, ABC):
    """Pure (conditions, segment, advantages) → loss; holds its stage.

    Targets the four-tier pipeline contract (``models``). The algorithm
    holds a reference to a
    :class:`unirl.models.types.diffusion.DiffusionStage` or
    :class:`unirl.models.types.ar.ARStage` and dispatches all
    model forward / SDE / CFG work into ``stage.replay(...)``. It does not
    know its slot key in the dispatcher; slot routing lives on the train stack.

    Class attributes:
        requires_ema_rollout: Whether the algorithm requires EMA weights
            during rollout sampling. On-policy algorithms (GRPO) MUST
            sample with the same weights used in training replay so the
            importance ratio equals 1 on the first step (default False).
            Off-policy / forward-process algorithms (DiffusionNFT) override to
            True so the rollout uses EMA-smoothed weights for higher-
            quality trajectories.
        supports_multi_update: Whether the algorithm is correct under
            ``num_updates_per_batch > 1`` (the train stack splitting one
            rollout into N optimizer steps over disjoint mini-batches).
            True only when the PPO ``old_logp`` anchor stays frozen across all
            N steps: ``FlowGRPO`` / ``FlowDPPO`` capture
            ``segment.sde_logp`` once in :meth:`prepare_segment`; ``GRPO`` /
            ``DRPO`` keep the rollout log-prob as the anchor for all N steps
            (verl ``bypass_mode`` parity — the ratio then also carries the
            rollout-vs-train engine gap), and ``DRPO`` under
            ``old_logp_source='replay'`` instead freezes a train-side anchor in
            :meth:`prepare_segment`. Default False — e.g. DiffusionNFT's multi-update
            path is unvalidated, and anchor-free algorithms (SFT) have nothing
            to freeze. ``TrainStack`` raises when a False algorithm is paired
            with ``num_updates_per_batch > 1``.
    """

    requires_ema_rollout: bool = False
    supports_multi_update: bool = False
    # Segment fields this algorithm freezes as the π_old anchor in
    # :meth:`prepare_segment` (GRPO: ``("sde_logp",)``; FlowDPPO:
    # ``("sde_logp", "sde_means")``). When the anchor is recomputed
    # (:meth:`recomputes_anchor`), the train stack re-slices and reassembles
    # exactly these fields at train-time geometry — it never hardcodes them.
    anchor_fields: Tuple[str, ...] = ()

    def recomputes_anchor(self) -> bool:
        """Whether the anchor must be recomputed at the EXACT ``(mini, micro)``
        batch geometry training uses — not merely whether a replay happens.

        True ⇒ :meth:`prepare_segment` replays the anchor AND bf16 batch-shape
        sensitivity matters, so the train stack drives it per micro-slice over
        those exact slices; the old/new forwards then match bit-for-bit
        (on-policy ratio = 1; FlowDPPO on-policy KL = 0). FlowDPPO is always True
        (``sde_means`` exist only via replay); ``FlowGRPO`` is True only
        under ``old_logp_source='replay'``. False (default) ⇒ one full-segment
        call suffices: the anchor is either the engine's own emission (no
        replay), or a replay where coarse geometry is acceptable (ratio ≈ 1) —
        e.g. ``DRPO`` replay mode, whose production path is rollout-anchored,
        so the bf16 geometry term sits below the rollout-vs-train engine gap.
        """
        return False

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "Segment",
    ) -> None:
        """Optional pre-step hook called once before the multi-update loop.

        Default no-op. Algorithms with no π_old anchor to freeze (e.g. DiffusionNFT, SFT)
        can ignore the hook entirely.

        Algorithms that establish a frozen anchor override this. The canonical
        use case is :class:`FlowGRPO` / :class:`FlowDPPO`, which here
        set ``segment.sde_logp`` according to ``old_logp_source``: ``"rollout"``
        keeps the rollout engine's best-effort emission (raising if it emitted
        nothing); ``"replay"`` recomputes via a ``torch.no_grad``
        ``stage.replay`` and overwrites it. Because this hook fires ONCE per
        ``RolloutResp`` — before the trainer's ``num_updates_per_batch`` train
        loop — the anchor is frozen at pre-update weights across all N updates,
        matching the on-policy ratio semantics of PPO-style algorithms.

        Args:
            conditions: ``RolloutResp.tracks[slot].conditions`` — stage-typed conditions
                are reconstructed inside the algorithm if needed.
            segment: ``RolloutResp.tracks[slot].segment`` for this algorithm's
                slot. Implementations may mutate field defaults that were
                left ``None`` by the rollout (lazy initialization); they
                must NOT mutate fields that the rollout already populated.
        """
        return None

    @abstractmethod
    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "Segment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Compute loss for one micro-batch and call ``.backward()``.

        Args:
            conditions: ``RolloutResp.tracks[slot].conditions`` — stage-typed conditions
                are reconstructed inside the algorithm if needed.
            segment: ``RolloutResp.tracks[slot].segment`` — diffusion algorithms
                read ``segment.sde_logp`` / ``segment.sde_indices`` /
                ``segment.sigmas``; AR algorithms read ``segment.log_probs`` /
                ``segment.cu_seqlens``.
            advantages: per-sample advantage signal ``[B]``.
            training_progress: training progress in ``[0, 1]`` for
                clip-range or other schedules.
            loss_scale: gradient accumulation factor (typically
                ``1 / num_micro_batches``).
        """
        ...


__all__ = [
    "AlgorithmStepResult",
    "StageAlgorithm",
    "gather_sde_field",
    "rollout_replay_logp_absdiff",
    "typed_conditions",
]
