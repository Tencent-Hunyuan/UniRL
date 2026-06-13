"""Family-agnostic single-stage train stack.

:class:`TrainStack` wraps one :class:`FSDPBackend` (training state: model +
optimizer + scheduler + EMA) and one :class:`StageAlgorithm` (loss + backward
against the bundle's trainable module) into a single-stage training driver. One
stack = one training track.

It owns the entire family-agnostic pipeline — device alignment, the π_old anchor
freeze, the per-update micro-accumulation loop, EMA, metrics — and defers exactly
ONE decision to an injected :class:`MicroPlanner` (composition, not inheritance):
how each update's samples are grouped into micro-batches. :class:`CountPlanner`
(the default) groups by fixed count; :class:`TokenBudgetPlanner` packs by token
budget. Swapping the strategy is a recipe-level ``micro_planner`` block, no subclass.

Sequencing per :meth:`train_track` call (one rollout)::

    track, plans = micro_planner.arrange(track)  # reorder (if packing) + plan
    prepare_segment(track, plans)                # once: freeze the π_old anchor
    for micros in plans:                         # num_updates_per_batch updates
        _run_update(track, micros=micros)        # one optimizer step each
    on_rollout_end()                             # once: EMA / rollout boundary

**Sort-then-slice.** Variable-length packing wants to group samples of similar
length, which would normally force arbitrary index lists threaded through the
whole pipeline. Instead the planner *reorders the track once up front* (length-sort
within each update, see :meth:`TokenBudgetPlanner.arrange`) so every micro is again
a **contiguous** ``(start, end)`` range — exactly the count-based geometry. The
stack therefore only ever slices, and the anchor reassembly is a plain ordered
``cat``; all packing-specific logic is isolated to the planner's one ``arrange``
call (a no-op for :class:`CountPlanner`).

``num_updates_per_batch`` partitions the rollout batch into that many disjoint
updates and runs one optimizer step per update — the FlowGRPO / DanceGRPO
schedule. Because ``prepare_segment`` captures the pre-update policy once, every
update shares the same PPO anchor; this is only correct for algorithms with
``supports_multi_update`` (the ctor enforces it). Defaults to 1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

from unirl.algorithms import AlgorithmStepResult, StageAlgorithm
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.distributed.tensor.batch import _move_value
from unirl.train.backend.fsdp import FSDPBackend
from unirl.types.rollout_resp import RolloutTrack
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Plan types
# --------------------------------------------------------------------------- #
# A plan is built before any forward runs:
#     Plan       = List[UpdatePlan]   # the whole rollout shard
#     UpdatePlan = List[Range]        # one optimizer step (= one "update")
#     Range      = (start, end)       # one micro's CONTIGUOUS sample membership
# Because packing reorders the track up front (sort-then-slice), a micro is ALWAYS
# a contiguous range — no index lists.
Range = Tuple[int, int]
UpdatePlan = List[Range]
Plan = List[UpdatePlan]


def _positive_int(*, name: str, value: object) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1. Got {resolved}.")
    return resolved


def _update_ranges(*, total_size: int, num_updates: int) -> Tuple[Range, ...]:
    """Partition ``[0, total_size)`` into ``num_updates`` equal contiguous updates.

    One range = one optimizer step. Even divisibility is required: the per-worker
    batch is fixed (DP sharding is even) and a ragged final update would silently
    drop samples and desync grad accumulation across DP ranks.
    """
    total = _positive_int(name="total_size", value=total_size)
    n = _positive_int(name="num_updates_per_batch", value=num_updates)
    if total % n != 0:
        raise ValueError(
            f"num_updates_per_batch={n} must evenly divide the per-worker batch "
            f"size ({total}); got remainder {total % n}. Adjust batch_size, "
            f"samples_per_prompt, or num_updates_per_batch."
        )
    update_size = total // n
    return tuple((i * update_size, (i + 1) * update_size) for i in range(n))


def _build_micro_batch_slices(
    *,
    total_size: int,
    micro_batch_size: int,
) -> Tuple[Range, ...]:
    """Contiguous fixed-count micro ranges over ``[0, total_size)``."""
    resolved_total_size = _positive_int(name="total_size", value=total_size)
    resolved_micro_batch_size = _positive_int(name="micro_batch_size", value=micro_batch_size)
    slices: List[Range] = []
    start = 0
    while start < resolved_total_size:
        end = min(start + resolved_micro_batch_size, resolved_total_size)
        slices.append((start, end))
        start = end
    return tuple(slices)


def _count_plan(*, total: int, num_updates: int, micro_batch_size: int) -> Plan:
    """Fixed-count plan: contiguous equal updates, each split into ``micro_batch_size`` micros.

    The diffusion / FlowGRPO "batched" schedule, and the fallback for the LLM path
    when a segment exposes no per-sample lengths. No collective: every DP rank
    produces the same update/micro counts because the per-rank batch is evenly
    sharded.
    """
    plan: Plan = []
    for u_start, u_end in _update_ranges(total_size=total, num_updates=num_updates):
        plan.append(
            [
                (u_start + ms, u_start + me)
                for ms, me in _build_micro_batch_slices(total_size=u_end - u_start, micro_batch_size=micro_batch_size)
            ]
        )
    return plan


# --------------------------------------------------------------------------- #
# Token-budget bin-packing (verl ppo_max_token_len_per_gpu parity)
# --------------------------------------------------------------------------- #
def _pack_micros_sum(
    *,
    indices: Sequence[int],
    lengths: Sequence[int],
    token_budget: int,
) -> List[List[int]]:
    """Sum-cost packing for a PACKED (varlen) replay: a micro's compute cost is
    the plain sum of its sequences' real tokens (no padding exists), exactly
    verl's ppo_max_token_len_per_gpu accounting. First-fit decreasing.
    """
    if int(token_budget) < 1:
        raise ValueError(f"token_budget must be >= 1; got {token_budget}")
    order = sorted(indices, key=lambda i: (-int(lengths[i]), i))
    bins: List[List[int]] = []
    bin_sum: List[int] = []
    for i in order:
        length = max(1, int(lengths[i]))
        placed = False
        for b in range(len(bins)):
            if bin_sum[b] + length <= token_budget:
                bins[b].append(i)
                bin_sum[b] += length
                placed = True
                break
        if not placed:
            bins.append([i])
            bin_sum.append(length)
    return bins


def _pack_micros_2d(
    *,
    indices: Sequence[int],
    prompt_lens: Sequence[int],
    resp_lens: Sequence[int],
    token_budget: int,
) -> List[List[int]]:
    """2D dense-cost packing: the replay forwards ``[B, P_max + T_max]`` where the
    prompt and response blocks pad SEPARATELY to their own in-micro maxes — so the
    true compute cost of a bin is ``(max(prompt) + max(resp)) * count``, not
    ``max(total) * count``. Packing on total length lets anti-correlated rows
    (long-prompt/short-resp with short-prompt/long-resp) blow up both pads at
    once. First-fit decreasing on total length with the exact 2D cost check.
    """
    if int(token_budget) < 1:
        raise ValueError(f"token_budget must be >= 1; got {token_budget}")
    order = sorted(indices, key=lambda i: (-(int(prompt_lens[i]) + int(resp_lens[i])), i))
    bins: List[List[int]] = []
    bin_pmax: List[int] = []
    bin_tmax: List[int] = []
    for i in order:
        p = max(0, int(prompt_lens[i]))
        t = max(1, int(resp_lens[i]))
        placed = False
        for b in range(len(bins)):
            cost = (max(bin_pmax[b], p) + max(bin_tmax[b], t)) * (len(bins[b]) + 1)
            if cost <= token_budget:
                bins[b].append(i)
                bin_pmax[b] = max(bin_pmax[b], p)
                bin_tmax[b] = max(bin_tmax[b], t)
                placed = True
                break
        if not placed:
            bins.append([i])
            bin_pmax.append(p)
            bin_tmax.append(t)
    return bins


def _pack_micros(
    *,
    indices: Sequence[int],
    lengths: Sequence[int],
    token_budget: int,
) -> List[List[int]]:
    """Pack ``indices`` into micro-batches under a dense-compute token budget.

    verl-style token-budget micro-batching (``ppo_max_token_len_per_gpu``) adapted
    to a DENSE-padded replay: a micro's compute cost is ``max_len_in_micro *
    batch_size`` (every row pads to the micro max), so the bin constraint is
    ``max_len * (count + 1) <= token_budget``. First-fit decreasing on length keeps
    rows of similar length together, which makes the dense pad waste ~0 without
    needing varlen attention. A sequence longer than the whole budget still gets
    its own micro (never dropped).
    """
    if int(token_budget) < 1:
        raise ValueError(f"token_budget must be >= 1; got {token_budget}")
    order = sorted(indices, key=lambda i: (-int(lengths[i]), i))
    bins: List[List[int]] = []
    bin_max: List[int] = []
    for i in order:
        length = max(1, int(lengths[i]))
        placed = False
        for b in range(len(bins)):
            new_max = max(bin_max[b], length)
            if new_max * (len(bins[b]) + 1) <= token_budget:
                bins[b].append(i)
                bin_max[b] = new_max
                placed = True
                break
        if not placed:
            bins.append([i])
            bin_max.append(length)
    return bins


def _partition_into_k(
    *,
    indices: Sequence[int],
    lengths: Sequence[int],
    k: int,
) -> List[List[int]]:
    """Partition ``indices`` into EXACTLY ``k`` non-empty bins, balancing dense cost.

    Used to equalize the micro-batch COUNT across DP ranks (see
    :func:`_arrange_packed`): FSDP forward/backward issues collectives per micro,
    so every rank must run the same number of micros or NCCL deadlocks. Greedy LPT:
    longest-first, each item goes to the bin whose dense cost (``max_len * count``)
    stays smallest after insertion; the first ``k`` items seed the bins so none is
    empty. May exceed the token budget — the budget is a throughput hint, parity is
    a correctness requirement.
    """
    if k < 1 or k > len(indices):
        raise ValueError(f"_partition_into_k: k={k} out of range for {len(indices)} samples")
    order = sorted(indices, key=lambda i: (-int(lengths[i]), i))
    bins: List[List[int]] = [[order[j]] for j in range(k)]
    bin_max: List[int] = [max(1, int(lengths[order[j]])) for j in range(k)]
    for i in order[k:]:
        length = max(1, int(lengths[i]))
        best, best_cost = 0, None
        for b in range(k):
            cost = max(bin_max[b], length) * (len(bins[b]) + 1)
            if best_cost is None or cost < best_cost:
                best, best_cost = b, cost
        bins[best].append(i)
        bin_max[best] = max(bin_max[best], length)
    return bins


def _sync_micro_count(local_count: int) -> int:
    """All-reduce(MAX) the per-rank micro count over the default process group.

    Token-budget packing depends on each rank's local sequence lengths, so the
    natural bin count differs across DP ranks — but FSDP collectives require
    micro-count parity (see :func:`_partition_into_k`). No-op (returns the local
    count) when torch.distributed is not initialized or world_size == 1.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return local_count
    t = torch.tensor([int(local_count)], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def _arrange_packed(
    resp_track: RolloutTrack,
    *,
    num_updates: int,
    token_budget: int,
    cost_model: str,
) -> Optional[Tuple[List[int], Plan]]:
    """Token-budget packed arrangement (verl ``ppo_max_token_len_per_gpu`` parity).

    Returns ``(perm, plan)`` where ``perm`` is a permutation of ``[0, total)`` that
    length-sorts each update's samples and ``plan`` is the resulting CONTIGUOUS
    range plan over the permuted track — so the caller applies ``track.select(perm)``
    once and everything downstream slices contiguously (sort-then-slice). Returns
    ``None`` when the segment exposes no per-sample ``lengths`` so the caller can
    fall back to a count plan.

    The bins from the FFD packers are index groups; concatenating them in order
    *is* the permutation, and each bin's size *is* a contiguous range in that
    permuted order. Update membership is unchanged — each update contributes its own
    ``[u_start, u_end)`` indices to ``perm`` in order, so the permuted update blocks
    sit at the same ``[u * M, (u+1) * M)`` positions as the count plan (the
    accumulated gradient per update is the same set-sum; micro losses are re-weighted
    by sample share in :meth:`TrainStack._run_update`).
    """
    total = int(resp_track.batch_size)
    segment = resp_track.segment
    raw = getattr(segment, "lengths", None) if segment is not None else None
    if not (isinstance(raw, torch.Tensor) and raw.numel() == total):
        logger.warning(
            "token-budget packing requested (budget=%s) but the segment exposes no "
            "per-sample lengths; falling back to count-based micro-batching.",
            token_budget,
        )
        return None
    resp_lens = [int(x) for x in raw.tolist()]
    # The dense replay forwards [B, P_max + T_max] with the prompt and response
    # blocks padded SEPARATELY — pack with the exact 2D cost when prompt lengths are
    # available (verl's token accounting also covers prompt+response). conditions is
    # a Dict[str, Condition], so read "prompt" via the dict accessor.
    prompt_lens: Optional[List[int]] = None
    conditions = getattr(resp_track, "conditions", None)
    if isinstance(conditions, dict):
        prompt = conditions.get("prompt")
    else:
        prompt = getattr(conditions, "prompt", None) if conditions is not None else None
    pmask = getattr(prompt, "attention_mask", None) if prompt is not None else None
    if isinstance(pmask, torch.Tensor) and pmask.dim() == 2 and int(pmask.shape[0]) == total:
        prompt_lens = [int(p) for p in pmask.long().sum(dim=-1).tolist()]
    totals = [r + p for r, p in zip(resp_lens, prompt_lens)] if prompt_lens is not None else list(resp_lens)

    perm: List[int] = []
    plan: Plan = []
    cursor = 0
    n_micros = 0
    real_tokens = 0
    padded_tokens = 0
    for u_start, u_end in _update_ranges(total_size=total, num_updates=num_updates):
        indices = list(range(u_start, u_end))
        if cost_model == "sum":
            bins = _pack_micros_sum(indices=indices, lengths=totals, token_budget=token_budget)
        elif prompt_lens is not None:
            bins = _pack_micros_2d(
                indices=indices, prompt_lens=prompt_lens, resp_lens=resp_lens, token_budget=token_budget
            )
        else:
            bins = _pack_micros(indices=indices, lengths=totals, token_budget=token_budget)
        # NCCL micro-count parity: FSDP fwd/bwd run collectives per micro, so every
        # DP rank must execute the SAME number of micros per optimizer step or the
        # process group deadlocks. Packing is local, so sync to the global max and
        # re-partition into exactly that many bins.
        k = _sync_micro_count(len(bins))
        if k != len(bins):
            bins = _partition_into_k(indices=indices, lengths=totals, k=k)
        # Concatenate the bins into the permutation; each bin becomes a contiguous
        # range in the permuted order.
        update_plan: UpdatePlan = []
        for b in bins:
            perm.extend(b)
            update_plan.append((cursor, cursor + len(b)))
            cursor += len(b)
            n_micros += 1
            real_tokens += sum(totals[i] for i in b)
            if cost_model == "sum":
                padded_tokens += sum(totals[i] for i in b)
            else:
                pmax = max(prompt_lens[i] for i in b) if prompt_lens is not None else 0
                tmax = max(resp_lens[i] for i in b)
                padded_tokens += (pmax + tmax if prompt_lens is not None else tmax) * len(b)
        plan.append(update_plan)
    logger.info(
        "token-budget packing: %d micros for %d samples (budget=%d), dense efficiency %.0f%% (%d/%d tokens)",
        n_micros,
        total,
        token_budget,
        100.0 * real_tokens / max(1, padded_tokens),
        real_tokens,
        padded_tokens,
    )
    return perm, plan


# --------------------------------------------------------------------------- #
# Micro-batch planners — the strategy injected into TrainStack (composition)
# --------------------------------------------------------------------------- #
@runtime_checkable
class MicroPlanner(Protocol):
    """How an update's samples are grouped into micro-batches.

    :meth:`arrange` returns ``(track, plan)``: the track to train on — possibly
    reordered so packed micros are contiguous (sort-then-slice) — and one
    :data:`UpdatePlan` of contiguous ranges per optimizer step. :meth:`validate` is
    the algorithm precondition the grouping needs, checked once when the stack is built.
    """

    def arrange(
        self, resp_track: RolloutTrack, *, num_updates: int, micro_batch_size: int
    ) -> Tuple[RolloutTrack, Plan]: ...

    def validate(self, algorithm: StageAlgorithm) -> None: ...


class CountPlanner:
    """Fixed-count micro-batches: every micro holds ``micro_batch_size`` samples.

    The original ``TrainStack`` behaviour. Uniform-shape latents (diffusion) make a
    sample COUNT a good proxy for compute, so micros are contiguous equal-count
    slices and the per-rank micro count is identical across DP ranks — no NCCL
    micro-count parity collective is needed. Never reorders the track; imposes no
    algorithm precondition.
    """

    def arrange(
        self, resp_track: RolloutTrack, *, num_updates: int, micro_batch_size: int
    ) -> Tuple[RolloutTrack, Plan]:
        return resp_track, _count_plan(
            total=int(resp_track.batch_size),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
        )

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None


class TokenBudgetPlanner:
    """Token-budget packed micro-batches (verl ``ppo_max_token_len_per_gpu``).

    Varlen LLM sequences differ wildly in length, so a fixed sample COUNT is a poor
    proxy for compute. :meth:`arrange` length-sorts each update's samples and
    bin-packs them under ``token_budget``, then reorders the track once so those
    packed micros are contiguous ranges (sort-then-slice — the stack stays a pure
    contiguous-slice driver). Falls back to fixed-count micros (``micro_batch_size``)
    when the segment exposes no per-sample lengths.

    ``cost_model`` picks how a micro's cost is accounted against the budget:

    - ``'dense'``: ``(max_prompt + max_resp) * count`` — rectangular replay that
      pads to the in-micro maxes (``padding_replay``).
    - ``'sum'``: sum of real tokens — packed varlen replay (``packed_replay``;
      = verl token accounting).
    """

    def __init__(self, *, token_budget: int, cost_model: str = "dense") -> None:
        self.token_budget = _positive_int(name=f"{type(self).__name__}.token_budget", value=token_budget)
        if cost_model not in ("dense", "sum"):
            raise ValueError(f"{type(self).__name__}.cost_model must be dense|sum, got {cost_model!r}")
        self.cost_model = str(cost_model)

    def arrange(
        self, resp_track: RolloutTrack, *, num_updates: int, micro_batch_size: int
    ) -> Tuple[RolloutTrack, Plan]:
        packed = _arrange_packed(
            resp_track,
            num_updates=num_updates,
            token_budget=self.token_budget,
            cost_model=self.cost_model,
        )
        if packed is None:
            return resp_track, _count_plan(
                total=int(resp_track.batch_size),
                num_updates=num_updates,
                micro_batch_size=micro_batch_size,
            )
        perm, plan = packed
        # One up-front gather reorders the whole track (segment / conditions /
        # advantages stay sample-aligned) so the packed micros are contiguous.
        return resp_track.select(perm), plan

    def validate(self, algorithm: StageAlgorithm) -> None:
        """Guard: token-budget packing is gradient-exact only under seq-mean losses.

        ``_run_update`` weights each micro's loss by its share of SAMPLES, which
        reproduces the whole-update gradient only when the micro loss is a mean over
        the micro's SEQUENCES (any ``seq-mean-*`` mode — both
        ``seq-mean-token-sum-norm`` and ``seq-mean-token-mean`` are
        grouping-invariant). A token-level mean (``token-mean``) pools tokens across
        sequences, so regrouping samples into uneven micros changes the gradient.
        Fail fast rather than silently alter optimization.
        """
        mode = getattr(algorithm, "loss_agg_mode", None)
        if mode is None or not str(mode).startswith("seq-mean"):
            raise ValueError(
                f"{type(self).__name__}: token-budget packing requires a sequence-mean loss "
                f"aggregation (loss_agg_mode starting with 'seq-mean'), because micro losses are "
                f"weighted by sample share. {type(algorithm).__name__} has loss_agg_mode={mode!r}, "
                f"which is not grouping-invariant under packing — the update gradient would change. "
                f"Use loss_agg_mode='seq-mean-token-sum-norm' (or 'seq-mean-token-mean'), or use a "
                f"CountPlanner (omit micro_planner) for count-based micro-batching."
            )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainStepResult:
    """Result of one full optimizer step on this stage."""

    loss: float
    grad_norm: float
    lr: float
    has_backward: bool
    micros: List[AlgorithmStepResult]
    metrics: Mapping[str, object]
    # Per-optimizer-step metrics when num_updates_per_batch > 1 (one Mapping per
    # update, in order); empty for the single-update path. Lets the trainer log one
    # wandb point per optimizer step instead of averaging the updates.
    per_update: Tuple[Mapping[str, object], ...] = ()


def _aggregate_update_results(results: List["TrainStepResult"]) -> "TrainStepResult":
    """Collapse one rollout's per-update results into a single summary.

    Scalars are averaged across the N optimizer steps (``lr`` is the last,
    post-step value), ``micros`` are concatenated, and algorithm metrics are
    averaged via :func:`aggregate_numeric_metrics`. Downstream logging then treats
    the whole rollout as one point, exactly as in the single-update path.
    """
    if len(results) == 1:
        return results[0]
    n = len(results)
    micros: List[AlgorithmStepResult] = [m for r in results for m in r.micros]
    metrics = aggregate_numeric_metrics([dict(r.metrics) for r in results if r.metrics])
    return TrainStepResult(
        loss=sum(r.loss for r in results) / n,
        grad_norm=sum(r.grad_norm for r in results) / n,
        lr=results[-1].lr,
        has_backward=any(r.has_backward for r in results),
        micros=micros,
        metrics=metrics,
    )


def _align_track_to_model(resp_track: RolloutTrack, *, device: torch.device) -> None:
    """Move a track's training inputs onto the model's device — SGLang returns them
    on CPU via Ray IPC. Uses :meth:`Batch.to_device` (recursive; carries
    framework-managed ``_packed_cu_seqlens`` and tensors nested in tuples/dicts) on
    the segment + conditions only, so heavy ``decoded`` / ``media_preview`` payloads
    stay off the GPU. dtype is left to the model, which casts what it feeds the
    network (see SD3DiffusionStep.predict_noise).

    Condition values are moved via ``_move_value`` (the same recursive mover
    ``Batch.to_device`` uses) rather than assuming each value is a ``Batch``: most
    are (e.g. ``TextTokenCondition``), but multimodal stages also carry raw
    per-sample ``FieldKind.CONCAT`` lists of tensors (Qwen2.5-VL's ``pixel_values``
    / ``image_grid_thw``), which have no ``.to_device`` of their own — ``_move_value``
    handles Batch / tensor / list / dict / None uniformly."""
    if resp_track.segment is not None:
        resp_track.segment = resp_track.segment.to_device(device)
    resp_track.conditions = {k: _move_value(v, device) for k, v in resp_track.conditions.items()}
    if resp_track.advantages is not None:
        resp_track.advantages = resp_track.advantages.to(device=device)


# --------------------------------------------------------------------------- #
# The stack
# --------------------------------------------------------------------------- #
class TrainStack(Remote):
    """Single-stage stage-driven train stack — family-agnostic.

    One stage only — no track-name dict, no optional-track semantics, no multi-track
    on_rollout_end fan-out. The ONLY family-varying decision — micro-batch grouping —
    is delegated to an injected ``micro_planner`` (count-based vs token-budget);
    everything else is shared. Defaults to :class:`CountPlanner` (the historical
    diffusion behaviour), so the 60+ count-based configs need no ``micro_planner``
    block.

    Created as a sibling ``Remote`` inside a placement block; takes handles to its
    FSDPBackend and StageAlgorithm siblings via sibling-handle auto-resolve.
    """

    def __init__(
        self,
        *,
        fsdp_backend: FSDPBackend,
        algorithm: StageAlgorithm,
        micro_batch_size: int = 1,
        max_grad_norm: float,
        num_updates_per_batch: int = 1,
        micro_planner: Optional[MicroPlanner] = None,
    ) -> None:
        super().__init__()
        cls = type(self).__name__
        if int(micro_batch_size) < 1:
            raise ValueError(f"{cls}.micro_batch_size must be >= 1; got {micro_batch_size}.")
        if float(max_grad_norm) <= 0.0:
            raise ValueError(f"{cls}.max_grad_norm must be > 0; got {max_grad_norm}.")
        self.num_updates_per_batch = _positive_int(name=f"{cls}.num_updates_per_batch", value=num_updates_per_batch)
        if self.num_updates_per_batch > 1 and not getattr(algorithm, "supports_multi_update", False):
            raise ValueError(
                f"num_updates_per_batch={self.num_updates_per_batch} requires an algorithm whose "
                f"old_logp anchor stays frozen across the N optimizer steps "
                f"(FlowGRPO / FlowDPPO / GRPO / DRPO). "
                f"{type(algorithm).__name__} sets supports_multi_update=False, so >1 optimizer "
                f"step would train against a moving anchor. Set num_updates_per_batch=1."
            )
        self.fsdp_backend = fsdp_backend
        self.algorithm = algorithm
        self.micro_batch_size = int(micro_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        # Composition: the micro-batch grouping strategy. None → the historical
        # fixed-count behaviour. The planner also owns the algorithm precondition its
        # grouping requires (e.g. token-budget packing needs a seq-mean loss),
        # checked once here at construction.
        self.micro_planner: MicroPlanner = micro_planner if micro_planner is not None else CountPlanner()
        self.micro_planner.validate(algorithm)

    def prepare_segment(self, resp_track: RolloutTrack, *, plans: Plan) -> None:
        """Freeze the π_old anchor once, before the ``num_updates_per_batch`` loop.

        No-op if ``segment`` is None. If the algorithm does NOT replay the anchor
        (``recomputes_anchor() == False`` — e.g. rollout GRPO), the anchor is the
        rollout engine's own emission, so one full-segment call suffices. If it DOES
        replay (replay GRPO; FlowDPPO always, for ``sde_means``), the recomputed
        ``anchor_fields`` are computed at the SAME micro geometry training will use —
        the contiguous ranges in ``plans`` (already aligned with the reordered track
        from :meth:`MicroPlanner.arrange`) — so the old/new forwards match
        bf16-element-for-element on those fields. Concretely, the on-policy PPO ratio
        is exactly 1 only where ``sde_logp`` is replayed (replay GRPO, or FlowDPPO
        under ``old_logp_source='replay'``), and the on-policy KL is exactly 0
        wherever ``sde_means`` is replayed (FlowDPPO always). A single micro
        degenerates to one full-segment call; only the algorithm's declared
        ``anchor_fields`` are re-sliced and reassembled (no hardcoded field names).
        Because every micro is a contiguous range covering the shard in order, the
        per-micro field chunks reassemble with a plain ordered ``cat``.
        """
        if resp_track.segment is None:
            return
        algorithm = self.algorithm
        if not algorithm.recomputes_anchor():
            algorithm.prepare_segment(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        micro_slices = [r for update in plans for r in update]
        if len(micro_slices) == 1:
            algorithm.prepare_segment(conditions=resp_track.conditions, segment=resp_track.segment)
            return
        collected: Dict[str, List[torch.Tensor]] = {field: [] for field in algorithm.anchor_fields}
        for start, end in micro_slices:
            micro = resp_track.slice(start, end)
            algorithm.prepare_segment(conditions=micro.conditions, segment=micro.segment)
            for field in collected:
                value = getattr(micro.segment, field, None)
                if value is None:
                    raise RuntimeError(
                        f"{type(self).__name__}.prepare_segment: {type(algorithm).__name__} declares "
                        f"anchor field {field!r} but a micro produced None."
                    )
                collected[field].append(value)
        for field, parts in collected.items():
            setattr(resp_track.segment, field, torch.cat(parts, dim=0))

    def _run_update(
        self,
        resp_track: RolloutTrack,
        *,
        micros: UpdatePlan,
        training_progress: float,
    ) -> TrainStepResult:
        """Run one optimizer step over the contiguous micro ranges of a single update.

        ``micros`` is one update's worth of ``(start, end)`` ranges produced by
        :meth:`MicroPlanner.arrange` so the forward geometry matches the π_old anchor
        frozen by :meth:`prepare_segment`.
        """
        if resp_track.advantages is None:
            raise ValueError(
                f"{type(self).__name__}._run_update: resp_track.advantages is None; "
                "upstream advantage pipeline must populate it before training."
            )
        if not micros:
            raise ValueError(f"{type(self).__name__}._run_update: empty micros.")

        bs = int(resp_track.batch_size)
        self.fsdp_backend.zero_grad()

        update_total = sum(end - start for start, end in micros)
        micro_results: List[AlgorithmStepResult] = []
        total_loss = 0.0
        has_backward = False

        single_micro = len(micros) == 1 and micros[0] == (0, bs)
        last_micro = len(micros) - 1
        for i, (start, end) in enumerate(micros):
            # Defer the per-block gradient reduce-scatter to the last micro-batch so
            # it runs once per optimizer step instead of once per micro-batch (no-op
            # unless defer_grad_sync + ZeRO-2). Must precede the backward.
            self.fsdp_backend.set_grad_sync(i == last_micro)
            micro_track = resp_track if single_micro else resp_track.slice(start, end)
            # Sample-share weighting: the algorithm's micro loss is a MEAN over the
            # micro's sequences (seq-mean agg modes), so the update gradient equals
            # the whole-update mean only when each micro is weighted by its share of
            # samples. With equal count-based micros this reduces to 1/len(micros);
            # with token-budget packing micros vary in size.
            loss_scale = (end - start) / float(update_total)
            result = self.algorithm.compute_loss_and_backward(
                conditions=micro_track.conditions,
                segment=micro_track.segment,
                advantages=micro_track.advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
            micro_results.append(result)
            total_loss += result.loss
            has_backward = has_backward or result.has_backward

        aggregated_metrics: Mapping[str, object] = aggregate_numeric_metrics(
            [r.metrics for r in micro_results if r.metrics]
        )

        # Under defer_grad_sync the deferred reduce-scatter only runs inside a
        # backward that executes after set_grad_sync(True) — the last micro's. If
        # that micro skipped backward while earlier ones ran, the accumulated grads
        # were never synced: the optimizer would silently step on empty grads now,
        # and the stale unsharded accumulation (which zero_grad cannot reach) would
        # leak into the NEXT step's reduce-scatter. Fail fast instead — mirrors
        # fsdp_wrap's stray-trainable guard.
        if has_backward and not micro_results[-1].has_backward and self.fsdp_backend.grad_sync_deferred:
            raise RuntimeError(
                f"{type(self).__name__}._run_update: defer_grad_sync deferred the gradient "
                "reduce-scatter to the last micro-batch, but it reported no backward (all-empty "
                "micro?) while earlier micro-batches did — the accumulated grads were never "
                "synced. Disable training.fsdp.defer_grad_sync or investigate the empty micro-batch."
            )

        if has_backward:
            grad_norm = float(self.fsdp_backend.optimizer_step(max_grad_norm=float(self.max_grad_norm)))
        else:
            grad_norm = 0.0
            logger.warning(
                "%s._run_update: no micro reported backward; skipping optimizer step.",
                type(self).__name__,
            )
        if torch.cuda.is_available():
            # CUDA memory footprint per optimizer step (leak diagnosis: tp2 path
            # showed progressive OOM). Surfaces as train/cuda_alloc_gb|cuda_reserved_gb.
            aggregated_metrics = {
                **dict(aggregated_metrics),
                "cuda_alloc_gb": torch.cuda.memory_allocated() / 2**30,
                "cuda_reserved_gb": torch.cuda.memory_reserved() / 2**30,
            }

        return TrainStepResult(
            loss=total_loss,
            grad_norm=grad_norm,
            lr=self._current_lr(),
            has_backward=has_backward,
            micros=micro_results,
            metrics=aggregated_metrics,
        )

    def on_rollout_end(self) -> None:
        """Per-rollout-boundary hook — delegates to the FSDPBackend's EMA."""
        self.fsdp_backend.on_rollout_end()

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(
        self,
        resp_track: RolloutTrack,
        *,
        training_progress: float,
    ) -> TrainStepResult:
        """Driver-callable: arrange → prepare → run updates (×N) → on_rollout_end.

        Combines the steps so worker-side mutations (``segment.sde_logp`` populated
        by ``prepare_segment``) flow into the subsequent update(s) without
        round-tripping through the driver. Dispatched ``DP_SCATTER`` so each DP
        worker receives its shard of ``resp_track``; per-shard loss/grad_norm/metrics
        merge back via ``pytree_merge``.

        ``arrange`` reorders the shard (if packing) and builds the contiguous plan;
        ``prepare_segment`` then freezes the π_old anchor once at that geometry,
        ``num_updates_per_batch`` optimizer steps run over disjoint updates, and
        ``on_rollout_end`` runs once — see :meth:`_run_updates`.
        """
        self._align_track_inputs(resp_track)
        # Arrange once: reorder the track so packed micros are contiguous (no-op for
        # CountPlanner) and produce the plan. The SAME (track, plans) feed both the
        # anchor freeze and the train loop so both run the exact same geometry.
        resp_track, plans = self.micro_planner.arrange(
            resp_track,
            num_updates=self.num_updates_per_batch,
            micro_batch_size=self.micro_batch_size,
        )
        self.prepare_segment(resp_track, plans=plans)
        result = self._run_updates(resp_track, plans=plans, training_progress=float(training_progress))
        self.on_rollout_end()
        return result

    def _run_updates(
        self,
        resp_track: RolloutTrack,
        *,
        plans: Plan,
        training_progress: float,
    ) -> TrainStepResult:
        """Run ``num_updates_per_batch`` optimizer steps over disjoint updates.

        The update/micro grouping comes from :meth:`MicroPlanner.arrange` — the same
        source :meth:`prepare_segment` froze the π_old anchor at — so every update's
        ``new_logp`` is computed at exactly the anchor's geometry. ``prepare_segment``
        must already have frozen the anchor so all updates train against the same
        pre-update policy. With a single optimizer step the result passes through
        unchanged; otherwise the per-update results are reduced into one summary and
        each update's own metrics are attached on ``per_update`` (see
        :func:`_aggregate_update_results`).
        """
        results = [self._run_update(resp_track, micros=micros, training_progress=training_progress) for micros in plans]
        if len(results) == 1:
            return results[0]
        aggregated = _aggregate_update_results(results)
        # Attach each optimizer step's own metrics (in order) so the trainer can log
        # one wandb point per optimizer step — the on-policy update0 and the
        # off-policy update1 stay distinct series instead of being averaged into one
        # misleading ``ratio_mean``. Structured data on the result object, which the
        # DP collect (``pytree_cat``) returns whole, so it rides along.
        per_update = tuple(
            {**dict(r.metrics), "loss": float(r.loss), "grad_norm": float(r.grad_norm), "lr": float(r.lr)}
            for r in results
        )
        return replace(aggregated, per_update=per_update)

    def _align_track_inputs(self, resp_track: RolloutTrack) -> None:
        """Move the track onto the model's device; see :func:`_align_track_to_model`."""
        device = next(self.fsdp_backend.trainable_module().parameters()).device
        _align_track_to_model(resp_track, device=device)

    def _current_lr(self) -> float:
        optimizer = self.fsdp_backend.optimizer
        param_groups = getattr(optimizer, "param_groups", None)
        if isinstance(param_groups, list) and param_groups:
            return float(param_groups[0]["lr"])
        scheduler = self.fsdp_backend.scheduler
        if scheduler is not None and hasattr(scheduler, "get_last_lr"):
            last = scheduler.get_last_lr()
            if isinstance(last, list) and last:
                return float(last[0])
        return 0.0
