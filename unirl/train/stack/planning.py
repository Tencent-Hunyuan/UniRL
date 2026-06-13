"""Micro-batch planning: how an update's samples are grouped into micro-batches.

Two layers, both in this module because they are bound — the planners are just
the OO face of the functions:

1. The **plan** algorithms (private). A plan is a recipe, built before any
   forward runs::

       Plan       = List[UpdatePlan]   # the whole rollout shard
       UpdatePlan = List[Range]        # one optimizer step (= one "update")
       Range      = (start, end)       # one micro's CONTIGUOUS sample membership

   Because packing reorders the track up front (sort-then-slice, see
   :meth:`TokenBudgetPlanner.arrange`), a micro is ALWAYS a contiguous range — no
   index lists. :func:`_count_plan` builds fixed-count micros (diffusion /
   FlowGRPO schedule); :func:`_arrange_packed` builds verl
   ``ppo_max_token_len_per_gpu`` parity packing for varlen LLMs. Both partition the
   rollout into equal contiguous *updates* first; they differ only in how each
   update's samples become micros.

2. The **planner** strategies (public): :class:`MicroPlanner` and its
   implementations :class:`CountPlanner` / :class:`TokenBudgetPlanner`. These are
   what :class:`~unirl.train.stack.base.TrainStack` composes (one injected
   ``micro_planner``) instead of subclassing — each wraps the plan functions above
   and owns the algorithm precondition its grouping requires
   (:meth:`MicroPlanner.validate`). Recipes reference them by ``_target_``.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import torch

from unirl.algorithms import StageAlgorithm
from unirl.types.rollout_resp import RolloutTrack

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
    by sample share in :meth:`~unirl.train.stack.base.TrainStack._run_update`).
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
