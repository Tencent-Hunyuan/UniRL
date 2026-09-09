"""Token-budget packed micro-batching (verl ``ppo_max_token_len_per_gpu`` parity)."""

from __future__ import annotations

import logging
from typing import Callable, List, NamedTuple, Optional, Tuple

import torch

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner.count import _count_plan
from unirl.train.stack.planner.types import Plan, UpdatePlan, _positive_int, _update_ranges
from unirl.types.sample import Part

logger = logging.getLogger(__name__)

Cost = Callable[[List["Sample"]], int]


class Sample(NamedTuple):
    """A sequence's packing sizes; build via :func:`_sample` (clamps resp>=1, prompt>=0)."""

    idx: int
    prompt: int
    resp: int

    @property
    def total(self) -> int:
        return self.prompt + self.resp


def _sample(idx: int, *, prompt: int, resp: int) -> Sample:
    """Build a :class:`Sample` with the resp>=1 / prompt>=0 clamps (so total>=1)."""
    return Sample(idx=idx, prompt=max(0, int(prompt)), resp=max(1, int(resp)))


def dense(micro: List[Sample]) -> int:
    return (max(s.prompt for s in micro) + max(s.resp for s in micro)) * len(micro)


def varlen_sum(micro: List[Sample]) -> int:
    return sum(s.total for s in micro)


def first_fit_decreasing(samples: List[Sample], *, cost: Cost, budget: int) -> List[List[Sample]]:
    """Pack samples into micros under ``budget`` (longest-first; an oversize sample gets its own micro)."""
    if int(budget) < 1:
        raise ValueError(f"token_budget must be >= 1; got {budget}")
    micros: List[List[Sample]] = []
    for s in sorted(samples, key=lambda s: (-s.total, s.idx)):
        for micro in micros:
            if cost(micro + [s]) <= budget:
                micro.append(s)
                break
        else:
            micros.append([s])
    return micros


def balance_into_k(samples: List[Sample], *, cost: Cost, k: int) -> List[List[Sample]]:
    """Re-pack into EXACTLY ``k`` micros (NCCL parity); each sample joins the micro it grows least."""
    if k < 1 or k > len(samples):
        raise ValueError(f"balance_into_k: k={k} out of range for {len(samples)} samples")
    order = sorted(samples, key=lambda s: (-s.total, s.idx))
    micros: List[List[Sample]] = [[s] for s in order[:k]]
    for s in order[k:]:
        min(micros, key=lambda micro: cost(micro + [s])).append(s)
    return micros


def _prompt_lengths(part: Part, total: int) -> Optional[List[int]]:
    """Per-sample prompt token counts from ``conditions['prompt'].attention_mask`` (None if absent)."""
    conditions = getattr(part, "conditions", None)
    if isinstance(conditions, dict):
        prompt = conditions.get("prompt")
    else:
        prompt = getattr(conditions, "prompt", None) if conditions is not None else None
    pmask = getattr(prompt, "attention_mask", None) if prompt is not None else None
    if isinstance(pmask, torch.Tensor) and pmask.dim() == 2 and int(pmask.shape[0]) == total:
        return [int(p) for p in pmask.long().sum(dim=-1).tolist()]
    return None


def _extract_samples(part: Part) -> Optional[List[Sample]]:
    """Clamped :class:`Sample` list from a track, or ``None`` when it exposes no per-sample lengths."""
    total = int(part.batch_size)
    segment = part.segment
    raw = getattr(segment, "lengths", None) if segment is not None else None
    if not (isinstance(raw, torch.Tensor) and raw.numel() == total):
        return None
    resp_lens = [int(x) for x in raw.tolist()]
    prompt_lens = _prompt_lengths(part, total)
    return [
        _sample(i, prompt=(prompt_lens[i] if prompt_lens is not None else 0), resp=resp_lens[i]) for i in range(total)
    ]


def _sync_micro_count(local_count: int) -> int:
    """All-reduce(MAX) the per-rank micro count for FSDP parity; no-op without torch.distributed."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() <= 1:
        return local_count
    t = torch.tensor([int(local_count)], dtype=torch.long, device="cuda" if torch.cuda.is_available() else "cpu")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return int(t.item())


def _log_packing_efficiency(micros: List[List[Sample]], *, cost: Cost, budget: int) -> None:
    """Log packing efficiency = real tokens / materialized slots across all micros."""
    real = sum(s.total for micro in micros for s in micro)
    padded = sum(cost(micro) for micro in micros)
    samples = sum(len(micro) for micro in micros)
    logger.info(
        "token-budget packing: %d micros for %d samples (budget=%d), efficiency %.0f%% (%d/%d tokens)",
        len(micros),
        samples,
        budget,
        100.0 * real / max(1, padded),
        real,
        padded,
    )


def _arrange_packed(
    samples: List[Sample], *, num_updates: int, token_budget: int, cost_model: str
) -> Tuple[List[int], Plan]:
    """Pack ``samples`` into ``(perm, plan)`` — a sort-then-slice permutation plus the contiguous range plan over it."""
    cost: Cost = varlen_sum if cost_model == "sum" else dense
    perm: List[int] = []
    plan: Plan = []
    all_micros: List[List[Sample]] = []
    cursor = 0
    for u_start, u_end in _update_ranges(total_size=len(samples), num_updates=num_updates):
        update_samples = samples[u_start:u_end]
        micros = first_fit_decreasing(update_samples, cost=cost, budget=token_budget)
        k = _sync_micro_count(len(micros))
        if k != len(micros):
            micros = balance_into_k(update_samples, cost=cost, k=k)
        update_plan: UpdatePlan = []
        for micro in micros:
            perm.extend(s.idx for s in micro)
            update_plan.append((cursor, cursor + len(micro)))
            cursor += len(micro)
        plan.append(update_plan)
        all_micros.extend(micros)
    _log_packing_efficiency(all_micros, cost=cost, budget=token_budget)
    return perm, plan


class TokenBudgetPlanner:
    """Token-budget packed micro-batches (verl ``ppo_max_token_len_per_gpu``)."""

    def __init__(self, *, token_budget: int, cost_model: str = "dense") -> None:
        self.token_budget = _positive_int(name=f"{type(self).__name__}.token_budget", value=token_budget)
        if cost_model not in ("dense", "sum"):
            raise ValueError(f"{type(self).__name__}.cost_model must be dense|sum, got {cost_model!r}")
        self.cost_model = str(cost_model)

    def arrange(self, part: Part, *, num_updates: int, micro_batch_size: int) -> Tuple[Part, Plan]:
        samples = _extract_samples(part)
        if samples is None:
            logger.warning(
                "token-budget packing requested (budget=%s) but the segment exposes no "
                "per-sample lengths; falling back to count-based micro-batching.",
                self.token_budget,
            )
            return part, _count_plan(
                total=int(part.batch_size),
                num_updates=num_updates,
                micro_batch_size=micro_batch_size,
            )
        perm, plan = _arrange_packed(
            samples, num_updates=num_updates, token_budget=self.token_budget, cost_model=self.cost_model
        )
        return part.select(perm), plan

    def validate(self, algorithm: StageAlgorithm) -> None:
        """Require a grouping-invariant loss-weighting contract."""
        if getattr(algorithm, "loss_weighting", "sample") == "token":
            return
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
