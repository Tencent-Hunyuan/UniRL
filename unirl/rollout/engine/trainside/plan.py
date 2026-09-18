"""Shape-bucket execution keys and the group-aligned micro-batch plan for trainside rollout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

Range = Tuple[int, int]
Signature = Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class GeometryKey:
    """Final execution geometry: what decides whether two rows can share one stacked forward."""

    model_adapter: str
    latent_shape: Tuple[int, ...]
    conditioning_layout: str


@dataclass(frozen=True)
class ExecutionKey:
    """Everything that must match before rows share one model forward step."""

    geometry: GeometryKey
    dtype: str
    cfg_rows: int
    schedule: str
    weight_version: Optional[int]
    parallel: str

    def geometry_token(self) -> str:
        """Short stable token naming one geometry bucket — the ``bucket_batch_sizes`` key."""
        shape = "x".join(str(int(dim)) for dim in self.geometry.latent_shape) or "?"
        return f"{self.geometry.model_adapter}|{shape}|{self.geometry.conditioning_layout}|cfg{int(self.cfg_rows)}"

    def token(self) -> str:
        """Full stable token, including the non-geometry settings, for logs and diagnostics."""
        version = "?" if self.weight_version is None else str(int(self.weight_version))
        return f"{self.geometry_token()}|{self.dtype}|{self.schedule}|v{version}|{self.parallel}"


@dataclass(frozen=True)
class BucketPlan:
    """Stable permutation of one frontier into per-bucket contiguous micro ranges."""

    permutation: torch.Tensor  # [N] long — perm[new_index] = old_index
    inverse: torch.Tensor  # [N] long — inverse[old_index] = new_index
    micros: Tuple[Range, ...]  # contiguous [start, end) ranges in permuted order
    micro_keys: Tuple[ExecutionKey, ...]  # one execution key per micro
    signature: Signature  # ordered (bucket token, request rows) per micro

    @property
    def is_passthrough(self) -> bool:
        """Whether the plan is one micro covering every row in original order."""
        rows = int(self.permutation.numel())
        if self.micros != ((0, rows),):
            return False
        return bool(torch.equal(self.permutation, torch.arange(rows, dtype=self.permutation.dtype)))


def plan_micro_batches(
    *,
    group_ids: Sequence[str],
    keys: Sequence[ExecutionKey],
    capacity_rows: Callable[[ExecutionKey], int],
) -> BucketPlan:
    """Bucket complete prompt groups into contiguous micro ranges with an exact inverse permutation."""
    if len(group_ids) != len(keys):
        raise ValueError(f"plan_micro_batches: {len(group_ids)} group ids for {len(keys)} execution keys")
    total = len(group_ids)
    if total == 0:
        empty = torch.empty(0, dtype=torch.long)
        return BucketPlan(permutation=empty, inverse=empty.clone(), micros=(), micro_keys=(), signature=())

    groups: Dict[str, List[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)

    group_keys: Dict[str, ExecutionKey] = {}
    for group_id, rows in groups.items():
        first = keys[rows[0]]
        for row in rows[1:]:
            if keys[row] != first:
                raise ValueError(
                    f"plan_micro_batches: prompt group {group_id!r} spans incompatible execution keys "
                    f"{first.token()!r} and {keys[row].token()!r}; the group cannot be scheduled as one unit "
                    "under a shared micro."
                )
        group_keys[group_id] = first

    buckets: Dict[str, List[str]] = {}
    for group_id in groups:
        buckets.setdefault(group_keys[group_id].geometry_token(), []).append(group_id)

    order: List[int] = []
    micros: List[Range] = []
    micro_keys: List[ExecutionKey] = []
    for token, bucket_groups in buckets.items():
        key = group_keys[bucket_groups[0]]
        requests = _capacity_requests(key=key, capacity_rows=capacity_rows(key))
        start = len(order)
        used = 0
        for group_id in bucket_groups:
            size = len(groups[group_id])
            if size > requests:
                raise ValueError(
                    f"plan_micro_batches: prompt group {group_id!r} has {size} rows, more than the {requests}-row "
                    f"capacity of bucket {token!r} ({capacity_rows(key)} execution rows). Raise the bucket capacity; "
                    "a group is never split."
                )
            if used and used + size > requests:
                micros.append((start, len(order)))
                micro_keys.append(key)
                start = len(order)
                used = 0
            order.extend(groups[group_id])
            used += size
        if used:
            micros.append((start, len(order)))
            micro_keys.append(key)

    permutation = torch.tensor(order, dtype=torch.long)
    inverse = torch.empty(total, dtype=torch.long)
    inverse[permutation] = torch.arange(total, dtype=torch.long)
    signature = tuple((key.geometry_token(), end - start) for key, (start, end) in zip(micro_keys, micros))
    return BucketPlan(
        permutation=permutation,
        inverse=inverse,
        micros=tuple(micros),
        micro_keys=tuple(micro_keys),
        signature=signature,
    )


def assert_rank_uniform_schedule(plan: BucketPlan, *, group_sizes: Sequence[int]) -> None:
    """Fail closed unless the micro schedule follows from rank-uniform inputs alone."""
    tokens = sorted({token for token, _ in plan.signature})
    if len(tokens) > 1:
        raise ValueError(
            "shape_bucket: this frontier mixes multiple execution buckets "
            f"({tokens}); per-bucket micro counts would then depend on the rank-local bucket mix, so the ranks "
            "sharing an FSDP all-gather would run different numbers of block forwards. A rank-uniform schedule "
            "exchange is not implemented — keep one execution geometry per shard."
        )
    distinct = {int(size) for size in group_sizes}
    if len(distinct) > 1:
        raise ValueError(
            f"shape_bucket: prompt groups in this frontier have unequal sizes {sorted(distinct)}; the micro count "
            "would then depend on the rank-local group layout. Split the shard so every group has one size."
        )


def prompt_group_sizes(group_ids: Sequence[str]) -> Tuple[int, ...]:
    """Row count of each prompt group, in first-appearance order."""
    counts: Dict[str, int] = {}
    for group_id in group_ids:
        name = str(group_id)
        counts[name] = counts.get(name, 0) + 1
    return tuple(counts.values())


def schedule_token(params: object) -> str:
    """Stable digest of the σ schedule and SDE step gate that a bucket must share."""
    sigmas = getattr(params, "sigmas", None)
    if sigmas is None:
        digest = "none"
    else:
        flat = sigmas.detach().to(torch.float32).reshape(-1)
        ends = f"{float(flat[0]):.6g}:{float(flat[-1]):.6g}" if flat.numel() else "?"
        digest = f"{int(flat.numel())}@{ends}"
    sde = getattr(params, "sde_indices", None)
    steps = int(getattr(params, "num_inference_steps", 0) or 0)
    return f"T{steps}/sde{len(sde) if sde is not None else -1}/{digest}"


def positive_int(*, name: str, value: object) -> int:
    """Reject bool/float/str and non-positive ints so a capacity can never be silently coerced."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int (bool, float and str are rejected); got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1; got {value}")
    return int(value)


def _capacity_requests(*, key: ExecutionKey, capacity_rows: int) -> int:
    """Convert a declared execution-row capacity into request rows for one bucket."""
    capacity = positive_int(name="capacity_rows", value=capacity_rows)
    cfg_rows = positive_int(name="ExecutionKey.cfg_rows", value=key.cfg_rows)
    requests = capacity // cfg_rows
    if requests < 1:
        raise ValueError(
            f"capacity_rows={capacity} is below this bucket's CFG execution width {cfg_rows}; one request would "
            "already exceed the declared capacity."
        )
    return requests


__all__ = [
    "BucketPlan",
    "ExecutionKey",
    "GeometryKey",
    "Range",
    "Signature",
    "assert_rank_uniform_schedule",
    "plan_micro_batches",
    "positive_int",
    "prompt_group_sizes",
    "schedule_token",
]
