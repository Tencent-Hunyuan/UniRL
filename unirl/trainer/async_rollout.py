from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from unirl.types.sample import Sample


def next_hard_boundary(
    trained_batches: int,
    *,
    num_rollouts: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    boundary = num_rollouts
    for interval in (eval_interval, save_interval):
        if interval > 0 and trained_batches < num_rollouts:
            boundary = min(boundary, ((trained_batches // interval) + 1) * interval)
    return boundary


def boundary_launch_slots(
    *,
    inflight_count: int,
    ready_count: int,
    max_inflight: int,
    trained_batches: int,
    num_rollouts: int,
    hard_boundary: int,
) -> int:
    remaining = min(num_rollouts, hard_boundary) - trained_batches - inflight_count - ready_count
    return max(0, min(max_inflight - inflight_count, remaining))


def rollout_version_metrics(
    *,
    train_version: int,
    output_version: int,
    num_updates_per_batch: int,
) -> dict[str, float]:
    staleness = train_version - output_version
    if staleness < 0:
        raise ValueError(f"rollout batch has future output version {output_version} > train version {train_version}")
    return {
        "async/output_version": output_version,
        "async/staleness_updates": staleness,
        "async/staleness_batches": staleness / num_updates_per_batch,
    }


def training_version_metrics(
    *,
    train_version: int,
    published_version: int,
    optimizer_updates: int,
    batches_since_sync: int,
) -> dict[str, int]:
    return {
        "async/train_version": train_version,
        "async/published_version": published_version,
        "async/publish_lag": train_version - published_version,
        "async/optimizer_updates": optimizer_updates,
        "async/batches_since_sync": batches_since_sync,
    }


def combine_rollout_chunks(groups: List[List["Sample"]]) -> Tuple["Sample", int, int]:
    chunks = [sample for group in groups for sample in group]
    if not chunks:
        raise ValueError("cannot combine an empty rollout result")
    rollout_ids = [_rollout_id(sample) for sample in chunks]
    if len(set(rollout_ids)) != 1:
        raise RuntimeError(f"rollout batch combines multiple generation ids: {sorted(set(rollout_ids))}")
    versions = {part.output_version for sample in chunks for part in sample.gen_parts()}
    if not versions or None in versions:
        raise RuntimeError("rollout batch is missing output_version provenance")
    if len(versions) != 1:
        raise RuntimeError(f"rollout batch has mixed output versions: {sorted(versions)}")
    output_version = int(next(iter(versions)))
    if len(chunks) == 1:
        return chunks[0], rollout_ids[0], output_version

    from unirl.types.sample import Sample

    return Sample.concat(chunks), rollout_ids[0], output_version


def _rollout_id(sample: "Sample") -> int:
    if not sample.parts or not sample.parts[0].metadata:
        raise RuntimeError("rollout Sample has no root rollout_id metadata")
    values = {row.get("rollout_id") for row in sample.parts[0].metadata}
    if None in values or len(values) != 1:
        raise RuntimeError(f"rollout Sample must carry one root rollout_id; got {values}")
    return int(next(iter(values)))


__all__ = [
    "boundary_launch_slots",
    "combine_rollout_chunks",
    "next_hard_boundary",
    "rollout_version_metrics",
    "training_version_metrics",
]
