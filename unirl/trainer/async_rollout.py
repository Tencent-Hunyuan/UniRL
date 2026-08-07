from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from unirl.types.sample import Sample


def launch_ceiling(rollout_id: int, *, sync_interval: int, max_staleness: int, num_rollouts: int) -> int:
    return min(num_rollouts, ((rollout_id // sync_interval) + 1 + max_staleness) * sync_interval)


def combine_rollout_chunks(groups: List[List["Sample"]]) -> Tuple["Sample", int]:
    chunks = [sample for group in groups for sample in group]
    if not chunks:
        raise ValueError("cannot combine an empty rollout result")
    rollout_ids = [_rollout_id(sample) for sample in chunks]
    if len(chunks) == 1:
        return chunks[0], rollout_ids[0]

    from unirl.types.sample import Sample

    return Sample.concat(chunks), max(rollout_ids)


def _rollout_id(sample: "Sample") -> int:
    if not sample.parts or not sample.parts[0].metadata:
        raise RuntimeError("rollout Sample has no root rollout_id metadata")
    values = {row.get("rollout_id") for row in sample.parts[0].metadata}
    if None in values or len(values) != 1:
        raise RuntimeError(f"rollout Sample must carry one root rollout_id; got {values}")
    return int(next(iter(values)))


__all__ = ["combine_rollout_chunks", "launch_ceiling"]
