from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from unirl.types.sample import Sample


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


__all__ = ["combine_rollout_chunks"]
