from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Set

if TYPE_CHECKING:
    from unirl.types.sample import Sample

RolloutFilter = Callable[[List["Sample"], int], List["Sample"]]


def _is_incomplete(sample: "Sample") -> bool:
    status = sample.parts[-1].harness_status if sample.parts else None
    if status in ("completed", "failed"):
        return False
    if status == "suspended":
        return True
    generated = sample.gen_parts()
    if not generated:
        return True
    return any(part.output_version is None for part in generated)


def _roots_of(sample: "Sample") -> Set[str]:
    if not sample.parts:
        return set()
    return set(sample.root_group_ids(0))


def identity(samples: List["Sample"], current_version: int) -> List["Sample"]:
    del current_version
    return samples


def chain(*filters: RolloutFilter) -> RolloutFilter:
    def apply(samples: List["Sample"], current_version: int) -> List["Sample"]:
        kept = samples
        for filter_fn in filters:
            kept = filter_fn(kept, current_version)
            if not kept:
                break
        return kept

    return apply


def drop_incomplete(samples: List["Sample"], current_version: int) -> List["Sample"]:
    del current_version
    rejected = set().union(*(_roots_of(sample) for sample in samples if _is_incomplete(sample)))
    return [sample for sample in samples if not (_roots_of(sample) & rejected)]


def keep_within_lag(max_lag: int) -> RolloutFilter:
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative; got {max_lag}")

    def apply(samples: List["Sample"], current_version: int) -> List["Sample"]:
        rejected = set()
        for sample in samples:
            versions = []
            for part in sample.gen_parts():
                if part.output_version is not None:
                    versions.append(int(part.output_version))
            if versions and max(versions) > current_version:
                raise RuntimeError("rollout has a future output version")
            if versions and current_version - min(versions) > max_lag:
                rejected.update(_roots_of(sample))
        return [sample for sample in samples if not (_roots_of(sample) & rejected)]

    return apply


def prefer_newer(samples: List["Sample"], current_version: int) -> List["Sample"]:
    del current_version

    def rollout_id(sample: "Sample") -> int:
        if not sample.parts or not sample.parts[0].metadata:
            raise RuntimeError("rollout Sample has no root rollout_id metadata")
        values = {row.get("rollout_id") for row in sample.parts[0].metadata}
        if None in values or len(values) != 1:
            raise RuntimeError(f"rollout Sample must carry one root rollout_id; got {values}")
        return int(next(iter(values)))

    return sorted(samples, key=rollout_id, reverse=True)


__all__ = ["RolloutFilter", "chain", "drop_incomplete", "identity", "keep_within_lag", "prefer_newer"]
