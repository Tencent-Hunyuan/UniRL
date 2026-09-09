"""Root-atomic retention filters applied by the rollout manager."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Callable, List

from unirl.rollout.manager.buffers import roots_of

if TYPE_CHECKING:
    from unirl.types.sample import Sample

RolloutFilter = Callable[[List["Sample"], int], List["Sample"]]


def identity(samples: List["Sample"], current_version: int) -> List["Sample"]:
    del current_version
    return samples


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
                rejected.update(roots_of(sample))
        return [sample for sample in samples if not (set(roots_of(sample)) & rejected)]

    return apply


def validate_filter_output(candidates: List["Sample"], kept: List["Sample"]) -> None:
    """Enforce the filter contract: ``kept`` is drawn from ``candidates`` without duplication."""
    candidate_ids = Counter(map(id, candidates))
    kept_ids = Counter(map(id, kept))
    if kept_ids - candidate_ids:
        raise RuntimeError("rollout filter returned a Sample outside its input")
    if any(count != 1 for count in kept_ids.values()):
        raise RuntimeError("rollout filter returned the same Sample more than once")


__all__ = ["RolloutFilter", "identity", "keep_within_lag", "validate_filter_output"]
