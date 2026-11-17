"""Reward-panel rules shared by the remote backend and the composite scorer; see README Gotchas."""

from __future__ import annotations

import math
from typing import Collection, Sequence

from unirl.config.require import require

AGGREGATION_METHODS = ("weighted_sum", "linear_sum", "mean", "min", "max")


def monitor_key(name: str) -> str:
    """``component_rewards`` key of a score-only monitor."""
    return f"monitor_{name}"


def validate_panel(
    owner: str,
    *,
    training: Collection[str],
    monitors: Collection[str],
    weighted: Collection[str],
    aggregation_method: str,
) -> None:
    """Reject a panel whose training / monitor / weight names or aggregation method are inconsistent."""
    require(len(training) > 0, f"{owner}: at least one training reward is required")
    both = set(training) & set(monitors)
    require(not both, f"{owner}: training rewards and monitors must be disjoint; both name {sorted(both)!r}")
    colliding = {monitor_key(name) for name in monitors} & set(training)
    require(
        not colliding, f"{owner}: training rewards must not be named {sorted(colliding)!r}; monitors report as that"
    )
    weighted_monitors = set(monitors) & set(weighted)
    require(not weighted_monitors, f"{owner}: monitors cannot carry weights; got {sorted(weighted_monitors)!r}")
    require(
        aggregation_method in AGGREGATION_METHODS,
        f"{owner}: aggregation_method must be one of {'/'.join(AGGREGATION_METHODS)}; got {aggregation_method!r}",
    )


def aggregate_panel(scores: Sequence[float], weights: Sequence[float], aggregation_method: str) -> float:
    """Aggregate one sample's training scores, in the dtype of ``scores``; a non-finite result raises."""
    if aggregation_method == "mean":
        result = sum(scores) / len(scores)
    elif aggregation_method == "min":
        result = min(scores)
    elif aggregation_method == "max":
        result = max(scores)
    else:
        result = sum(s * w for s, w in zip(scores, weights))
        if aggregation_method == "weighted_sum":
            total_w = sum(weights)
            result = result / total_w if total_w > 0 else 0.0
    require(math.isfinite(result), f"{aggregation_method} over finite scores produced a non-finite reward")
    return result


__all__ = ["AGGREGATION_METHODS", "aggregate_panel", "monitor_key", "validate_panel"]
