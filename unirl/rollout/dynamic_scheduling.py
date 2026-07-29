"""Real-time, per-rank dynamic scheduling for async rollout (WIP — placeholder).

Today a single ``generate`` is a whole-batch ``DP_SCATTER``: one launch shards a
full batch across the rollout slab and the job is only *ready* when the SLOWEST
rank finishes (``RayGenerationDispatcher.is_ready`` waits on ``num_returns ==
len(refs)``). With the on-policy defaults (``max_inflight=1`` /
``buffer_max_staleness=0``) that means a fast rank sits idle until the tail rank
drains the batch — the DP long-tail bubble.

This module will schedule generation at a finer, per-rank granularity so a rank
that finishes early pulls the next unit of work instead of blocking on the tail,
without weakening the on-policy staleness guarantees that
``AsyncRolloutScheduler`` enforces today.

Nothing here is wired into the rollout path yet — this is a placeholder to stake
out the branch and the seam. See ``unirl/rollout/async_runtime.py`` for the
current whole-batch scheduler this will refine.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicScheduleConfig:
    """Knobs for per-rank dynamic scheduling (placeholder — not consumed yet)."""

    enabled: bool = False


class DynamicRolloutScheduler:
    """Per-rank dynamic rollout scheduler.

    TODO(async-rollout-dynamic-scheduling):
      - dispatch generation per rank (not whole-batch ``DP_SCATTER``) so an
        early-finishing rank claims the next work unit;
      - keep the versioned-buffer freshness/staleness eviction intact;
      - drain/quiesce semantics compatible with the mandatory pre-weight-sync
        barrier.
    """

    def __init__(self, config: DynamicScheduleConfig | None = None) -> None:
        self._config = config or DynamicScheduleConfig()

    def next_batch(self, *args, **kwargs):  # noqa: D401,ANN002,ANN003
        raise NotImplementedError(
            "async-rollout-dynamic-scheduling: per-rank dynamic scheduling is a WIP placeholder"
        )


__all__ = ["DynamicScheduleConfig", "DynamicRolloutScheduler"]
