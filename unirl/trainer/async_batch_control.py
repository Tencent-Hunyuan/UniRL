"""Batch admission and weight-publication control for async AR and diffusion; versions count optimizer updates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AsyncBatchControl:
    """Track train/published versions and gate batch generation."""

    weight_sync_interval: int
    num_updates_per_batch: int
    train_version: int = 0
    published_version: int = 0
    batches_since_sync: int = 0

    def __post_init__(self) -> None:
        if self.weight_sync_interval < 1:
            raise ValueError(f"weight_sync_interval must be >= 1, got {self.weight_sync_interval}")
        if self.num_updates_per_batch < 1:
            raise ValueError(f"num_updates_per_batch must be >= 1, got {self.num_updates_per_batch}")
        if self.published_version > self.train_version:
            raise ValueError(
                f"published_version cannot be ahead of train_version: {self.published_version} > {self.train_version}"
            )
        if not 0 <= self.batches_since_sync <= self.weight_sync_interval:
            raise ValueError(
                "batches_since_sync must be within the current publication interval: "
                f"0 <= {self.batches_since_sync} <= {self.weight_sync_interval}"
            )

    @property
    def max_staleness(self) -> int:
        """Maximum batch-entry staleness, in rollout batches."""
        return self.weight_sync_interval - 1

    @property
    def staleness_budget(self) -> int:
        """``max_staleness`` expressed in optimizer updates."""
        return self.max_staleness * self.num_updates_per_batch

    @property
    def publish_lag(self) -> int:
        return self.train_version - self.published_version

    @property
    def admission_depth(self) -> int:
        """Batches one published rollout snapshot may serve."""
        return self.weight_sync_interval

    @property
    def publication_due(self) -> bool:
        return self.batches_since_sync >= self.weight_sync_interval

    def restore(self, train_version: int) -> None:
        self.train_version = train_version
        self.published_version = 0
        self.batches_since_sync = 0

    def staleness(self, output_version: int) -> int:
        """Optimizer updates between train and the policy that produced a batch."""
        stale = self.train_version - output_version
        if stale < 0:
            raise ValueError(
                f"rollout batch has future output version {output_version} > train version {self.train_version}"
            )
        return stale

    def record_optimizer_updates(self, optimizer_updates: int) -> None:
        self.train_version += optimizer_updates
        self.batches_since_sync += 1

    def launch_slots(
        self,
        *,
        inflight_count: int,
        ready_count: int,
        max_inflight: int,
        trained_batches: int,
        num_rollouts: int,
        hard_boundary: int,
    ) -> int:
        if self.publication_due:
            return 0
        freshness = self.weight_sync_interval - self.batches_since_sync
        allowed = min(freshness, num_rollouts - trained_batches, hard_boundary - trained_batches)
        return max(0, min(max_inflight - inflight_count, allowed - inflight_count - ready_count))

    def sync_rollout(self, engine: Any, rollout: Any, weight_sync: Any, *, force: bool = False) -> None:
        if not force and self.publish_lag == 0:
            self.batches_since_sync = 0
            return
        engine.quiesce()
        if engine.ready_count:
            raise RuntimeError(
                f"cannot sync rollout weights with completed batches queued: ready_count={engine.ready_count}"
            )
        weight_sync.sync()
        rollout.set_version(self.train_version)
        self.published_version = self.train_version
        self.batches_since_sync = 0

    def output_metrics(self, output_version: int) -> dict[str, float]:
        staleness = self.staleness(output_version)
        return {
            "async/output_version": output_version,
            "async/staleness_updates": staleness,
            "async/staleness_batches": staleness / self.num_updates_per_batch,
        }

    def train_metrics(self, optimizer_updates: int) -> dict[str, int]:
        return {
            "async/train_version": self.train_version,
            "async/published_version": self.published_version,
            "async/publish_lag": self.publish_lag,
            "async/optimizer_updates": optimizer_updates,
            "async/batches_since_sync": self.batches_since_sync,
        }


def max_publication_gap_batches(
    control: AsyncBatchControl,
    *,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    """Maximum publication gap after eval/checkpoint boundaries are applied."""
    max_gap = control.admission_depth
    for interval in (eval_interval, save_interval):
        if interval > 0:
            max_gap = min(max_gap, interval)
    return max_gap


def log_admission_notes(
    control: AsyncBatchControl,
    *,
    max_inflight: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> None:
    """Warn when configured admission capacity cannot be fully used."""
    max_gap = max_publication_gap_batches(
        control,
        eval_interval=eval_interval,
        save_interval=save_interval,
    )

    if control.weight_sync_interval == 1:
        logger.warning(
            "weight_sync_interval=1 admits one generation at a time; generation cannot overlap the preceding "
            "train batch"
        )
    if max_inflight > control.admission_depth:
        logger.warning(
            "max_inflight=%d exceeds the staleness admission depth %d; the extra concurrency cannot be used",
            max_inflight,
            control.admission_depth,
        )
    if max_gap < control.admission_depth:
        logger.warning(
            "eval/checkpoint boundaries cap the maximum publication gap at %d batches, "
            "below weight_sync_interval=%d "
            "(eval_interval=%d, save_interval=%d), so derived max_staleness=%d is never fully "
            "spent — data tops out at %d batches stale",
            max_gap,
            control.admission_depth,
            eval_interval,
            save_interval,
            control.max_staleness,
            max_gap - 1,
        )


def next_hard_boundary(
    trained_batches: int,
    *,
    num_rollouts: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    """Nearest eval/checkpoint/final boundary for launch admission."""
    boundary = num_rollouts
    for interval in (eval_interval, save_interval):
        if interval > 0 and trained_batches < num_rollouts:
            boundary = min(boundary, ((trained_batches // interval) + 1) * interval)
    return boundary


def unwrap_replicated_int(value: object, *, name: str) -> int:
    """Normalize a BROADCAST return and verify all worker replicas agree."""
    if isinstance(value, (list, tuple)):
        if not value or any(not isinstance(item, int) for item in value):
            raise TypeError(f"{name} returned invalid worker values: {value!r}")
        if any(item != value[0] for item in value[1:]):
            raise RuntimeError(f"{name} disagrees across workers: {value!r}")
        return value[0]
    if not isinstance(value, int):
        raise TypeError(f"{name} returned {type(value).__name__}, expected int")
    return value


__all__ = [
    "AsyncBatchControl",
    "log_admission_notes",
    "max_publication_gap_batches",
    "next_hard_boundary",
    "unwrap_replicated_int",
]
