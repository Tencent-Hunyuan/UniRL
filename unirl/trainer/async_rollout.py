"""Shared driver-side policy for the async batch trainers (AR and diffusion)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

from unirl.rollout.manager import (
    RolloutManager,
    keep_within_lag,
    required_worker_concurrency,
    validate_worker_inflight,
)
from unirl.trainer.base import unwrap_replicated_int
from unirl.types.sampling import total_samples_per_prompt

if TYPE_CHECKING:
    from unirl.rollout.manager import RolloutSnapshot
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


def resolve_separate_worker_concurrency(
    *,
    num_devices: int,
    train_fraction: float,
    per_worker_inflight: int,
    configured_concurrency: Optional[int],
    engine_concurrency: Optional[int],
) -> Tuple[int, int | List[int]]:
    """Resolve serial train workers and concurrent rollout workers before pool setup."""
    train_devices = round(train_fraction * num_devices)
    if train_devices <= 0 or train_devices >= num_devices:
        raise ValueError(
            f"train_fraction={train_fraction} yields {train_devices} train devices "
            f"of {num_devices}; must leave a non-empty rollout slab."
        )
    rollout_concurrency = (
        required_worker_concurrency(per_worker_inflight) if configured_concurrency is None else configured_concurrency
    )
    engine_concurrency = None if engine_concurrency is None else int(engine_concurrency)
    validate_worker_inflight(
        per_worker_inflight,
        worker_max_concurrency=rollout_concurrency,
        engine_concurrency=engine_concurrency,
    )
    worker_concurrency = (
        [1] * train_devices + [rollout_concurrency] * (num_devices - train_devices)
        if configured_concurrency is None
        else rollout_concurrency
    )
    return train_devices, worker_concurrency


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


def _admission_headrooms(
    *,
    outstanding_prompts: int,
    max_inflight_prompts: int,
    batch_size: int,
    trained_batches: int,
    hard_boundary: int,
    batches_since_sync: int,
    max_staleness_batches: int,
) -> tuple[int, int, int]:
    """Capacity, freshness, and hard-boundary headroom in prompts, each non-negative."""
    if outstanding_prompts % batch_size:
        raise RuntimeError(
            f"outstanding prompt count must stay batch-aligned to {batch_size}; got {outstanding_prompts}"
        )
    # A newly launched prompt uses the currently published weights. It may be
    # consumed now (lag=batches_since_sync), then one batch later, and so on.
    # Keep enough horizon for cross-publication carry when the configured lag
    # budget permits it; unlike a sync boundary, eval/save/final remain hard.
    freshness_batches = max(0, max_staleness_batches - batches_since_sync + 1)
    remaining_batches = max(0, hard_boundary - trained_batches)
    return (
        max(0, max_inflight_prompts - outstanding_prompts),
        max(0, freshness_batches * batch_size - outstanding_prompts),
        max(0, remaining_batches * batch_size - outstanding_prompts),
    )


def _rollout_manager_metrics(
    snapshot: RolloutSnapshot,
    *,
    train_version: int,
    phase: Literal["pre_sync", "step_end"],
) -> Dict[str, Optional[float]]:
    """Map a native batch snapshot to fixed-cardinality prompt gauges; see README.md."""
    pool = snapshot.pool
    oldest = snapshot.oldest_ready_output_version
    metrics = {
        "queued_prompts": pool.queued_tasks,
        "ready_groups": snapshot.ready_groups,
        "max_ready_lag_updates": None if oldest is None else train_version - oldest,
    }
    if phase == "pre_sync":
        metrics.update(
            launch_reserved_prompts=pool.launch_reserved_tasks,
            running_prompts=pool.running_tasks,
            completed_unrouted_prompts=pool.completed_unrouted_tasks,
            slot_occupancy=(pool.running_tasks + pool.launch_reserved_tasks) / pool.capacity_tasks,
        )
    return {f"async/{phase}_{key}": value for key, value in metrics.items()}


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


def combine_rollout_prompts(groups: List[List["Sample"]]) -> Tuple["Sample", int]:
    """Combine dynamically completed prompt trees from one behavior-policy version."""
    chunks = [sample for group in groups for sample in group]
    if not chunks:
        raise ValueError("cannot combine an empty rollout result")

    from unirl.types.sample import Sample

    combined = Sample.concat(chunks)
    versions = {part.output_version for part in combined.gen_parts()}
    if len(versions) != 1 or None in versions:
        raise RuntimeError(f"rollout batch needs one output_version across its generated parts; got {versions}")
    return combined, next(iter(versions))


class AsyncRolloutTrainerMixin:
    """One async batch loop shared by ``AsyncARTrainer`` and ``AsyncDiffusionTrainer``."""

    def _async_wandb_extra(self) -> Dict[str, object]:
        """Trainer-specific keys merged into the wandb run config."""
        return {}

    def _refill_before_score(self) -> bool:
        """Whether replacement rollout work may start before reward scoring."""
        return False

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        """Run the trainer's evaluation at a synced, empty rollout boundary."""
        raise NotImplementedError

    def _score_completed(self, rollout_id: int, completed: "Sample") -> "Sample":
        scored = self.reward.score_and_attach(completed)
        self._drop_decoded(scored, rollout_id=rollout_id)
        return scored

    def _train_async_loop(
        self,
        *,
        num_rollouts: int,
        save_interval: int,
        save_dir: Optional[str],
        load_dir: Optional[str],
        save_mode: str,
    ) -> None:
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        self._train_version = unwrap_replicated_int(
            self.backend.get_optimizer_step_count(),
            name="backend optimizer step count",
        )
        self._batches_since_sync = 0
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        staleness_budget = self._max_staleness * self._num_updates_per_batch
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "max_inflight": self._max_inflight,
                "max_inflight_prompts": self._max_inflight_prompts,
                "per_worker_inflight": self._per_worker_inflight,
                "weight_sync_interval": self._weight_sync_interval,
                "max_staleness": self._max_staleness,
                "staleness_budget": staleness_budget,
                "num_updates_per_batch": self._num_updates_per_batch,
                **self._async_wandb_extra(),
            },
        )

        self._next_generation_id = start_rollout
        engine_slots = self.rollout.engine_slots
        launchers = [lambda sample, slot=slot: slot.launch("generate_on_slot", sample) for slot in engine_slots]
        self._rollout_manager = RolloutManager(
            self.rollout,
            launchers=launchers,
            capacities=[self._per_worker_inflight] * len(engine_slots),
            group_size=total_samples_per_prompt(self.sampling_params),
            filter_fn=keep_within_lag(staleness_budget),
        )

        if resumed or self.eval_interval > 0:
            self._sync_rollout(force=True, require_empty=True)
        if self.eval_interval > 0:
            self._boundary_evaluate(start_rollout, initial=True)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                self._admitted_prompts_delta = 0
                self._carry_resubmitted_prompts_delta = 0
                self._rollout_metrics: Dict[str, Optional[float]] = {}
                hard_boundary = next_hard_boundary(
                    rollout_id,
                    num_rollouts=num_rollouts,
                    eval_interval=self.eval_interval,
                    save_interval=save_interval,
                )
                sample, output_version = self._next_rollout_batch(
                    rollout_id,
                    hard_boundary=hard_boundary,
                )
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    sample,
                    training_progress=training_progress,
                    rollout_id=rollout_id,
                    t0=t0,
                    extra_metrics=rollout_version_metrics(
                        train_version=self._train_version,
                        output_version=output_version,
                        num_updates_per_batch=self._num_updates_per_batch,
                    ),
                )
                if self.wandb_logger.enabled:
                    self._rollout_metrics.update(
                        _rollout_manager_metrics(
                            self._rollout_manager.snapshot(), train_version=self._train_version, phase="pre_sync"
                        )
                    )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                eval_due = self.eval_interval > 0 and step % self.eval_interval == 0
                save_due = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                sync_due = step < num_rollouts and self._batches_since_sync >= self._weight_sync_interval
                if eval_due or save_due or sync_due:
                    self._sync_rollout(require_empty=eval_due or save_due)

                if step >= num_rollouts and not self._rollout_manager.empty:
                    raise RuntimeError("final rollout boundary requires an empty RolloutManager")

                if self.wandb_logger.enabled:
                    snapshot, delta = self._rollout_manager.drain_metrics()
                    self._rollout_metrics.update(
                        _rollout_manager_metrics(snapshot, train_version=self._train_version, phase="step_end")
                    )
                    self._rollout_metrics.update(
                        {
                            "async/capacity_prompts": snapshot.pool.capacity_tasks,
                            "async/launched_prompts_delta": delta.launched_tasks,
                            "async/completed_prompts_delta": delta.completed_tasks,
                            "async/dropped_groups_delta": delta.dropped_groups,
                            "async/carried_prompts_delta": delta.carried_tasks,
                            "async/finished_prompts_delta": delta.finished_tasks,
                            "async/admitted_prompts_delta": self._admitted_prompts_delta,
                            "async/carry_resubmitted_prompts_delta": self._carry_resubmitted_prompts_delta,
                        }
                    )
                    self.wandb_logger.log_rollout(step, self._rollout_metrics)

                if eval_due:
                    self._boundary_evaluate(rollout_id, initial=False)
                if save_due:
                    self.maybe_save_checkpoint(
                        rollout_id,
                        num_rollouts,
                        save_interval=save_interval,
                        save_dir=save_dir,
                        save_mode=save_mode,
                    )
        finally:
            try:
                self._rollout_manager.close()
            finally:
                self._finish_wandb()

    def _sync_rollout(self, *, force: bool = False, require_empty: bool = False) -> None:
        manager = self._rollout_manager
        needs_publish = force or manager.published_version != self._train_version
        if not needs_publish:
            if require_empty and not manager.empty:
                raise RuntimeError("eval/checkpoint boundary requires an empty RolloutManager")
            self._batches_since_sync = 0
            return

        # Quiesce engine work before weight publication, but keep completed
        # groups in the manager. The lag filter decides whether buffered work
        # remains trainable after publication, matching continuous prompt carry.
        carried = manager.quiesce(current_version=self._train_version)
        if require_empty and (carried or not manager.empty):
            raise RuntimeError("eval/checkpoint boundary requires an empty RolloutManager")
        if not require_empty:
            # A publication must not split one training batch across behavior
            # policy versions. Finish partially generated carry in full, or
            # only the pristine prefix needed to align the ready buffer.
            started = any(
                any(part.output_version is not None for part in sample.gen_parts())
                or sample.parts[-1].harness_status == "suspended"
                for sample in carried
            )
            if started:
                manager.finish(carried, current_version=self._train_version)
                carried = []
            _, ready_count = manager.counts
            prompts_to_finish = (-ready_count) % self.batch_size
            if prompts_to_finish:
                if len(carried) < prompts_to_finish:
                    raise RuntimeError(
                        "cannot batch-align rollout carry before weight publication: "
                        f"ready={ready_count}, carried={len(carried)}, batch_size={self.batch_size}"
                    )
                finishing = carried[:prompts_to_finish]
                carried = carried[prompts_to_finish:]
                manager.finish(finishing, current_version=self._train_version)
            inflight_count, ready_count = manager.counts
            if inflight_count or ready_count % self.batch_size:
                raise RuntimeError(
                    "rollout carry remained unaligned after completing the publication prefix: "
                    f"inflight={inflight_count}, ready={ready_count}, batch_size={self.batch_size}"
                )
        manager.sync_weights(self.weight_sync, output_version=self._train_version)
        if carried:
            manager.submit(carried)
            self._carry_resubmitted_prompts_delta += len(carried)
        self._batches_since_sync = 0

    def _next_rollout_batch(
        self,
        rollout_id: int,
        *,
        hard_boundary: int,
    ) -> Tuple["Sample", int]:
        self._admit_prompts(
            phase="collect",
            trained_batches=rollout_id,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync,
        )

        manager = self._rollout_manager
        groups = manager.collect(
            self.batch_size,
            current_version=self._train_version,
        )
        completed, output_version = combine_rollout_prompts(groups)

        scored = None
        if not self._refill_before_score():
            scored = self._score_completed(rollout_id, completed)

        # Consuming a batch releases capacity immediately. Refill before reward
        # for AR and before training for trainers that require reap-time scoring.
        self._admit_prompts(
            phase="refill",
            trained_batches=rollout_id + 1,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync + 1,
        )

        if scored is None:
            scored = self._score_completed(rollout_id, completed)

        return scored, output_version

    def _admit_prompts(
        self,
        *,
        phase: Literal["collect", "refill"],
        trained_batches: int,
        hard_boundary: int,
        batches_since_sync: int,
    ) -> None:
        inflight_count, ready_count = self._rollout_manager.counts
        headrooms = _admission_headrooms(
            outstanding_prompts=inflight_count + ready_count,
            max_inflight_prompts=self._max_inflight_prompts,
            batch_size=self.batch_size,
            trained_batches=trained_batches,
            hard_boundary=hard_boundary,
            batches_since_sync=batches_since_sync,
            max_staleness_batches=self._max_staleness,
        )
        prompts = min(headrooms)
        self._submit_prompts(prompts)
        if self.wandb_logger.enabled:
            self._rollout_metrics.update(
                {
                    f"async/admission_{phase}_capacity_headroom_prompts": headrooms[0],
                    f"async/admission_{phase}_freshness_headroom_prompts": headrooms[1],
                    f"async/admission_{phase}_boundary_headroom_prompts": headrooms[2],
                }
            )

    def _submit_prompts(self, count: int) -> None:
        batch_count, remainder = divmod(count, self.batch_size)
        if remainder:
            raise RuntimeError(f"prompt admission must submit whole batches of {self.batch_size}; got {count} prompts")
        for _ in range(batch_count):
            request = self._build_request_sample(
                self.data_source.get_samples(self.batch_size),
                self._next_generation_id,
            )
            self._next_generation_id += 1
            prompts = request.split()
            if len(prompts) != self.batch_size:
                raise RuntimeError(f"request batch split into {len(prompts)} prompts; expected {self.batch_size}")
            self._rollout_manager.submit(prompts)
            self._admitted_prompts_delta += len(prompts)


__all__ = [
    "AsyncRolloutTrainerMixin",
    "combine_rollout_prompts",
    "next_hard_boundary",
    "resolve_separate_worker_concurrency",
    "rollout_version_metrics",
    "training_version_metrics",
]
