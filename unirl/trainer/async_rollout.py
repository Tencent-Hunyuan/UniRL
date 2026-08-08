"""Shared driver-side policy for the async batch trainers (AR and diffusion)."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional, Tuple

from unirl.rollout.manager import RolloutManager, keep_within_lag
from unirl.trainer.base import unwrap_replicated_int
from unirl.types.sampling import total_samples_per_prompt

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class _PromptUnitSource:
    """Split normal data batches into independently dispatched prompt trees."""

    def __init__(
        self,
        data_source: object,
        build_request: Callable[["Sample", int], "Sample"],
        *,
        batch_size: int,
        start_batch_id: int,
    ) -> None:
        self._data_source = data_source
        self._build_request = build_request
        self._batch_size = int(batch_size)
        self._next_batch_id = int(start_batch_id)
        self._units: Deque["Sample"] = deque()

    def take(self, count: int) -> List["Sample"]:
        while len(self._units) < count:
            request = self._build_request(
                self._data_source.get_samples(self._batch_size),
                self._next_batch_id,
            )
            self._next_batch_id += 1
            units = request.split()
            if len(units) != self._batch_size:
                raise RuntimeError(
                    f"request batch split into {len(units)} prompt units; expected {self._batch_size}"
                )
            self._units.extend(units)
        return [self._units.popleft() for _ in range(count)]


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


def boundary_launch_units(
    *,
    outstanding_units: int,
    max_inflight_units: int,
    batch_size: int,
    trained_batches: int,
    num_rollouts: int,
    hard_boundary: int,
    batches_since_sync: int,
    weight_sync_interval: int,
) -> int:
    """Prompt units admissible without crossing a sync or hard boundary."""
    freshness_batches = max(0, weight_sync_interval - batches_since_sync)
    remaining_batches = max(0, min(num_rollouts, hard_boundary) - trained_batches)
    allowed_units = min(freshness_batches, remaining_batches) * batch_size
    return max(0, min(max_inflight_units - outstanding_units, allowed_units - outstanding_units))


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


def combine_rollout_units(
    groups: List[List["Sample"]],
    *,
    require_single_rollout_id: bool = False,
) -> Tuple["Sample", int]:
    """Combine dynamically completed prompt trees sharing one published version."""
    chunks = [sample for group in groups for sample in group]
    if not chunks:
        raise ValueError("cannot combine an empty rollout result")
    if require_single_rollout_id:
        rollout_ids = {_rollout_id(sample) for sample in chunks}
        if len(rollout_ids) != 1:
            raise RuntimeError(
                "rollout batch combines multiple generation ids with incompatible shared schedules: "
                f"{sorted(rollout_ids)}"
            )
    versions = {part.output_version for sample in chunks for part in sample.gen_parts()}
    if not versions or None in versions:
        raise RuntimeError("rollout batch is missing output_version provenance")
    if len(versions) != 1:
        raise RuntimeError(f"rollout batch has mixed output versions: {sorted(versions)}")
    output_version = int(next(iter(versions)))
    if len(chunks) == 1:
        return chunks[0], output_version

    from unirl.types.sample import Sample

    return Sample.concat(chunks), output_version


def _rollout_id(sample: "Sample") -> int:
    if not sample.parts or not sample.parts[0].metadata:
        raise RuntimeError("rollout Sample has no root rollout_id metadata")
    values = {row.get("rollout_id") for row in sample.parts[0].metadata}
    if None in values or len(values) != 1:
        raise RuntimeError(f"rollout Sample must carry one root rollout_id; got {values}")
    return int(next(iter(values)))


class AsyncRolloutTrainerMixin:
    """One async batch loop shared by ``AsyncARTrainer`` and ``AsyncDiffusionTrainer``."""

    def _async_wandb_extra(self) -> Dict[str, object]:
        """Trainer-specific keys merged into the wandb run config."""
        return {}

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        """Run the trainer's evaluation at a synced, empty rollout boundary."""
        raise NotImplementedError

    def _refill_before_score(self) -> bool:
        """Whether this trainer may launch replacement work before scoring."""
        return False

    def _score_completed(self, gen_id: int, completed: "Sample") -> "Sample":
        scored = self.reward.score_and_attach(completed)
        self._drop_decoded(scored, rollout_id=gen_id)
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
        staleness_budget = (self._weight_sync_interval - 1) * self._num_updates_per_batch
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "max_inflight": self._max_inflight,
                "max_inflight_units": self._max_inflight_units,
                "per_worker_inflight": self._per_worker_inflight,
                "weight_sync_interval": self._weight_sync_interval,
                "max_staleness": self._weight_sync_interval - 1,
                "staleness_budget": staleness_budget,
                "num_updates_per_batch": self._num_updates_per_batch,
                **self._async_wandb_extra(),
            },
        )

        self._unit_source = _PromptUnitSource(
            self.data_source,
            self._build_request_sample,
            batch_size=self.batch_size,
            start_batch_id=start_rollout,
        )
        engine_slots = self.rollout.engine_slots
        launchers = [
            lambda sample, slot=slot: slot.launch("generate_one", sample)
            for slot in engine_slots
        ]
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
                hard_boundary = next_hard_boundary(
                    rollout_id,
                    num_rollouts=num_rollouts,
                    eval_interval=self.eval_interval,
                    save_interval=save_interval,
                )
                sample, output_version = self._next_rollout_batch(
                    rollout_id,
                    num_rollouts=num_rollouts,
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
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                eval_due = self.eval_interval > 0 and step % self.eval_interval == 0
                save_due = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                sync_due = step < num_rollouts and self._batches_since_sync >= self._weight_sync_interval
                if eval_due or save_due or sync_due:
                    self._sync_rollout(require_empty=eval_due or save_due)

                if step >= num_rollouts and not self._rollout_manager.empty:
                    raise RuntimeError("final rollout boundary requires an empty RolloutManager")

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

        carried = manager.quiesce(current_version=self._train_version)
        if require_empty and (carried or not manager.empty):
            raise RuntimeError("eval/checkpoint boundary requires an empty RolloutManager")
        if needs_publish:
            manager.sync_weights(self.weight_sync, output_version=self._train_version)
        if carried:
            manager.submit(carried)
        self._batches_since_sync = 0

    def _next_rollout_batch(
        self,
        rollout_id: int,
        *,
        num_rollouts: int,
        hard_boundary: int,
    ) -> Tuple["Sample", int]:
        manager = self._rollout_manager
        inflight_count, ready_count = manager.counts
        if inflight_count + ready_count == 0:
            units = boundary_launch_units(
                outstanding_units=0,
                max_inflight_units=self._max_inflight_units,
                batch_size=self.batch_size,
                trained_batches=rollout_id,
                num_rollouts=num_rollouts,
                hard_boundary=hard_boundary,
                batches_since_sync=self._batches_since_sync,
                weight_sync_interval=self._weight_sync_interval,
            )
            self._submit_prompt_units(units)

        groups = manager.collect(self.batch_size, current_version=self._train_version)
        completed, output_version = combine_rollout_units(
            groups,
            require_single_rollout_id=getattr(self, "_require_single_generation", False),
        )
        scored = self._score_completed(rollout_id, completed)

        inflight_count, ready_count = manager.counts
        units = boundary_launch_units(
            outstanding_units=inflight_count + ready_count,
            max_inflight_units=self._max_inflight_units,
            batch_size=self.batch_size,
            trained_batches=rollout_id + 1,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync + 1,
            weight_sync_interval=self._weight_sync_interval,
        )
        self._submit_prompt_units(units)
        return scored, output_version

    def _submit_prompt_units(self, count: int) -> None:
        if count:
            self._rollout_manager.submit(self._unit_source.take(count))


__all__ = [
    "AsyncRolloutTrainerMixin",
    "boundary_launch_units",
    "combine_rollout_units",
    "next_hard_boundary",
    "rollout_version_metrics",
    "training_version_metrics",
]
