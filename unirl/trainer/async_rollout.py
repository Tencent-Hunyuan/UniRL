"""Shared driver-side policy for the async batch trainers (AR and diffusion)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from unirl.rollout.manager import RolloutManager, keep_within_lag
from unirl.trainer.base import unwrap_replicated_int
from unirl.types.sampling import total_samples_per_prompt

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


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
    max_staleness_batches: int,
) -> int:
    """Prompt units admissible within capacity, staleness, and hard boundaries."""
    if max_inflight_units % batch_size:
        raise ValueError(f"max_inflight_units ({max_inflight_units}) must be divisible by batch_size ({batch_size})")
    if outstanding_units % batch_size:
        raise RuntimeError(
            f"prompt-unit outstanding count must stay batch-aligned to {batch_size}; got {outstanding_units}"
        )
    # A newly launched unit uses the currently published weights. It may be
    # consumed now (lag=batches_since_sync), then one batch later, and so on.
    # Keep enough horizon for cross-publication carry when the configured lag
    # budget permits it; unlike a sync boundary, eval/save/final remain hard.
    freshness_batches = max(0, max_staleness_batches - batches_since_sync + 1)
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
    """Combine dynamically completed prompt trees from one behavior-policy version."""
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

    def _refill_before_score(self) -> bool:
        """Whether replacement rollout work may start before reward scoring."""
        return False

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        """Run the trainer's evaluation at a synced, empty rollout boundary."""
        raise NotImplementedError

    def _score_completed(self, rollout_id: int, completed: "Sample") -> "Sample":
        for attempt in range(2):
            try:
                scored = self.reward.score_and_attach(completed)
                self._drop_decoded(scored, rollout_id=rollout_id)
                return scored
            except Exception:
                if attempt:
                    raise
                logger.warning("reward scoring failed for rollout %d; retrying the same completed batch", rollout_id)
        raise AssertionError("unreachable")

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
        max_staleness = getattr(self, "_max_staleness", self._weight_sync_interval - 1)
        staleness_budget = max_staleness * self._num_updates_per_batch
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "max_inflight": self._max_inflight,
                "max_inflight_units": self._max_inflight_units,
                "per_worker_inflight": self._per_worker_inflight,
                "weight_sync_interval": self._weight_sync_interval,
                "max_staleness": max_staleness,
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

        # Quiesce engine work before weight publication, but keep completed
        # groups in the manager. The lag filter decides whether buffered work
        # remains trainable after publication, matching continuous-unit carry.
        carried = manager.quiesce(current_version=self._train_version)
        if require_empty and (carried or not manager.empty):
            raise RuntimeError("eval/checkpoint boundary requires an empty RolloutManager")
        if not require_empty:
            # A publication must not split one training batch across behavior
            # policy versions. Partially generated trajectories cannot cross a
            # publication at all, so finish their whole carry window. Otherwise
            # finish only the pristine prefix needed to align the ready buffer.
            started = any(
                any(part.output_version is not None for part in sample.gen_parts())
                or (sample.parts and sample.parts[-1].harness_status == "suspended")
                for sample in carried
            )
            if started:
                manager.finish(carried, current_version=self._train_version)
                carried = []
            _, ready_count = manager.counts
            units_to_finish = (-ready_count) % self.batch_size
            if units_to_finish:
                if len(carried) < units_to_finish:
                    raise RuntimeError(
                        "cannot batch-align rollout carry before weight publication: "
                        f"ready={ready_count}, carried={len(carried)}, batch_size={self.batch_size}"
                    )
                finishing = carried[:units_to_finish]
                carried = carried[units_to_finish:]
                manager.finish(finishing, current_version=self._train_version)
            inflight_count, ready_count = manager.counts
            if inflight_count or ready_count % self.batch_size:
                raise RuntimeError(
                    "rollout carry remained unaligned after completing the publication prefix: "
                    f"inflight={inflight_count}, ready={ready_count}, batch_size={self.batch_size}"
                )
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
        units = boundary_launch_units(
            outstanding_units=inflight_count + ready_count,
            max_inflight_units=self._max_inflight_units,
            batch_size=self.batch_size,
            trained_batches=rollout_id,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync,
            max_staleness_batches=self._max_staleness,
        )
        self._submit_prompt_units(units)

        groups = manager.collect(
            self.batch_size,
            current_version=self._train_version,
        )
        completed, output_version = combine_rollout_units(
            groups,
            require_single_rollout_id=getattr(self, "_require_single_generation", False),
        )

        scored = None
        if not self._refill_before_score():
            scored = self._score_completed(rollout_id, completed)

        # Consuming a batch releases capacity immediately. Refill before reward
        # for AR and before training for trainers that require reap-time scoring.
        inflight_count, ready_count = manager.counts
        units = boundary_launch_units(
            outstanding_units=inflight_count + ready_count,
            max_inflight_units=self._max_inflight_units,
            batch_size=self.batch_size,
            trained_batches=rollout_id + 1,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync + 1,
            max_staleness_batches=self._max_staleness,
        )
        self._submit_prompt_units(units)

        if scored is None:
            scored = self._score_completed(rollout_id, completed)

        return scored, output_version

    def _submit_prompt_units(self, count: int) -> None:
        batch_count, remainder = divmod(int(count), self.batch_size)
        if remainder:
            raise RuntimeError(
                f"prompt-unit admission must submit whole batches of {self.batch_size}; got {count} units"
            )
        for _ in range(batch_count):
            request = self._build_request_sample(
                self.data_source.get_samples(self.batch_size),
                self._next_generation_id,
            )
            self._next_generation_id += 1
            units = request.split()
            if len(units) != self.batch_size:
                raise RuntimeError(f"request batch split into {len(units)} prompt units; expected {self.batch_size}")
            self._rollout_manager.submit(units)


__all__ = [
    "AsyncRolloutTrainerMixin",
    "boundary_launch_units",
    "combine_rollout_units",
    "next_hard_boundary",
    "rollout_version_metrics",
    "training_version_metrics",
]
