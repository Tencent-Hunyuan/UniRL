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


def boundary_launch_slots(
    *,
    inflight_count: int,
    ready_count: int,
    max_inflight: int,
    trained_batches: int,
    num_rollouts: int,
    hard_boundary: int,
    batches_since_sync: int,
    weight_sync_interval: int,
) -> int:
    """Generations admissible now, bounded by concurrency, remaining batches, and the sync window."""
    freshness = weight_sync_interval - batches_since_sync
    allowed = min(freshness, min(num_rollouts, hard_boundary) - trained_batches)
    return max(0, min(max_inflight - inflight_count, allowed - inflight_count - ready_count))


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


class AsyncRolloutTrainerMixin:
    """One async batch loop shared by ``AsyncARTrainer`` and ``AsyncDiffusionTrainer``."""

    def _async_wandb_extra(self) -> Dict[str, object]:
        """Trainer-specific keys merged into the wandb run config."""
        return {}

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        """Run the trainer's evaluation at a synced, empty rollout boundary."""
        raise NotImplementedError

    def _build_async_sample(self, gen_id: int) -> "Sample":
        """Consume one data batch and build the request Sample for ``gen_id``."""
        return self._build_request_sample(self.data_source.get_samples(self.batch_size), gen_id)

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
                "weight_sync_interval": self._weight_sync_interval,
                "max_staleness": self._weight_sync_interval - 1,
                "staleness_budget": staleness_budget,
                "num_updates_per_batch": self._num_updates_per_batch,
                **self._async_wandb_extra(),
            },
        )

        self._rollout_manager = RolloutManager(
            self.rollout,
            launchers=[lambda sample: self.rollout.launch_nowait("generate", sample)],
            capacities=[self._max_inflight],
            group_size=total_samples_per_prompt(self.sampling_params),
            filter_fn=keep_within_lag(staleness_budget),
        )
        self._next_generation_id = start_rollout

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
            slots = boundary_launch_slots(
                inflight_count=0,
                ready_count=0,
                max_inflight=self._max_inflight,
                trained_batches=rollout_id,
                num_rollouts=num_rollouts,
                hard_boundary=hard_boundary,
                batches_since_sync=self._batches_since_sync,
                weight_sync_interval=self._weight_sync_interval,
            )
            self._submit_generations(slots)

        groups = manager.collect(self.batch_size, current_version=self._train_version)
        completed, gen_id, output_version = combine_rollout_chunks(groups)
        scored = self._score_completed(gen_id, completed)

        inflight_count, ready_count = manager.counts
        slots = boundary_launch_slots(
            inflight_count=inflight_count,
            ready_count=ready_count + 1,
            max_inflight=self._max_inflight,
            trained_batches=rollout_id,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
            batches_since_sync=self._batches_since_sync,
            weight_sync_interval=self._weight_sync_interval,
        )
        self._submit_generations(slots)
        return scored, output_version

    def _submit_generations(self, count: int) -> None:
        for _ in range(count):
            gen_id = self._next_generation_id
            self._rollout_manager.submit([self._build_async_sample(gen_id)])
            self._next_generation_id += 1


__all__ = [
    "AsyncRolloutTrainerMixin",
    "boundary_launch_slots",
    "combine_rollout_chunks",
    "next_hard_boundary",
    "rollout_version_metrics",
    "training_version_metrics",
]
