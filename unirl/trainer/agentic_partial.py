"""Colocate partial-rollout agentic trainers (LIN-531).

The **synchronous / colocate** driver of partial rollout — the missing quadrant next to the
barrier `AgenticTrainer` (colocate, waits for every trajectory) and the fully-async
`AsyncAgenticTrainer` (disaggregated slabs). Train and rollout still **time-share each GPU**
(`sleep()`/`wake_up()`), single loop, one optimizer step per rollout; the only change from the
barrier is *how the rollout completes*:

- **Over-sample** `oversample_batch_size` prompt-groups per drive, **commit the freshest
  `batch_size` complete GRPO groups**, and **`abort` the in-flight tail at a turn boundary**
  instead of draining every trajectory. The slow straggler no longer gates the round.
- **Tail policy** — `carry` re-submits the checkpointed tail next round when the environment can
  reconstruct its state from the `Sample` (current stateless tools); `drop` discards it for
  **stateful** envs like ALFWorld, whose `reset` starts a fresh episode.

Motivation (LIN-531 ALFWorld comparison): the fully-async trainer *lost* on ALFWorld because
disaggregation halves the generation GPUs; colocate+partial keeps all GPUs for generation while
still cutting the straggler tail — the slime "over-sample + abort + recycle" / verl `bypass_mode`
pattern. The driver-side :class:`~unirl.rollout.engine.asynchronous.AsyncAgenticRolloutEngine`
(submit/poll/finalize_if_drained/quiesce + group assembly and versioned buffering) is shared with
``AsyncAgenticTrainer``; only the colocate wake/sync/sleep choreography is new.

Correctness: `TensorWeightSync.sync` writes the live SRT weight pool, so it must run **awake +
decode-idle** — sync sits at the top (post-`wake_up`, pre-`submit`, barrier parity) and `abort`
(not `sleep`) provides the pre-sleep quiesce. The off-policy carried tail is corrected per-token
(each gen Part keeps its own `output_version` + logprobs); `buffer_max_staleness` is converted
to an update-version budget for completed groups.
"""

from __future__ import annotations

import logging
import sys
import time
from collections import Counter
from typing import Dict, List, Literal, Optional

from unirl.rollout.engine.asynchronous import AsyncAgenticRolloutEngine, root_of
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.agentic_env import _EnvRewardSource
from unirl.trainer.async_batch_control import unwrap_replicated_int
from unirl.types.sample import Part, Sample

logger = logging.getLogger(__name__)


class AgenticPartialTrainer(AgenticTrainer):
    """Colocate partial-rollout trainer (over-sample → commit-N → abort tail → carry/drop)."""

    _POLL_INTERVAL_S = 0.1
    _MAX_REFILLS = 64  # underflow guard: refills of a drained-but-short buffer before giving up

    def __init__(
        self,
        *,
        samples_per_prompt: int,
        oversample_batch_size: Optional[int] = None,
        buffer_max_staleness: Optional[int] = None,
        tail_policy: Literal["carry", "drop"] = "carry",
        **kwargs,
    ) -> None:
        num_updates_per_batch = int(kwargs["stack_cfg"].get("num_updates_per_batch", 1))
        super().__init__(**kwargs)
        self._n = samples_per_prompt
        self._oversample = oversample_batch_size or self.batch_size
        if self._oversample < self.batch_size:
            raise ValueError(f"oversample_batch_size={self._oversample} must be >= batch_size={self.batch_size}")
        self._buffer_max_staleness = buffer_max_staleness
        self._num_updates_per_batch = num_updates_per_batch
        self._tail_policy = tail_policy
        if self._tail_policy not in ("carry", "drop"):
            raise ValueError(f"tail_policy must be 'carry' or 'drop'; got {self._tail_policy!r}")
        self._drive_seq = 0
        self._last_dropped_trajectories = 0
        self._last_dropped_roots = 0
        self._last_discarded_completed_trajectories = 0
        self._gt_by_root: Dict[str, Optional[str]] = {}
        self._carried: List[Sample] = []

    def _build_tasks(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """A drive's task list: uniquely namespaced fresh siblings + carried partials."""
        self._drive_seq += 1
        fresh = self._build_request_sample(self.data_source.get_samples(self._oversample), rollout_id)
        fresh = fresh.map_sample_ids(lambda sample_id: f"d{self._drive_seq}:{sample_id}")
        root = fresh.parts[0]
        root_meta = root.metadata or [None] * len(root.sample_ids)
        for sid, md in zip(root.sample_ids, root_meta):
            self._gt_by_root[sid] = (md or {}).get("answer")
        tasks = [prompt for prompt in fresh.split() for _ in range(self._n)]
        tasks.extend(carried)
        return tasks

    def _reconstruct_request(self, trajs: List[Sample]) -> Sample:
        """A request whose root Part carries every trajectory's root id + ground-truth answer,
        so the inherited answer-grader (`_rewards_and_groups`) works unchanged. The env-reward
        subclass ignores it."""
        roots = [tr.parts[0].sample_ids[0] for tr in trajs]
        return Sample.request(Part.input(roots, metadata=[{"answer": self._gt_by_root.get(r)} for r in roots]))

    def _apply_tail_policy(self, carried: List[Sample], rollout_id: int) -> None:
        """Store safe carried tails or purge abandoned roots and their siblings."""
        self._last_dropped_trajectories = 0
        self._last_dropped_roots = 0
        self._last_discarded_completed_trajectories = 0
        if self._tail_policy == "carry":
            self._carried = carried
            return

        roots = {root_of(sample) for sample in carried}
        self._last_dropped_trajectories = len(carried)
        self._last_dropped_roots = len(roots)
        self._last_discarded_completed_trajectories = self._engine.discard_roots(roots)
        for root in roots:
            self._gt_by_root.pop(root, None)
        self._carried = []
        logger.info(
            "rollout %d partial: dropped %d tail trajectories across %d roots; discarded %d completed siblings",
            rollout_id,
            self._last_dropped_trajectories,
            self._last_dropped_roots,
            self._last_discarded_completed_trajectories,
        )

    def _drain_buffer(self, n: int, *, staleness_budget: int) -> Optional[List[List[Sample]]]:
        """Drain fresh groups and forget ground truth for stale evictions."""
        picked = self._engine.drain_freshest(n, staleness_budget=staleness_budget)
        for group in self._engine.pop_evicted():
            if group:
                self._gt_by_root.pop(root_of(group[0]), None)
        return picked

    def _collect_until(
        self,
        batch_size: int,
        rollout_id: int,
        staleness_budget: int,
    ) -> List[List[Sample]]:
        """Pump the in-flight drive until the buffer holds ``batch_size`` complete groups within
        the staleness bound, then drain the freshest. If the drive drains without filling the
        buffer (small over-sample / failures / eviction), refill with a fresh drive."""
        refills = 0
        while True:
            self._engine.poll()
            picked = self._drain_buffer(batch_size, staleness_budget=staleness_budget)
            if picked is not None:
                return picked
            if self._engine.finalize_if_drained() is None:
                time.sleep(self._POLL_INTERVAL_S)
                continue
            picked = self._drain_buffer(batch_size, staleness_budget=staleness_budget)
            if picked is not None:
                return picked

            refills += 1
            if refills > self._MAX_REFILLS:
                raise RuntimeError(
                    f"colocate-partial rollout {rollout_id}: buffer underflow after {refills} refills "
                    f"(buffer={self._engine.buffered_groups()} < batch={batch_size}); raise oversample_batch_size "
                    f"or buffer_max_staleness."
                )
            self._engine.submit(self._build_tasks([], rollout_id))

    def _drive_partial(
        self,
        rollout_id: int,
        sync_weights: bool,
        staleness_budget: int,
    ) -> List[List[Sample]]:
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            train_version = unwrap_replicated_int(
                self.backend.get_optimizer_step_count(),
                name="backend optimizer step count",
            )
            self._engine.sync_weights(self.weight_sync, train_version=train_version)
        tasks = self._build_tasks(self._carried, rollout_id)
        self._carried = []
        self._engine.submit(tasks)
        groups = self._collect_until(self.batch_size, rollout_id, staleness_budget)
        carried = self._engine.quiesce()
        self.rollout.sleep()
        tail_depths = [len(t.gen_parts()) for t in carried]
        logger.info(
            "rollout %d partial: committed %d groups; %s tail=%d trajectories, turns=%s",
            rollout_id,
            len(groups),
            self._tail_policy,
            len(carried),
            dict(sorted(Counter(tail_depths).items())),
        )
        self._apply_tail_policy(carried, rollout_id)
        return groups

    # Train loop — override ARTrainer.train (the tail must carry across rollouts)

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        interval = max(1, weight_sync_interval)
        max_staleness_syncs = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        # buffer_max_staleness counts weight syncs; the buffer compares optimizer-update versions.
        staleness_budget_updates = max_staleness_syncs * interval * self._num_updates_per_batch

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self._oversample)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "oversample_batch_size": self._oversample,
                "buffer_max_staleness_intervals": max_staleness_syncs,
                "buffer_staleness_budget_updates": staleness_budget_updates,
                "tail_policy": self._tail_policy,
                "weight_sync_interval": interval,
            },
        )

        self._engine = AsyncAgenticRolloutEngine(self.rollout, group_size=self._n, start_gen_id=start_rollout)
        self._carried = []

        try:
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                training_progress = rollout_id / max(1, num_rollouts - 1)
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )

                groups = self._drive_partial(
                    rollout_id,
                    sync_weights,
                    staleness_budget_updates,
                )
                trajs: List[Sample] = [t for group in groups for t in group]
                versions = [part.output_version for traj in trajs for part in traj.gen_parts()]
                version_metrics: Dict[str, int] = {"partial/published_version": self._engine.version}
                if versions and all(version is not None for version in versions):
                    head_version = min(versions)
                    version_metrics.update(
                        {
                            "partial/head_output_version": head_version,
                            "partial/head_staleness_updates": self._engine.version - head_version,
                            "partial/output_version_span_updates": max(versions) - head_version,
                        }
                    )
                rewards, group_ids = self._rewards_and_groups(self._reconstruct_request(trajs), trajs, rollout_id)
                for root in {root_of(traj) for traj in trajs}:
                    self._gt_by_root.pop(root, None)
                result, mean_reward = self._advantage_train_and_log(
                    trajs,
                    rewards,
                    group_ids,
                    rollout_id=rollout_id,
                    training_progress=training_progress,
                    t0=t0,
                    extra_metrics={
                        "partial/committed_groups": len(groups),
                        "partial/carried_trajectories": len(self._carried),
                        "partial/dropped_trajectories": self._last_dropped_trajectories,
                        "partial/dropped_roots": self._last_dropped_roots,
                        "partial/discarded_completed_trajectories": self._last_discarded_completed_trajectories,
                        "partial/assembler_pending_roots": self._engine.pending_groups(),
                        "partial/buffer_groups": self._engine.buffered_groups(),
                        **version_metrics,
                    },
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id=rollout_id)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                carried = self._engine.quiesce()
                self._apply_tail_policy(carried, num_rollouts)
            except BaseException:  # noqa: BLE001 — preserve an active training failure
                if active_error:
                    logger.warning("AgenticPartialTrainer cleanup failed", exc_info=True)
                else:
                    raise
            finally:
                self._finish_wandb()


class AgenticEnvPartialTrainer(_EnvRewardSource, AgenticPartialTrainer):
    """Colocate partial-rollout trainer whose reward is the environment's per-trajectory return
    (ALFWorld etc.). Reward SOURCE only (the shared
    :class:`~unirl.trainer.agentic_env._EnvRewardSource`) — the partial-rollout machinery is
    inherited. Recipes use ``tail_policy: drop`` because stateful-episode envs restart a carried
    partial."""


__all__ = ["AgenticPartialTrainer", "AgenticEnvPartialTrainer"]
