"""Async diffusion RL trainer — disaggregated train/rollout slabs for DiT.

Diffusion sibling of :class:`~unirl.trainer.async_ar.AsyncARTrainer`. It subclasses
:class:`~unirl.trainer.diffusion.DiffusionTrainer` with ``layout="separate"`` to
REUSE its two-slab build (train slab + dedicated rollout engine slab), the
cross-slab weight-sync wiring (``RemoteLoraWeightSync`` for the BAGEL recipe;
``NCCLWeightSync`` is also supported by ``_connect_separate``) and the diffusion
plumbing (``_build_request_sample`` / ``_drop_decoded`` / ``evaluate`` /
checkpoint / FlowGRPO ``stack.train_track``).

The async loop runs over the shared
:class:`~unirl.rollout.engine.asynchronous.AsyncBatchRolloutEngine` (the same engine
``AsyncARTrainer`` drives) — one single-threaded driver loop over non-blocking
Ray dispatch, no producer thread and no locks. This trainer supplies only the
diffusion hooks:

* ``_build_async_sample`` — one data batch → one request ``Sample``.
* ``_score_completed`` — reward at reap time, then split into tree-complete
  groups. Generation overlaps training; reward scoring itself does not.
* ``_advantage_and_train`` — advantage + FlowGRPO optimizer step over the
  freshest ``batch_size`` groups; it never calls the reward.

Two numeric knobs (identical semantics to AsyncARTrainer):
  * ``max_inflight`` — must be ``1`` so a reap-time transfer never competes with
    a queued generation on the rollout workers.
  * ``buffer_max_staleness`` — regular rollout-weight syncs a buffered group may
    cross. ``0`` (default) never crosses a sync; ``>0`` enables a bounded
    policy-lag buffer.

``_next_step`` polls (reaps) BEFORE topping up launches, which is what makes the
overlap fast here: reaping a generation pulls its trajectory segment off the
rollout slab (the reward's cross-slab localize, an NCCL send issued on the
rollout workers), so a generation launched ahead of that send blocks it —
measured ~150s/rollout on BAGEL instead of ~8s. Reaping first hands the send
idle workers, and the launch that follows still happens before the step returns,
so the next generation overlaps this step's training.

Draining all in-flight generations before each weight sync is MANDATORY (a
weight + KV update corrupts an in-flight generation); that is the
single-threaded ``_drain_all`` quiesce.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, List, Optional, Tuple

import torch

from unirl.distributed.tensor import hydrate
from unirl.rollout.engine.asynchronous import AsyncBatchRolloutEngine, launch_ceiling
from unirl.train.stack import TrainStepResult
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class AsyncDiffusionTrainer(DiffusionTrainer):
    """Disaggregated async diffusion trainer (two slabs, resident engine, cross-slab sync)."""

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        buffer_max_staleness: Optional[int] = None,
        **diffusion_kwargs: Any,
    ) -> None:
        layout = diffusion_kwargs.setdefault("layout", "separate")
        if layout != "separate":
            raise ValueError(f"AsyncDiffusionTrainer requires layout='separate', got {layout!r}.")
        max_inflight = int(max_inflight)
        if max_inflight != 1:
            raise ValueError(
                "AsyncDiffusionTrainer requires max_inflight=1: multiple queued generations "
                "block the reap-time cross-slab transfer on the rollout workers; "
                f"got {max_inflight}."
            )
        super().__init__(**diffusion_kwargs)

        if self.weight_sync is None:
            raise ValueError(
                "AsyncDiffusionTrainer requires a cross-slab weight sync; add a `sync:` block to the recipe."
            )

        self._max_inflight = max_inflight
        self._buffer_max_staleness = buffer_max_staleness

    def _build_async_sample(self, gen_id: int) -> Sample:
        """Consume one data batch and build the request Sample for ``gen_id``."""
        return self._build_request_sample(self.data_source.get_samples(self.batch_size), gen_id)

    def _score_completed(self, gen_id: int, completed: Sample) -> List[Sample]:
        """Score a completed Sample and split it into tree-complete groups.

        Runs at reap time inside the engine — before the next launch and before
        training consumes the batch — and must precede ``_drop_decoded`` (the
        reward reads the decoded primitive). Keyed by ``gen_id`` so media panels
        behave like the synchronous path. The filled ``Sample`` is
        self-contained (it carries its input Parts).
        """
        scored = self.reward.score_and_attach(completed)
        self._drop_decoded(scored, rollout_id=gen_id)
        return scored.split()

    def _drain_all(self) -> None:
        """Finish + buffer EVERY in-flight generation (the single-threaded quiesce).

        Mandatory before a weight sync (a weight + KV update corrupts an in-flight
        generate), before eval/checkpoint (shared engine), and in ``finally`` (no
        leaked ObjectRefs).
        """
        self._async_engine.quiesce()

    def _advantage_and_train(
        self,
        sample: Sample,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer step for a SCORED ``Sample`` (rewards already attached)."""
        if t0 is None:
            t0 = time.perf_counter()
        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            part.rewards = hydrate(part.rewards)
            if isinstance(part.component_rewards, dict):
                part.component_rewards = {name: hydrate(value) for name, value in part.component_rewards.items()}
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
        part = part.compute_advantages(normalize=True, use_global_std=self._adv_use_global_std)
        sample = sample.replace_frontier(part)
        result = self.stack.train_track(sample.parts[-1], training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(rollout_id, result, sample, step_time_s=time.perf_counter() - t0)
        self._reset_transport_buffers()
        return result, mean_reward

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
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        M = self._max_inflight

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "max_inflight": M,
                "buffer_max_staleness": stale,
                "weight_sync_interval": interval,
                "train_fraction": self._train_fraction,
            },
        )

        self._async_engine = AsyncBatchRolloutEngine(
            self.rollout,
            complete=self._score_completed,
            start_gen_id=start_rollout,
        )

        if resumed and self.weight_sync is not None:
            self.weight_sync.sync()
        if self.eval_interval > 0:
            self.evaluate(start_rollout, sync_weights=False, sleep_after=False)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                picked = self._next_step(rollout_id, interval, M, stale, num_rollouts)
                sample = Sample.concat(picked)
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    sample, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    self._drain_all()
                    self.evaluate(step, sync_weights=False, sleep_after=False)
                if save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts):
                    self._drain_all()
                    self.maybe_save_checkpoint(
                        rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                    )
                if step % interval == 0 and self.weight_sync is not None:
                    self._drain_all()
                    self.weight_sync.sync()
                    self._async_engine.bump_weight_version()
        finally:
            # Cleanup failures must not mask the exception that stopped training.
            active_exception = sys.exc_info()[0] is not None
            try:
                self._drain_all()
            except Exception:
                if not active_exception:
                    raise
                logger.exception("Failed to drain in-flight generations during async diffusion teardown")
            finally:
                self._finish_wandb()

    def _next_step(
        self,
        rollout_id: int,
        interval: int,
        M: int,
        stale: int,
        num_rollouts: int,
    ) -> List[Sample]:
        """Reap completed generations, top up launches, and return the freshest
        ``batch_size`` scored group Samples for ``rollout_id`` (blocking on the
        oldest in-flight generation if the buffer is short).

        Polls BEFORE topping up: reaping pulls the trajectory segment off the
        rollout slab, so it must not queue behind a freshly launched generation,
        and the post-reap launch is what overlaps this step (module docstring).

        The launch clamp is the load-bearing on-policy guarantee: a generation
        launched now is consumed later, so bound how far ahead we launch to
        ``stale`` weight-syncs. ``stale=0`` ⇒ never launch into a future
        sync-window ⇒ no generation crosses a regular rollout-weight sync.
        """
        engine = self._async_engine
        while True:
            ceiling = launch_ceiling(rollout_id, sync_interval=interval, max_staleness=stale, num_rollouts=num_rollouts)
            engine.poll()
            while engine.next_gen_id < ceiling and engine.inflight < M:
                engine.submit(self._build_async_sample(engine.next_gen_id))
            picked = engine.drain_freshest(self.batch_size, max_staleness=stale)
            engine.pop_evicted()
            if picked is not None:
                return picked
            if engine.inflight:
                engine.wait_oldest()
            else:
                raise RuntimeError("async rollout buffer underflow with no in-flight generations")
