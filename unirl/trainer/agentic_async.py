"""Fully-async agentic RL trainer — disaggregated producer/consumer (LIN-531).

Sibling of :class:`~unirl.trainer.agentic.AgenticTrainer` (synchronous + *colocated*:
the agentic engine's ``generate`` barrier and the FSDP train shard time-share each
GPU). ``AsyncAgenticTrainer`` places training and rollout on **disjoint GPU slabs**
(like :class:`~unirl.trainer.async_ar.AsyncARTrainer`), keeps the agentic engine
**resident**, pushes weights cross-slab via ``NCCLWeightSync``, and overlaps
multi-turn generation with training.

Mechanism vs policy (LIN-531): the **engine** exposes a ``submit`` / ``poll`` /
``finalize_if_drained`` / ``abort`` interface over a background drain (see
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`), consumed
through the driver-side :class:`~unirl.rollout.manager.AgenticManager`
(group assembly + versioned buffering); this **trainer** owns the *policy* —

* **Producer** — keep the rollout slab saturated: ``submit`` a pool of fresh prompt
  siblings + resumed carried partials and ``poll`` completed trajectories; the facade
  buckets them by root id into complete GRPO groups and stamps each into its
  staleness-bounded versioned buffer.
* **Consumer** — drain the freshest ``batch_size`` complete groups (within
  ``buffer_max_staleness``), reward + GRPO advantage + one optimizer step (reusing
  :class:`AgenticTrainer`'s helpers), then **quiesce + sync**: ``abort`` the in-flight
  tail at a turn boundary, apply the configured ``tail_policy`` (carry only when the
  environment can resume from the ``Sample``; otherwise drop), then
  ``engine.sync_weights`` (one call: push + version bump).

ONE single-threaded loop (the ``AsyncARTrainer`` shape): with disjoint slabs the
rollout slab keeps generating in the background (the engine's per-worker drain) while
the driver polls / trains — concurrency from disaggregation, not driver threads. In
steady state the buffer is refilled *during* the previous train step, so :meth:`_next_batch`
returns without waiting. Staleness bounds how far the producer leads the consumer; the
per-token rollout-anchored ratio corrects the off-policy gap (a carried trajectory whose
turns span weight versions is correct per-token because each gen ``Part`` keeps its own
``weight_version`` + logprobs).

.. note::
   The GPU integration (two-slab placement, NCCL sync, and the train loop) follows
   ``AsyncARTrainer`` and requires an end-to-end GPU run for validation.
"""

from __future__ import annotations

import inspect
import logging
import sys
import time
from typing import Dict, List, Literal, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.rollout.manager import AgenticManager, root_of
from unirl.train.stack import TrainStepResult
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.sample import Part, Sample, _part_with_field
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class AsyncAgenticTrainer(AgenticTrainer):
    """Disaggregated fully-async agentic trainer (two slabs, resident engine, NCCL sync)."""

    _POLL_INTERVAL_S = 0.02
    _MAX_REFILLS = 64  # underflow guard: refills of a drained-but-short buffer before we give up

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        samples_per_prompt: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        stop: Optional[List[str]] = None,
        train_fraction: float = 0.5,
        oversample_batch_size: Optional[int] = None,
        buffer_max_staleness: Optional[int] = None,
        tail_policy: Literal["carry", "drop"] = "carry",
    ) -> None:
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        self.batch_size = int(batch_size)
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.balance_shards = False
        self.eval_interval = 0
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None
        self._stop = list(stop) if stop else ["</tool_call>"]

        self._train_fraction = float(train_fraction)
        self._buffer_max_staleness = buffer_max_staleness
        self._tail_policy = str(tail_policy)
        if self._tail_policy not in ("carry", "drop"):
            raise ValueError(f"tail_policy must be 'carry' or 'drop'; got {self._tail_policy!r}")
        self._drive_seq = 0
        self._carried_tail_trajectories = 0
        self._dropped_tail_trajectories = 0
        self._dropped_tail_roots = 0
        self._discarded_completed_trajectories = 0
        self._gt_by_root: Dict[str, Optional[str]] = {}
        self._n = int(samples_per_prompt)
        self._oversample = int(oversample_batch_size) if oversample_batch_size else self.batch_size
        if self._oversample < self.batch_size:
            raise ValueError(f"oversample_batch_size={self._oversample} must be >= batch_size={self.batch_size}")

        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        self._rollout_devices = self.num_devices - self._train_devices

        with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
        with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                raise ValueError(
                    "AsyncAgenticTrainer needs a dedicated agentic-rollout engine on the "
                    "separate slab (its inner sglang/vllm engine lives cross-slab)."
                )
            self.rollout = remote(**rollout_parsed)

        self.rollout.set_workers(self.rollout.workers, self.rollout.role_name)

        if self.weight_sync is not None:
            self._connect_separate(sync_cfg)

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab NCCL handshake (identical to AsyncARTrainer)."""
        target = str(sync_cfg.get("_target_", ""))
        if not target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"AsyncAgenticTrainer (separate slabs) requires a cross-slab weight sync "
                f"(NCCLWeightSync); got sync._target_={target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(self.rollout.workers),
        )

    def _build_tasks(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """A drive's task list: ``n`` fresh siblings per new prompt + carried partials.

        Fresh prompts (``oversample_batch_size`` of them) are split into per-prompt
        request trees and replicated ``n`` times (the GRPO siblings share a root id);
        carried partials (their root ids + turn history preserved) are appended as-is,
        to be resumed by the engine's ``_run_one`` from ``len(gen_parts())``.

        Fresh roots are namespaced by ``_drive_seq`` as well as ``rollout_id`` so a
        refill can never reuse a drained drive's root ids for different source rows.
        Carried partials keep the ids they were submitted under, so they still rejoin
        their own siblings in the assembler.
        """
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

    def _submit_drive(self, carried: List[Sample], rollout_id: int) -> None:
        """Submit a fresh over-sampled drive (non-blocking). Call only when the prior
        drive is finalized/quiesced (else two drains would double-pull)."""
        self._engine.submit(self._build_tasks(carried, rollout_id))

    def _apply_tail_policy(self, carried: List[Sample], rollout_id: int) -> List[Sample]:
        """Keep resumable tails or purge all state belonging to dropped roots."""
        if self._tail_policy == "carry":
            self._carried_tail_trajectories += len(carried)
            logger.info("rollout %d async: carry tail=%d trajectories", rollout_id, len(carried))
            return carried

        roots = {root_of(sample) for sample in carried}
        discarded_completed = self._engine.discard_roots(roots)
        for root in roots:
            self._gt_by_root.pop(root, None)
        self._dropped_tail_trajectories += len(carried)
        self._dropped_tail_roots += len(roots)
        self._discarded_completed_trajectories += discarded_completed
        logger.info(
            "rollout %d async: drop tail=%d trajectories across %d roots; discarded %d completed siblings",
            rollout_id,
            len(carried),
            len(roots),
            discarded_completed,
        )
        return []

    def _log_tail_metrics(self, rollout_step: int) -> None:
        """Emit tail counters after abort/policy, including the final step."""
        self.wandb_logger.log_rollout(
            rollout_step,
            {
                "async/carried_tail_trajectories": self._carried_tail_trajectories,
                "async/dropped_tail_trajectories": self._dropped_tail_trajectories,
                "async/dropped_tail_roots": self._dropped_tail_roots,
                "async/discarded_completed_trajectories": self._discarded_completed_trajectories,
                "async/assembler_pending_roots": self._engine.pending_groups(),
            },
        )

    def _drain_buffer(self, n: int, *, max_staleness: int) -> Optional[List[List[Sample]]]:
        """Drain fresh groups and forget ground truth for stale evictions."""
        picked = self._engine.drain_freshest(n, max_staleness=max_staleness)
        for group in self._engine.pop_evicted():
            if group:
                self._gt_by_root.pop(root_of(group[0]), None)
        return picked

    def _next_batch(self, rollout_id: int) -> List[List[Sample]]:
        """Pump the producer until the buffer holds ``batch_size`` complete groups within
        the staleness bound, then drain the freshest ones. If the in-flight drive drains
        without filling the buffer (staleness eviction / failures / small over-sample),
        refill with a fresh drive."""
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        refills = 0
        while True:
            self._engine.poll()
            picked = self._drain_buffer(self.batch_size, max_staleness=stale)
            if picked is not None:
                return picked
            if self._engine.finalize_if_drained() is None:
                time.sleep(self._POLL_INTERVAL_S)
                continue
            picked = self._drain_buffer(self.batch_size, max_staleness=stale)
            if picked is not None:
                return picked

            refills += 1
            if refills > self._MAX_REFILLS:
                raise RuntimeError(
                    f"async-agentic rollout {rollout_id}: buffer underflow after {refills} refills "
                    f"(buffer={self._engine.buffered_groups()} < batch={self.batch_size}); raise "
                    f"oversample_batch_size or buffer_max_staleness."
                )
            self._submit_drive([], rollout_id)

    def _train_on_groups(
        self, groups: List[List[Sample]], *, training_progress: float, rollout_id: int, t0: float
    ) -> Tuple[TrainStepResult, float]:
        """Reward + GROUP-relative advantage + one step over ``batch_size`` complete
        groups. Reuses :class:`AgenticTrainer`'s reward/GRPO/log helpers; the train-part
        assembly mirrors :meth:`AgenticTrainer.train_step` (steps 5-6)."""
        trajs: List[Sample] = [t for group in groups for t in group]
        roots = [tr.parts[0].sample_ids[0] for tr in trajs]
        request = Sample.request(Part.input(roots, metadata=[{"answer": self._gt_by_root.get(r)} for r in roots]))
        rewards, group_ids = self._rewards_and_groups(request, trajs, rollout_id)
        for root in set(roots):
            self._gt_by_root.pop(root, None)
        finite = torch.isfinite(rewards)
        mean_reward = float(rewards[finite].mean().item()) if bool(finite.any()) else 0.0
        advantages = self._group_advantages(rewards, group_ids)

        train_parts: List[Part] = []
        for i, tr in enumerate(trajs):
            adv_i = float(advantages[i].item())
            for gp in tr.gen_parts():
                gp = _part_with_field(gp, "advantages", torch.full((gp.batch_size,), adv_i, dtype=torch.float32))
                gp = _part_with_field(gp, "primitives", {})
                gp = _part_with_field(gp, "rewards", None)
                train_parts.append(gp)

        depths = [len(tr.gen_parts()) for tr in trajs]
        if not train_parts:
            logger.warning("AsyncAgenticTrainer rollout %d produced no trainable turns.", rollout_id)
            return TrainStepResult(0.0, 0.0, 0.0, False, [], {}), mean_reward

        train_part = self._pad_to_dp_multiple(Part.concat(train_parts))
        result = self.stack.train_track(train_part, training_progress=float(training_progress))

        log_sample = self._build_log_sample(trajs, rewards, advantages, rollout_id)
        versions = [gp.weight_version for tr in trajs for gp in tr.gen_parts() if gp.weight_version is not None]
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            log_sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={
                "agent/mean_turns": (sum(depths) / len(depths)) if depths else 0.0,
                "agent/max_turns": max(depths) if depths else 0,
                "async/buffer_groups": self._engine.buffered_groups(),
                "async/weight_version": self._engine.weight_version,
                "async/version_span": (max(versions) - min(versions)) if versions else 0,
                "async/assembler_pending_roots": self._engine.pending_groups(),
                "async/carried_tail_trajectories": self._carried_tail_trajectories,
                "async/dropped_tail_trajectories": self._dropped_tail_trajectories,
                "async/dropped_tail_roots": self._dropped_tail_roots,
                "async/discarded_completed_trajectories": self._discarded_completed_trajectories,
            },
        )
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
        save_mode: str = "full",
    ) -> None:
        interval = max(1, weight_sync_interval)
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        for _ in range(start_rollout):
            self.data_source.get_samples(self._oversample)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "buffer_max_staleness": stale,
                "oversample_batch_size": self._oversample,
                "train_fraction": self._train_fraction,
                "tail_policy": self._tail_policy,
                "weight_sync_interval": interval,
            },
        )

        self._engine = AgenticManager(self.rollout, group_size=self._n, start_gen_id=start_rollout)

        if start_rollout < num_rollouts and start_rollout and self.weight_sync is not None:
            self._engine.sync_weights(self.weight_sync)  # push restored weights into the fresh engine
        if start_rollout < num_rollouts:
            self._submit_drive(carried=[], rollout_id=start_rollout)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                groups = self._next_batch(rollout_id)
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._train_on_groups(
                    groups, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                need_save = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                need_sync = step % interval == 0 and self.weight_sync is not None
                if need_save or need_sync:
                    checkpointed = self._engine.quiesce()
                    carried = self._apply_tail_policy(checkpointed, rollout_id)
                    if checkpointed:
                        self._log_tail_metrics(step)
                    if need_save:
                        self.maybe_save_checkpoint(
                            rollout_id,
                            num_rollouts,
                            save_interval=save_interval,
                            save_dir=save_dir,
                            save_mode=save_mode,
                        )
                    if need_sync:
                        self._engine.sync_weights(self.weight_sync)
                    if step < num_rollouts:
                        self._submit_drive(carried=carried, rollout_id=step)
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                checkpointed = self._engine.quiesce()
                self._apply_tail_policy(checkpointed, num_rollouts)
                if checkpointed:
                    self._log_tail_metrics(num_rollouts)
            except BaseException:  # noqa: BLE001 — preserve an active training failure
                if active_error:
                    logger.warning("AsyncAgenticTrainer cleanup failed", exc_info=True)
                else:
                    raise
            finally:
                self._finish_wandb()
