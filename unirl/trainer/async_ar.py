"""Async autoregressive RL over separate train and rollout GPU slabs.

The trainer owns optimizer-version admission and publication policy through
``AsyncBatchControl``. The driver-side ``RolloutManager`` owns bounded dispatch,
completion-order grouping, filtering, and quiescence.
"""

import inspect
import logging
import time
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.models.qwen3_5.validation import validate_qwen3_5_training_contract
from unirl.rollout.manager import RolloutManager, keep_within_lag
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer.async_batch_control import (
    AsyncBatchControl,
    log_admission_notes,
    max_publication_gap_batches,
    next_hard_boundary,
    unwrap_replicated_int,
)
from unirl.trainer.async_rollout import combine_rollout_chunks
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


def _rollout_dp_size_from_parsed_config(rollout_parsed: dict, *, world_size: int) -> int:
    """Compute the rollout Handle's DP width before constructing GPU roles."""
    from unirl.distributed.group.handle import _parallel_shape_from_init_kwargs

    role_cls = rollout_parsed["role_cls"]
    init_kwargs = {key: value for key, value in rollout_parsed.items() if key != "role_cls"}
    sp_size, tp_size, pp_size, _ = _parallel_shape_from_init_kwargs(
        init_kwargs,
        world_size,
        role_cls,
    )
    non_dp_width = sp_size if sp_size > 1 else tp_size * pp_size
    return world_size // non_dp_width


class AsyncARTrainer(ARTrainer):
    """Disaggregated async AR trainer (two slabs, resident engine, NCCL sync)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
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
        advantage_mode: str = "grpo",
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        train_fraction: float = 0.5,
        max_inflight: int = 1,
        weight_sync_interval: int = 1,
    ) -> None:
        validate_qwen3_5_training_contract(
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            rollout_cfg=rollout_cfg,
            stack_cfg=stack_cfg,
        )
        # Skips ARTrainer.__init__: it opens the colocate placement block we replace with two slabs.
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.advantage_mode = str(advantage_mode).strip().lower()
        if self.advantage_mode not in ("grpo", "gae"):
            raise ValueError(f"AsyncARTrainer: advantage_mode must be 'grpo' or 'gae', got {advantage_mode!r}")
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None
        rollout_parsed = parse_hydra_cfg(rollout_cfg)
        if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
            raise ValueError(
                "AsyncARTrainer needs a dedicated-rollout engine (vllm/sglang) on the "
                "separate slab; the trainside direct-sampling engine needs the pipeline "
                "as a local sibling and cannot live cross-slab."
            )
        self._rollout_anchor_device = None

        self._train_fraction = float(train_fraction)
        self._max_inflight = max(1, int(max_inflight))
        self._control = AsyncBatchControl(
            weight_sync_interval=weight_sync_interval,
            num_updates_per_batch=stack_cfg.get("num_updates_per_batch", 1),
        )
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        # Require rollout batches to divide evenly across train and rollout slabs.
        self._rollout_devices = self.num_devices - self._train_devices
        prompts = self.batch_size
        total = prompts * total_samples_per_prompt(self.sampling_params)
        if total % self._train_devices != 0:
            raise ValueError(
                f"batch_size * samples_per_prompt = {total} is not divisible by the train "
                f"slab size {self._train_devices}; adjust batch_size / samples_per_prompt / train_fraction."
            )
        rollout_dp_size = _rollout_dp_size_from_parsed_config(
            rollout_parsed,
            world_size=self._rollout_devices,
        )
        if prompts % rollout_dp_size != 0:
            raise ValueError(
                f"batch_size = {prompts} prompts is not divisible by the rollout DP size "
                f"{rollout_dp_size} ({self._rollout_devices} rollout GPUs; each prompt-tree "
                "DP-scatters whole); adjust batch_size / train_fraction / rollout TP."
            )

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
            self.rollout = remote(**rollout_parsed)

        if self.weight_sync is None:
            raise ValueError("AsyncARTrainer requires a cross-slab weight sync; add a `sync:` block.")
        self._connect_separate(sync_cfg)

    def _prepare_rollout(self, *, sync_weights: bool) -> bool:
        """Sync a resident separate-slab engine without colocate handoffs."""
        if sync_weights:
            self._control.sync_rollout(self._rollout_manager, self.weight_sync)
        return False

    def _finish_rollout(self, *, train_state_offloaded: bool) -> None:
        """Keep the separate rollout slab resident across train/eval phases."""
        if train_state_offloaded:
            raise RuntimeError("AsyncARTrainer cannot offload its disjoint training slab during rollout")

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake (NCCL branch of diffusion.py:191-208).

        Rank 0 picks a rendezvous addr/port, is handed the rollout slab's Worker
        actor handles, then ``connect`` fires each rollout worker's
        ``init_weights_update_group`` non-blocking and joins the broadcast group
        itself. Only ``NCCLWeightSync`` is supported here (always cross-slab
        full-weight); a non-NCCL target is a config error.
        """
        target = str(sync_cfg.get("_target_", ""))
        if not target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"AsyncARTrainer (separate slabs) requires a cross-slab weight sync "
                f"(NCCLWeightSync); got sync._target_={target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        tp_size = self.rollout.tp_size
        pp_size = self.rollout.pp_size
        targets = self.rollout.tp_zero_workers
        self.weight_sync.set_rollout_targets(targets, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(targets) * tp_size,
            tp_size=tp_size,
            pp_size=pp_size,
        )

    def _build_async_sample(self, gen_id: int) -> Sample:
        """Consume one data batch and build the request Sample for ``gen_id``."""
        return self._build_request_sample(self.data_source.get_samples(self.batch_size), gen_id)

    def _score_completed(self, gen_id: int, completed: Sample) -> Sample:
        scored = self.reward.score_and_attach(completed)
        self._drop_decoded(scored, rollout_id=gen_id)
        return scored

    def _advantage_and_train(
        self,
        sample: Sample,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer updates for a scored ``Sample`` (rewards already attached)."""
        if t0 is None:
            t0 = time.perf_counter()
        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            part.rewards = hydrate(part.rewards)
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
        if self.advantage_mode == "grpo":
            part = part.compute_advantages(
                normalize=self.normalize_adv_by_std,
                scope=self.adv_normalization_scope,
            )
        sample = sample.with_parts([*sample.parts[:-1], part])
        train_part = part
        if self.balance_shards:
            train_part = part.balance_shards(self._train_devices)
        result = self.stack.train_track(train_part, training_progress=float(training_progress))
        self._control.record_optimizer_updates(result.optimizer_updates)
        if extra_metrics is not None:
            extra_metrics.update(self._control.train_metrics(result.optimizer_updates))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            sample,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
            extra_metrics=extra_metrics,
        )
        self._reset_transport_buffers()
        return result, mean_reward

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        train_version = unwrap_replicated_int(
            self.backend.get_optimizer_step_count(),
            name="backend optimizer step count",
        )
        self._control.restore(train_version)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "max_inflight": self._max_inflight,
                "weight_sync_interval": self._control.weight_sync_interval,
                "max_staleness": self._control.max_staleness,
                "staleness_budget": self._control.staleness_budget,
                "num_updates_per_batch": self._control.num_updates_per_batch,
                "max_publication_gap_batches": max_publication_gap_batches(
                    self._control,
                    eval_interval=self.eval_interval,
                    save_interval=save_interval,
                ),
            },
        )
        # Not in __init__: save_interval arrives with train() and clamps the publication interval.
        log_admission_notes(
            self._control,
            max_inflight=self._max_inflight,
            eval_interval=self.eval_interval,
            save_interval=save_interval,
        )

        self._rollout_manager = RolloutManager(
            self.rollout,
            launchers=[lambda sample: self.rollout.launch_nowait("generate", sample)],
            capacities=[self._max_inflight],
            group_size=total_samples_per_prompt(self.sampling_params),
            filter_fn=keep_within_lag(self._control.staleness_budget),
        )
        self._next_generation_id = start_rollout

        if resumed or self.eval_interval > 0:
            self._control.sync_rollout(self._rollout_manager, self.weight_sync, force=True)
        if self.eval_interval > 0:
            self.evaluate(rollout_id=-1)

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
                    extra_metrics=self._control.output_metrics(output_version),
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                eval_due = self.eval_interval > 0 and step % self.eval_interval == 0
                save_due = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                sync_due = step < num_rollouts and self._control.publication_due
                if eval_due or save_due or sync_due:
                    self._control.sync_rollout(self._rollout_manager, self.weight_sync)

                if eval_due:
                    self.evaluate(rollout_id=rollout_id)
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

    def _next_rollout_batch(
        self,
        rollout_id: int,
        *,
        num_rollouts: int,
        hard_boundary: int,
    ) -> Tuple[Sample, int]:
        manager = self._rollout_manager
        inflight_count, ready_count = manager.counts
        if inflight_count + ready_count == 0:
            slots = self._control.launch_slots(
                inflight_count=0,
                ready_count=0,
                max_inflight=self._max_inflight,
                trained_batches=rollout_id,
                num_rollouts=num_rollouts,
                hard_boundary=hard_boundary,
            )
            self._submit_generations(slots)

        groups = manager.collect(self.batch_size, current_version=self._control.train_version)
        completed, gen_id, output_version = combine_rollout_chunks(groups)
        scored = self._score_completed(gen_id, completed)

        inflight_count, ready_count = manager.counts
        slots = self._control.launch_slots(
            inflight_count=inflight_count,
            ready_count=ready_count + 1,
            max_inflight=self._max_inflight,
            trained_batches=rollout_id,
            num_rollouts=num_rollouts,
            hard_boundary=hard_boundary,
        )
        self._submit_generations(slots)
        return scored, output_version

    def _submit_generations(self, count: int) -> None:
        for _ in range(count):
            gen_id = self._next_generation_id
            self._rollout_manager.submit([self._build_async_sample(gen_id)])
            self._next_generation_id += 1
