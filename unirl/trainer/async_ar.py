"""Async autoregressive RL over separate train and rollout GPU slabs."""

import inspect
import time
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer, ar_preflight
from unirl.trainer.async_rollout import AsyncRolloutTrainerMixin, training_version_metrics
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra


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


class AsyncARTrainer(AsyncRolloutTrainerMixin, ARTrainer):
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
        self._allowed_input_primitives = ar_preflight(
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
        self._weight_sync_interval = int(weight_sync_interval)
        self._num_updates_per_batch = int(stack_cfg.get("num_updates_per_batch", 1))
        if self._weight_sync_interval < 1:
            raise ValueError(f"weight_sync_interval must be >= 1, got {self._weight_sync_interval}")
        if self._num_updates_per_batch < 1:
            raise ValueError(f"num_updates_per_batch must be >= 1, got {self._num_updates_per_batch}")
        self._train_version = 0
        self._batches_since_sync = 0
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
            self._sync_rollout(require_empty=True)
        return False

    def _finish_rollout(self, *, train_state_offloaded: bool) -> None:
        """Keep the separate rollout slab resident across train/eval phases."""
        if train_state_offloaded:
            raise RuntimeError("AsyncARTrainer cannot offload its disjoint training slab during rollout")

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake (NCCL branch of diffusion.py:191-208)."""
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
        self._train_version += result.optimizer_updates
        self._batches_since_sync += 1
        if extra_metrics is not None:
            extra_metrics.update(
                training_version_metrics(
                    train_version=self._train_version,
                    published_version=self._rollout_manager.published_version,
                    optimizer_updates=result.optimizer_updates,
                    batches_since_sync=self._batches_since_sync,
                )
            )
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
        self._train_async_loop(
            num_rollouts=num_rollouts,
            save_interval=save_interval,
            save_dir=save_dir,
            load_dir=load_dir,
            save_mode=save_mode,
        )

    def _async_wandb_extra(self) -> Dict[str, object]:
        return {"adv_normalization_scope": self.adv_normalization_scope}

    def _refill_before_score(self) -> bool:
        """Overlap AR generation with reward scoring and training."""
        return True

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        self.evaluate(rollout_id=-1 if initial else rollout_id)
