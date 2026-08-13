"""Async diffusion RL over separate train and rollout GPU slabs."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.async_rollout import AsyncRolloutTrainerMixin, training_version_metrics
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.types.sample import Sample


class AsyncDiffusionTrainer(AsyncRolloutTrainerMixin, DiffusionTrainer):
    """Disaggregated async diffusion trainer (two slabs, resident engine, cross-slab sync)."""

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        weight_sync_interval: int = 1,
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
        if bool(diffusion_kwargs.get("offload_train_during_reward", False)):
            raise ValueError(
                "AsyncDiffusionTrainer does not support offload_train_during_reward: async scoring "
                "runs at reap time outside _reward_phase(), so the option would be silently ignored "
                "and a reward sharing the train slab could still OOM. Remove the option or use the "
                "synchronous trainer."
            )
        super().__init__(**diffusion_kwargs)

        if self.weight_sync is None:
            raise ValueError(
                "AsyncDiffusionTrainer requires a cross-slab weight sync; add a `sync:` block to the recipe."
            )

        self._max_inflight = max_inflight
        self._weight_sync_interval = int(weight_sync_interval)
        self._num_updates_per_batch = int(diffusion_kwargs["stack_cfg"].get("num_updates_per_batch", 1))
        if self._weight_sync_interval < 1:
            raise ValueError(f"weight_sync_interval must be >= 1, got {self._weight_sync_interval}")
        if self._num_updates_per_batch < 1:
            raise ValueError(f"num_updates_per_batch must be >= 1, got {self._num_updates_per_batch}")
        self._train_version = 0
        self._batches_since_sync = 0

    def _advantage_and_train(
        self,
        sample: Sample,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
        extra_metrics: Optional[dict[str, float]] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer updates for a scored ``Sample`` (rewards already attached)."""
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
        save_mode: str = "auto",
    ) -> None:
        self._train_async_loop(
            num_rollouts=num_rollouts,
            save_interval=save_interval,
            save_dir=save_dir,
            load_dir=load_dir,
            save_mode=save_mode,
        )

    def _async_wandb_extra(self) -> Dict[str, object]:
        return {"train_fraction": self._train_fraction}

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        self.evaluate(rollout_id if initial else rollout_id + 1, sync_weights=False, sleep_after=False)
