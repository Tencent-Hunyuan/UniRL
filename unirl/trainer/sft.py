"""SFTTrainer — driver orchestrator for supervised finetuning / behavior cloning.

One role: an :class:`~unirl.train.sft.policy.SFTPolicy` (config-chosen task
adapter + FSDP + optimizer) on the whole device pool. Each step::

    records = data_source.get_samples(batch_size)     # driver-side manifest rows
    policy.train_batch(records, step, dp_size)        # worker-side load->loss->backward
    policy.optimizer_step(max_grad_norm); policy.zero_grad()

No rollout engine / reward / advantages / weight sync — mirrors the ReFL
domain's shape (``unirl/trainer/refl.py``), the codified template for
non-GRPO domains. Success signal: the flow-matching loss falls and periodic
samples improve.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.distributed.tensor.ref import hydrate, map_tree
from unirl.trainer.base import BaseTrainer
from unirl.utils.hydra import remote_hydra

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    """Supervised trainer: manifest records -> task loss -> optimizer step."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        policy_cfg: DictConfig,
        data_source_cfg: DictConfig,
        max_grad_norm: float = 1.0,
        eval_interval: int = 0,
        eval_num_samples: int = 1,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self.eval_interval = int(eval_interval)
        self.eval_num_samples = int(eval_num_samples)
        self.data_source = instantiate(data_source_cfg)

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.policy = remote_hydra(policy_cfg)
        self.policy.initialize()
        # BaseTrainer.maybe_save/load_checkpoint operate on ``self.backend``.
        self.backend = self.policy

        self.dp_size = int(self.policy.dp_size)
        if self.batch_size % self.dp_size:
            raise ValueError(f"batch_size={self.batch_size} must be divisible by policy dp={self.dp_size}")
        logger.info("SFTTrainer ready: dp=%d batch=%d max_grad_norm=%.2f", self.dp_size, self.batch_size, self.max_grad_norm)

    def train_step(self, records: List[Dict[str, Any]], *, step: int) -> Tuple[Dict[str, float], float, float]:
        t0 = time.perf_counter()
        per_worker = self.policy.train_batch(records=records, step=step, dp_size=self.dp_size)
        grad_norm = self.policy.optimizer_step(max_grad_norm=self.max_grad_norm)
        if isinstance(grad_norm, list):  # BROADCAST -> one result per worker
            grad_norm = grad_norm[0]
        self.policy.zero_grad()

        metrics: Dict[str, float] = {}
        worker_metrics = per_worker if isinstance(per_worker, list) else [per_worker]
        for worker in worker_metrics:
            for key, value in worker.items():
                metrics[key] = metrics.get(key, 0.0) + float(value) / len(worker_metrics)
        return metrics, float(grad_norm or 0.0), time.perf_counter() - t0

    def _run_eval(self, step: int, save_dir: Optional[str]) -> None:
        records = self.data_source.eval_samples(self.eval_num_samples)
        outputs = self.policy.sample_media(records=records, step=step)
        if isinstance(outputs, list):  # one entry per worker; dp rank 0 carries media
            outputs = next((o for o in outputs if o is not None), None)
        if outputs is None:
            return
        outputs = map_tree(outputs, hydrate)  # materialize worker-side TensorRef proxies
        out_dir = os.path.join(save_dir or ".", "samples")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"step_{step}.pt")
        torch.save(outputs, path)
        logger.info("eval samples @ step %d -> %s", step, path)

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        start = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for step in range(start, num_rollouts):
                records = self.data_source.get_samples(self.batch_size)
                metrics, grad_norm, dt = self.train_step(records, step=step)
                logger.info(
                    "step %d/%d  loss=%.5f grad_norm=%.4f  %.1fs",
                    step + 1,
                    num_rollouts,
                    metrics.get("loss/total", float("nan")),
                    grad_norm,
                    dt,
                )
                self.wandb_logger.log_step(
                    step + 1,
                    {"train/loss": metrics.get("loss/total", 0.0), "train/grad_norm": grad_norm, "perf/step_time_s": dt, **metrics},
                    prefix="",
                )
                if self.eval_interval and (step + 1) % self.eval_interval == 0:
                    self._run_eval(step + 1, save_dir)
                self.maybe_save_checkpoint(
                    step,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
        finally:
            self._finish_wandb()


__all__ = ["SFTTrainer"]
