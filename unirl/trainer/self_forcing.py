"""Dedicated WAN Self-Forcing DMD trainer."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.trainer.base import BaseTrainer
from unirl.utils.hydra import remote_hydra
from unirl.utils.misc import aggregate_numeric_metrics

logger = logging.getLogger(__name__)

_TRAINER_STATE = "self_forcing_trainer_state.json"
_DATA_STATE = "self_forcing_data_state.json"
_COMPLETE = "_COMPLETE"


class WAN21SelfForcingTrainer(BaseTrainer):
    """Prompt batches → one generator DMD update + N fake-score updates."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        generator_bundle_cfg: DictConfig,
        generator_pipeline_cfg: DictConfig,
        generator_backend_cfg: DictConfig,
        fake_score_bundle_cfg: DictConfig,
        fake_score_pipeline_cfg: DictConfig,
        fake_score_backend_cfg: DictConfig,
        real_score_bundle_cfg: DictConfig,
        real_score_pipeline_cfg: DictConfig,
        real_score_backend_cfg: DictConfig,
        stack_cfg: DictConfig,
        track_builder_cfg: DictConfig,
        data_source_cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = int(batch_size)
        self.fake_score_updates_per_generator = int(stack_cfg.get("fake_score_updates_per_generator", 5))
        self.data_source = instantiate(data_source_cfg)

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.generator_bundle = remote_hydra(generator_bundle_cfg)
            self.generator_pipeline = remote_hydra(
                generator_pipeline_cfg,
                bundle=self.generator_bundle,
            )
            self.generator_backend = remote_hydra(
                generator_backend_cfg,
                bundle=self.generator_bundle,
            )

            self.fake_score_bundle = remote_hydra(fake_score_bundle_cfg)
            self.fake_score_pipeline = remote_hydra(
                fake_score_pipeline_cfg,
                bundle=self.fake_score_bundle,
            )
            self.fake_score_backend = remote_hydra(
                fake_score_backend_cfg,
                bundle=self.fake_score_bundle,
            )

            self.real_score_bundle = remote_hydra(real_score_bundle_cfg)
            self.real_score_pipeline = remote_hydra(
                real_score_pipeline_cfg,
                bundle=self.real_score_bundle,
            )
            self.real_score_backend = remote_hydra(
                real_score_backend_cfg,
                bundle=self.real_score_bundle,
            )

            self.stack = remote_hydra(
                stack_cfg,
                generator_pipeline=self.generator_pipeline,
                generator_backend=self.generator_backend,
                fake_score_pipeline=self.fake_score_pipeline,
                fake_score_backend=self.fake_score_backend,
                real_score_pipeline=self.real_score_pipeline,
                real_score_backend=self.real_score_backend,
            )
            self.track_builder = remote_hydra(
                track_builder_cfg,
                pipeline=self.generator_pipeline,
            )

        self.backend = self.generator_backend  # BaseTrainer memory-monitor compatibility.
        self.dp_size = self.stack.dp_size
        if self.batch_size % self.dp_size:
            raise ValueError(
                f"WAN21SelfForcingTrainer: batch_size={self.batch_size} must be divisible by dp={self.dp_size}."
            )

    def train_step(self, records: List[Dict[str, Any]], *, training_progress: float = 0.0) -> Dict[str, Any]:
        part = self.track_builder.build(records)
        if part.batch_size != len(records):
            raise RuntimeError(f"WAN21SelfForcingTrainer: built {part.batch_size} rows from {len(records)} records.")
        rows = self.stack.train_track(part, training_progress=training_progress)
        if not rows:
            raise RuntimeError("WAN21SelfForcingTrainer: stack returned no result rows.")
        summary = aggregate_numeric_metrics(
            [{key: value for key, value in row.items() if key != "metrics"} for row in rows]
        )
        summary["fake_score_updates"] = int(round(summary["fake_score_updates"]))
        summary["metrics"] = aggregate_numeric_metrics([dict(row.get("metrics") or {}) for row in rows])
        return summary

    # ------------------------------------------------------------------
    # Paired checkpoint lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def _checkpoint_base(save_dir: Optional[str]) -> str:
        return os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints-self-forcing")

    def _save_checkpoint(
        self,
        step: int,
        num_steps: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str,
    ) -> None:
        if save_interval <= 0 or (step % save_interval != 0 and step < num_steps):
            return
        path = os.path.join(self._checkpoint_base(save_dir), f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        complete = os.path.join(path, _COMPLETE)
        if os.path.exists(complete):
            os.remove(complete)
        logger.info("Saving paired Self-Forcing checkpoint at step %d -> %s", step, path)
        self.generator_backend.save(
            os.path.join(path, "generator"),
            step=step,
            mode=save_mode,
        )
        self.generator_backend.wait_for_checkpoint()
        self.fake_score_backend.save(
            os.path.join(path, "fake_score"),
            step=step,
            mode=save_mode,
        )
        self.fake_score_backend.wait_for_checkpoint()

        state = {
            "outer_step": step,
            "wandb_run_id": self.wandb_logger.run_id,
            "optimizer_step": self.wandb_logger.optimizer_step,
        }
        self._atomic_json(os.path.join(path, _TRAINER_STATE), state)
        self._atomic_json(os.path.join(path, _DATA_STATE), self.data_source.state_dict())
        tmp = f"{complete}.tmp.{os.getpid()}"
        with open(tmp, "w") as handle:
            handle.write("ok\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, complete)

    @staticmethod
    def _atomic_json(path: str, payload: Dict[str, Any]) -> None:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _load_checkpoint(self, load_dir: Optional[str]) -> int:
        if not load_dir:
            return 0
        path = os.path.abspath(load_dir)
        if not os.path.exists(os.path.join(path, _COMPLETE)):
            raise RuntimeError(f"Incomplete Self-Forcing checkpoint (missing {_COMPLETE}): {path}")
        gen_step = self.generator_backend.load(os.path.join(path, "generator"))
        fake_step = self.fake_score_backend.load(os.path.join(path, "fake_score"))
        if isinstance(gen_step, list):
            gen_step = gen_step[0]
        if isinstance(fake_step, list):
            fake_step = fake_step[0]
        if int(gen_step or 0) != int(fake_step or 0):
            raise RuntimeError(
                f"Paired Self-Forcing checkpoint step mismatch: generator={gen_step}, fake_score={fake_step}."
            )
        state_path = os.path.join(path, _TRAINER_STATE)
        if os.path.exists(state_path):
            with open(state_path) as handle:
                state = json.load(handle)
            self._resume_state = {
                "wandb_run_id": state.get("wandb_run_id"),
                "optimizer_step": state.get("optimizer_step", 0),
            }
        data_path = os.path.join(path, _DATA_STATE)
        if os.path.exists(data_path):
            with open(data_path) as handle:
                self.data_source.load_state_dict(json.load(handle))
        return int(gen_step or 0)

    def _wait_for_checkpoints(self, *, timeout: Optional[float] = None) -> None:
        for backend in (self.generator_backend, self.fake_score_backend):
            if timeout is None:
                backend.wait_for_checkpoint()
            else:
                backend.wait_for_checkpoint(_ray_get_timeout=timeout)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        num_steps: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        start_step = self._load_checkpoint(load_dir)
        self._init_wandb(
            num_rollouts=num_steps,
            extra={
                "trainer": "wan21_self_forcing_dmd",
                "fake_score_updates_per_generator": self.fake_score_updates_per_generator,
            },
        )
        try:
            for step in range(start_step, num_steps):
                t0 = time.perf_counter()
                records = self.data_source.get_samples(self.batch_size)
                result = self.train_step(
                    records,
                    training_progress=step / max(1, num_steps - 1),
                )
                dt = time.perf_counter() - t0
                logger.info(
                    "step %d/%d  gen=%.6f gnorm=%.4f  fake=%.6f fnorm=%.4f  %.1fs",
                    step + 1,
                    num_steps,
                    float(result["generator_loss"]),
                    float(result["generator_grad_norm"]),
                    float(result["fake_score_loss"]),
                    float(result["fake_score_grad_norm"]),
                    dt,
                )
                metrics = dict(result.get("metrics") or {})
                self.wandb_logger.log_step(
                    step + 1,
                    {
                        "train/generator_loss": float(result["generator_loss"]),
                        "train/generator_grad_norm": float(result["generator_grad_norm"]),
                        "train/generator_lr": float(result["generator_lr"]),
                        "train/fake_score_loss": float(result["fake_score_loss"]),
                        "train/fake_score_grad_norm": float(result["fake_score_grad_norm"]),
                        "train/fake_score_lr": float(result["fake_score_lr"]),
                        "train/fake_score_updates": int(result["fake_score_updates"]),
                        "train/epoch": self.data_source.epoch,
                        "perf/step_time_s": dt,
                        **{f"train/{key}": value for key, value in metrics.items()},
                    },
                    prefix="",
                )
                self.wandb_logger.set_optimizer_step(step + 1)
                self._save_checkpoint(
                    step + 1,
                    num_steps,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
        finally:
            self._finish_wandb()


__all__ = ["WAN21SelfForcingTrainer"]
