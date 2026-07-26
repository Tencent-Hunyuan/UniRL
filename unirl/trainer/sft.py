"""SFTTrainer — driver orchestrator for supervised finetuning.

The supervised sibling of :class:`~unirl.trainer.ar.ARTrainer` /
:class:`~unirl.trainer.diffusion.DiffusionTrainer`: same consumer side
(``bundle → pipeline → backend → algorithm → stack`` siblings on one
placement), but the data producer is a dataset-backed ``SupervisedTrackBuilder``
instead of a rollout engine — no reward service, no advantages, no weight
sync, no sampling params. Each step::

    records = data_source.get_samples(batch_size)       # driver-side rows
    track   = track_builder.build(records)              # worker-side encode → RolloutTrack
    result  = stack.train_track(track, ...)             # the SAME stack RL uses

The algorithm (``unirl.algorithms.SFT`` / ``FlowMatchSFT``) declares
``requires_advantages=False``; everything else about the stack — micro
planning, grad accumulation, the token-weighted global loss normalization,
EMA, checkpointing through the backend — is shared with the RL trainers, so
SFT inherits every stack/backend improvement for free (and doubles as the
cheapest end-to-end regression exercise of that machinery).

Supervised-only concerns owned here: epoch semantics with an exact
``{epoch, position}`` resume cursor (saved beside each checkpoint), and
full-validation-set eval loss through ``stack.eval_track`` — the final partial
eval batch is padded to the DP width with ``_eval_pad`` rows the loss masks
out, so no tail sample is dropped and no padded row is counted.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer
from unirl.utils.hydra import remote_hydra

logger = logging.getLogger(__name__)

_DATA_STATE_FILENAME = "sft_data_state.json"


class SFTTrainer(BaseTrainer):
    """Supervised trainer: dataset records → stage loss → optimizer step."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        track_builder_cfg: DictConfig,
        data_source_cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
        eval_interval: int = 0,
        eval_batch_size: int = 8,
        eval_num_samples: int = -1,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self.eval_interval = eval_interval
        self.eval_batch_size = max(1, eval_batch_size)
        self.eval_num_samples = -1 if eval_num_samples < 0 else eval_num_samples

        # Driver-side data iterator (not a Remote) — records stay light dicts;
        # tokenization / media loading run worker-side in the track builder.
        self.data_source = instantiate(data_source_cfg)

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
            self.track_builder = remote_hydra(track_builder_cfg, pipeline=self.pipeline)

        self.dp_size = self.stack.dp_size
        if self.batch_size % self.dp_size:
            raise ValueError(f"SFTTrainer: batch_size={self.batch_size} must be divisible by dp={self.dp_size}")
        logger.info("SFTTrainer ready: dp=%d batch=%d", self.dp_size, self.batch_size)

    # ------------------------------------------------------------------
    # One optimizer step
    # ------------------------------------------------------------------

    def train_step(self, records: List[Dict[str, Any]], *, training_progress: float = 0.0) -> TrainStepResult:
        """records → worker-side track build → stack train. No rollout legs."""
        track = self.track_builder.build(records)
        if track.batch_size != len(records):
            # AReaL's single-controller once broadcast SFT batches instead of
            # scattering them — 8× duplicated tokens with a correct-LOOKING loss.
            # Token conservation is cheap to assert; assert it.
            raise RuntimeError(f"SFTTrainer: track builder built {track.batch_size} rows from {len(records)} records.")
        return self.stack.train_track(track, training_progress=training_progress)

    # ------------------------------------------------------------------
    # Validation loss (full set, exact)
    # ------------------------------------------------------------------

    def evaluate(self, step: int) -> float:
        """Weighted eval loss over the full validation set; logs ``eval/loss``."""
        loss_sum = 0.0
        weight_sum = 0.0
        batches = 0
        for records in self.data_source.iter_eval_batches(self.eval_batch_size, eval_num_samples=self.eval_num_samples):
            records = self._pad_to_dp(records)
            metrics = self.stack.eval_track(self.track_builder.build(records))
            loss_sum += float(metrics["loss"]) * float(metrics["weight"])
            weight_sum += float(metrics["weight"])
            batches += 1
        if weight_sum <= 0.0:
            logger.warning("SFTTrainer.evaluate: no eval data (eval_num_samples=%s).", self.eval_num_samples)
            return float("nan")
        eval_loss = loss_sum / weight_sum
        logger.info(
            "EVAL step %d  eval_loss=%.5f  (weight=%.0f over %d batches of <=%d)",
            step + 1,
            eval_loss,
            weight_sum,
            batches,
            self.eval_batch_size,
        )
        self.wandb_logger.log_eval(step + 1, {"loss": eval_loss})
        return eval_loss

    def _pad_to_dp(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Pad a partial eval batch up to a DP multiple with zero-weight rows.

        DP_SCATTER needs divisibility; dropping the tail would silently shrink
        the eval set. Padded rows are duplicates flagged ``_eval_pad`` — the
        track builders zero their loss weight, so coverage stays exact.
        """
        records = list(records)
        while len(records) % self.dp_size:
            pad = dict(records[-1])
            pad["_eval_pad"] = True
            pad["sample_id"] = f"{pad.get('sample_id', 'sft')}/pad{len(records)}"
            records.append(pad)
        return records

    # ------------------------------------------------------------------
    # Data-cursor sidecar (exact mid-epoch resume)
    # ------------------------------------------------------------------

    def _save_data_state(self, step: int, num_steps: int, *, save_interval: int, save_dir: Optional[str]) -> None:
        """Write the dataset cursor beside the checkpoint this step produced
        (same cadence/path arithmetic as :meth:`BaseTrainer.maybe_save_checkpoint`)."""
        if save_interval <= 0:
            return
        step_1 = step + 1
        if step_1 % save_interval != 0 and step_1 < num_steps:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step_1}", _DATA_STATE_FILENAME)
        with open(path, "w") as fh:
            json.dump(self.data_source.state_dict(), fh)

    def _load_data_state(self, load_dir: Optional[str], start_step: int) -> None:
        if not load_dir:
            return
        path = os.path.join(os.path.abspath(load_dir), _DATA_STATE_FILENAME)
        if os.path.exists(path):
            with open(path) as fh:
                self.data_source.load_state_dict(json.load(fh))
            logger.info("Restored dataset cursor from %s (epoch=%.3f)", path, self.data_source.epoch)
            return
        # Sidecar-less checkpoint: replay the stream to the resume point (exact
        # for a fixed seed — the shuffle is seed+epoch generated).
        logger.warning("No %s beside the checkpoint; fast-forwarding %d batches.", _DATA_STATE_FILENAME, start_step)
        for _ in range(start_step):
            self.data_source.get_samples(self.batch_size)

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
        save_mode: str = "auto",
    ) -> None:
        """``num_steps`` optimizer steps of ``records → build → train_track``.

        ``num_steps`` is the TOTAL budget (resume continues toward it); one
        step consumes ``batch_size`` samples, so N epochs ≈
        ``N * len(dataset) / batch_size`` steps (``train/epoch`` tracks the
        exact position).
        """
        start_step = self.maybe_load_checkpoint(load_dir, num_rollouts=num_steps)
        self._load_data_state(load_dir, start_step)
        self._init_wandb(num_rollouts=num_steps)
        try:
            if self.eval_interval > 0:
                self.evaluate(step=-1)  # baseline eval-loss at step 0
            for step in range(start_step, num_steps):
                t0 = time.perf_counter()
                training_progress = step / max(1, num_steps - 1)
                records = self.data_source.get_samples(self.batch_size)
                result = self.train_step(records, training_progress=training_progress)
                dt = time.perf_counter() - t0
                logger.info(
                    "step %d/%d  loss=%.5f grad_norm=%.4f lr=%.2e epoch=%.3f  %.1fs",
                    step + 1,
                    num_steps,
                    result.loss,
                    result.grad_norm,
                    result.lr,
                    self.data_source.epoch,
                    dt,
                )
                self.wandb_logger.log_step(
                    step + 1,
                    {
                        "train/loss": result.loss,
                        "train/grad_norm": result.grad_norm,
                        "train/lr": result.lr,
                        "train/epoch": self.data_source.epoch,
                        "perf/step_time_s": dt,
                        **{f"train/{k}": v for k, v in dict(result.metrics).items()},
                    },
                    prefix="",
                )
                if self.eval_interval > 0 and (step + 1) % self.eval_interval == 0:
                    self.evaluate(step=step)
                self.maybe_save_checkpoint(
                    step, num_steps, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
                self._save_data_state(step, num_steps, save_interval=save_interval, save_dir=save_dir)
        finally:
            self._finish_wandb()


__all__ = ["SFTTrainer"]
