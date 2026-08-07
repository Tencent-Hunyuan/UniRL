"""REFLTrainer — recipe driver for WAN REFL/BPTT (video reward backprop).

Two roles, always — a :class:`experimental.refl.roles.ReflActorRole` (FSDP WAN +
grad BPTT sampling + optimizer) and a frozen differentiable video reward
(:class:`unirl.reward.service.RewardService`), colocated on the same worker
slab so decoded video never leaves the GPU. Each step runs, under the
distributed ``enable_grad()`` context::

    gen = actor.generate_samples(texts, images, params)   # grad through BPTT window + VAE
    rew = reward.score_differentiable(gen.decoded, prompts, records)
    actor.forward_backward_loss(rewards=rew, kl_loss=gen.kl_loss)
    # ctx exit → grads route reward → decode → DiT LoRA params
    actor.step(max_grad_norm)

No advantages / replay / ratio / rollout-engine / weight-sync — REFL is not a
PG-RL loop. Success signal: the reward curve rises.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.distributed.tensor.grad_context import enable_grad
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import remote_hydra

logger = logging.getLogger(__name__)


def _text_inputs(inputs: Sample) -> Texts:
    """Read the text root from a data-source input ``Sample``."""
    prompts = inputs.parts[0].primitives.get("text") if inputs.parts else None
    if not isinstance(prompts, Texts):
        raise TypeError(f"REFL data-source root requires Texts, got {type(prompts).__name__}.")
    return prompts


def _image_inputs(inputs: Sample, *, text_count: int) -> Optional[Images]:
    """Read and validate the optional packed I2V first-frame conditioning."""
    image_inputs: List[Images] = []
    for part_index, part in enumerate(inputs.parts[1:], start=1):
        if "image" not in part.primitives:
            continue
        image = part.primitives["image"]
        if not isinstance(image, Images):
            raise TypeError(f"REFL I2V condition at part {part_index} requires Images, got {type(image).__name__}.")
        image_inputs.append(image)

    if not image_inputs:
        return None
    if len(image_inputs) != 1:
        raise ValueError(f"REFL supports exactly one I2V image condition batch, got {len(image_inputs)}.")

    images = image_inputs[0]
    if len(images) != text_count:
        raise ValueError(f"REFL I2V image count {len(images)} != text count {text_count}.")
    return images


def _records(inputs: Sample) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Per-sample data-source metadata (e.g. ``ref_video_path`` for Face reward)."""
    metadata = inputs.parts[0].metadata if inputs.parts else None
    return list(metadata) if metadata else None


class REFLTrainer(BaseTrainer):
    """REFL/BPTT recipe driver: actor + differentiable reward, 3-RPC step."""

    def __init__(self, *, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg, logging_cfg=cfg.get("logging"))
        self.cfg = cfg
        self.batch_size = int(cfg.batch_size)
        self.max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
        self.data_source = instantiate(cfg.data_source)

        sampling = build_sampling_dict(cfg.sampling)
        params = sampling.get("diffusion")
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError("REFLTrainer requires cfg.sampling to build DiffusionSamplingParams.")
        if total_samples_per_prompt(sampling) != 1:
            raise ValueError(
                "REFLTrainer trains one sample per prompt (no advantage groups); set sampling.samples_per_prompt=1."
            )
        self.sampling_params = params

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.actor = remote_hydra(cfg.actor)
            self.reward = remote_hydra(cfg.reward)
        self.actor.initialize()
        self.backend = self.actor

        adp, rdp = self.actor.dp_size, self.reward.dp_size
        if self.batch_size % adp or self.batch_size % rdp:
            raise ValueError(f"batch_size={self.batch_size} must be divisible by actor dp={adp} and reward dp={rdp}")
        logger.info(
            "REFLTrainer ready: actor dp=%d reward dp=%d batch=%d max_grad_norm=%.2f",
            adp,
            rdp,
            self.batch_size,
            self.max_grad_norm,
        )

    def train_step(self, inputs: Sample) -> Dict[str, float]:
        """One enable_grad() generate → score → backward, then optimizer step."""
        t0 = time.perf_counter()
        texts = _text_inputs(inputs)
        images = _image_inputs(inputs, text_count=len(texts))
        records = _records(inputs)
        with enable_grad():
            gen = self.actor.generate_samples(
                texts=texts,
                images=images,
                params=self.sampling_params,
            )
            rewards = self.reward.score_differentiable(gen.decoded, list(texts.texts), records)
            loss_metrics = self.actor.forward_backward_loss(rewards=rewards, kl_loss=gen.kl_loss)
        step_result = self.actor.step(max_grad_norm=self.max_grad_norm)
        if isinstance(step_result, list):  # BROADCAST → one result per worker
            step_result = step_result[0]

        return {
            "loss": float(np.mean(loss_metrics.loss)),
            "reward_loss": float(np.mean(loss_metrics.reward_loss)),
            "kl_loss": float(np.mean(loss_metrics.kl_loss)),
            "reward_mean": float(np.mean(loss_metrics.reward_mean)),
            "grad_norm": float(step_result.get("grad_norm", 0.0)),
            "lr": float(step_result.get("lr", 0.0)),
            "step_time_s": time.perf_counter() - t0,
        }

    def train(
        self,
        *,
        num_rollouts: Optional[int] = None,
        save_interval: Optional[int] = None,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: Optional[str] = None,
    ) -> None:
        cfg = self.cfg
        num_rollouts = int(num_rollouts if num_rollouts is not None else cfg.get("num_rollouts", 100))
        save_interval = int(save_interval if save_interval is not None else cfg.get("save_interval", 0))
        save_dir = save_dir if save_dir is not None else cfg.get("save_dir")
        load_dir = load_dir if load_dir is not None else cfg.get("load_dir")
        save_mode = str(save_mode if save_mode is not None else cfg.get("save_mode", "adapter"))

        start = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        for _ in range(start):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(start, num_rollouts):
                inputs = self.data_source.get_samples(self.batch_size)
                metrics = self.train_step(inputs)
                logger.info(
                    "rollout %d/%d  reward=%.4f loss=%.4f kl=%.4f grad_norm=%.4f  %.1fs",
                    rollout_id + 1,
                    num_rollouts,
                    metrics["reward_mean"],
                    metrics["loss"],
                    metrics["kl_loss"],
                    metrics["grad_norm"],
                    metrics["step_time_s"],
                )
                self.wandb_logger.log_step(
                    rollout_id + 1,
                    {
                        "rollout/mean_reward": metrics["reward_mean"],
                        "train/loss": metrics["loss"],
                        "train/reward_loss": metrics["reward_loss"],
                        "train/kl_loss": metrics["kl_loss"],
                        "train/grad_norm": metrics["grad_norm"],
                        "train/lr": metrics["lr"],
                        "perf/step_time_s": metrics["step_time_s"],
                    },
                    prefix="",
                )
                self.maybe_save_checkpoint(
                    rollout_id,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
        finally:
            self._finish_wandb()


__all__ = ["REFLTrainer"]
