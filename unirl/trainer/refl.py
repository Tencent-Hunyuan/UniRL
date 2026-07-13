"""RewardBackpropTrainer — driver orchestrator for ReFL (direct reward backprop).

Two roles, always: a :class:`ReFLPolicy` (FSDP SD3 + grad DRaFT-K sampling +
optimizer) and a frozen differentiable reward (:class:`RewardService`), placed on
disjoint device fractions. Each step runs, under the distributed
``enable_grad()`` context::

    img = policy.sample_and_decode(prompts)      # grad through final K steps + VAE
    rew = reward.score_differentiable(img)       # frozen reward, grad → image
    policy.loss_backward(rew)                     # -reward.mean() seed
    # ctx exit → grad lands on FSDP transformer params
    policy.optimizer_step(max_grad_norm)

No advantages / replay / ratio / segment / weight-sync — those are PG-RL concepts
ReFL does not use. Success signal: the reward curve rises.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.distributed.tensor.grad_context import enable_grad
from unirl.distributed.tensor.ref import hydrate
from unirl.trainer.base import BaseTrainer
from unirl.types.primitives import Texts
from unirl.utils.hydra import remote_hydra

logger = logging.getLogger(__name__)


class RewardBackpropTrainer(BaseTrainer):
    """ReFL trainer: policy + frozen differentiable reward, grad via enable_grad()."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        policy_cfg: DictConfig,
        reward_cfg: DictConfig,
        data_source_cfg: DictConfig,
        max_grad_norm: float = 1.0,
        reward_fraction: float = 0.25,
        eval_interval: int = 0,
        eval_num_prompts: int = 12,
        eval_cfg_text_scale: float = 4.0,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = int(batch_size)
        self.max_grad_norm = float(max_grad_norm)
        # Periodic eval on the eval set (run.eval_data_path), logged under eval/*.
        # eval_interval=0 disables it. ReFL has no rollout engine / tracks: eval
        # samples via ReFLPolicy.eval_sample (deterministic ODE, no grad) and scores
        # with the same differentiable reward, mean logged as eval/reward.
        # eval_cfg_text_scale sets the eval CFG strength (maps onto the SD3-family
        # guidance_scale; recipes train at guidance_scale=1.0 = no CFG; set it to
        # the train value to eval in the training regime instead) — mirrors
        # DiffusionTrainer's knob of the same name.
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.data_source = instantiate(data_source_cfg)

        # Unified reward placement — SAME knob/semantics as DiffusionTrainer's
        # ``reward_fraction``: ``> 0`` carves the (frozen, differentiable) reward
        # its OWN disjoint slab (the tail of the pool), policy takes the rest; the
        # DRaFT-K gradient crosses the slab boundary back into the policy via the
        # distributed ``enable_grad()`` context. ``== 0`` colocates reward on the
        # policy's cards (cheapest grad) at the cost of sharing its GPU memory.
        reward_frac = float(reward_fraction)
        if not 0.0 <= reward_frac < 1.0:
            raise ValueError(f"reward_fraction must be in [0, 1), got {reward_frac}")
        if reward_frac > 0.0:
            with placement(self.pool, fraction=1.0 - reward_frac, shared_workers=True):
                self.policy = remote_hydra(policy_cfg)
            with placement(self.pool, fraction=reward_frac, shared_workers=True):
                self.reward = remote_hydra(reward_cfg)
        else:
            with placement(self.pool, fraction=1.0, shared_workers=True):
                self.policy = remote_hydra(policy_cfg)
                self.reward = remote_hydra(reward_cfg)

        self.policy.initialize()
        # BaseTrainer.maybe_save/load_checkpoint operate on ``self.backend``.
        self.backend = self.policy

        pdp, rdp = self.policy.dp_size, self.reward.dp_size
        if self.batch_size % pdp or self.batch_size % rdp:
            raise ValueError(f"batch_size={self.batch_size} must be divisible by policy dp={pdp} and reward dp={rdp}")
        logger.info(
            "RewardBackpropTrainer ready: policy dp=%d reward dp=%d batch=%d max_grad_norm=%.2f",
            pdp,
            rdp,
            self.batch_size,
            self.max_grad_norm,
        )

    def train_step(self, prompts: Texts, *, rollout_id: int) -> Tuple[float, float, float]:
        """One enable_grad() sample → score → backward → step. Returns
        (mean_reward, grad_norm, step_time_s)."""
        t0 = time.perf_counter()
        with enable_grad():
            images = self.policy.sample_and_decode(prompts=prompts, rollout_id=rollout_id)
            rewards = self.reward.score_differentiable(images=images, prompts=prompts)
            # Detached value for logging (does not disturb the worker-side graph).
            mean_reward = float(hydrate(rewards).float().mean().item())
            self.policy.loss_backward(rewards=rewards)
        grad_norm = self.policy.optimizer_step(max_grad_norm=self.max_grad_norm)
        if isinstance(grad_norm, list):  # BROADCAST → one result per worker
            grad_norm = grad_norm[0]
        self.policy.zero_grad()
        return mean_reward, float(grad_norm or 0.0), time.perf_counter() - t0

    def evaluate(self, step: int) -> float:
        """Periodic eval — mean reward over the eval prompt set (no training).

        ReFL has no rollout engine or tracks, so this mirrors :meth:`train_step`'s
        sample→score path (minus ``enable_grad``/backward): pull
        ``eval_num_prompts`` eval prompts (``run.eval_data_path``), sample images
        with :meth:`ReFLPolicy.eval_sample` (deterministic ODE, ``model.eval()`` +
        ``no_grad``, CFG at ``eval_cfg_text_scale``), score with the differentiable
        reward, and log the mean under ``eval/reward``. Returns it.

        ``step`` keys the wandb log axis (and ``eval_sample``'s seed), mirroring
        :meth:`DiffusionTrainer.evaluate` — so re-running a checkpoint via
        ``num_rollouts=0 load_dir=checkpoint-k`` (→ baseline ``evaluate(k)``) evals
        the restored weights at the same step. NOTE: eval is NOT bit-exact — the
        init latent is drawn unseeded (see ``eval_sample``), so A vs B agree only
        within sampling noise (~σ), not byte-identically.

        Chunked by ``self.batch_size`` — both ``eval_sample`` and
        ``score_differentiable`` are DP_SCATTER, and ``batch_size`` is validated
        divisible by both the policy and reward dp sizes in ``__init__``; a ragged
        tail (``eval_num_prompts`` not a multiple of ``batch_size``) is floored off.
        """
        eval_inputs = self.data_source.get_eval_samples(self.eval_num_prompts)
        prompts = eval_inputs.primitives["text"]
        if not isinstance(prompts, Texts):
            prompts = Texts(texts=list(prompts))
        texts = list(prompts.texts)
        chunk = max(1, self.batch_size)
        usable = len(texts) - len(texts) % chunk or len(texts)
        reward_sum, reward_n = 0.0, 0
        for start in range(0, usable, chunk):
            sub = Texts(texts=texts[start : start + chunk])
            images = self.policy.eval_sample(prompts=sub, rollout_id=step, guidance_scale=self.eval_cfg_text_scale)
            rewards = self.reward.score_differentiable(images=images, prompts=sub)
            r = hydrate(rewards).float()
            reward_sum += float(r.sum().item())
            reward_n += int(r.numel())
        mean_reward = reward_sum / max(1, reward_n)
        logger.info(
            "EVAL step %d  eval_reward(%d prompts, cfg=%.1f)=%.4f",
            step,
            self.eval_num_prompts,
            self.eval_cfg_text_scale,
            mean_reward,
        )
        self.wandb_logger.log_eval(step, {"reward": mean_reward})
        return mean_reward

    def train(
        self,
        *,
        num_rollouts: int,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "adapter",
    ) -> None:
        start = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            if self.eval_interval > 0:
                self.evaluate(start)  # baseline eval before any training
            for rollout_id in range(start, num_rollouts):
                inputs = self.data_source.get_samples(self.batch_size)
                prompts = inputs.primitives["text"]
                if not isinstance(prompts, Texts):
                    prompts = Texts(texts=list(prompts))
                mean_reward, grad_norm, dt = self.train_step(prompts, rollout_id=rollout_id)
                logger.info(
                    "rollout %d/%d  reward=%.4f grad_norm=%.4f  %.1fs",
                    rollout_id + 1,
                    num_rollouts,
                    mean_reward,
                    grad_norm,
                    dt,
                )
                self.wandb_logger.log_step(
                    rollout_id + 1,
                    {
                        "rollout/mean_reward": mean_reward,
                        "train/loss": -mean_reward,
                        "train/grad_norm": grad_norm,
                        "perf/step_time_s": dt,
                    },
                    prefix="",
                )
                # eval(k) BEFORE save(checkpoint-k) at the same step, so a
                # resumed checkpoint re-runs the same eval (A/B consistency).
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id + 1)
                self.maybe_save_checkpoint(
                    rollout_id,
                    num_rollouts,
                    save_interval=save_interval,
                    save_dir=save_dir,
                    save_mode=save_mode,
                )
        finally:
            self._finish_wandb()


__all__ = ["RewardBackpropTrainer"]
