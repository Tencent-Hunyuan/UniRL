#!/usr/bin/env python
"""UniRL agentic ENV-reward (multi-turn, env-sourced reward) training entry point (LIN-519).

Drives :class:`unirl.trainer.agentic_env.AgenticEnvTrainer` over the
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine` with an interactive
:class:`~unirl.rollout.env.protocol.Environment`. Sibling of ``train_agentic.py``;
the reward is the environment's own per-trajectory return — task-success or a shaped
signal, attached to each trajectory by the engine — so no reward backend is scored and
the recipe's ``reward`` block is built but unused.

Launch (single node; rank 0 owns the driver + the agentic coordinator):
  QWEN3_INSTRUCT_PATH=/path/to/Qwen3-8B DATA_PATH=/path/to/alfworld_games.jsonl \
  ALFWORLD_DATA=/path/to/alfworld/data \
  python -m unirl.train_agentic_env --config-name=alfworld/alfworld_grpo num_devices=8

This entrypoint serves every env-reward agentic recipe; ALFWorld
(:class:`~unirl.rollout.env.alfworld.AlfworldEnv`, ``examples/alfworld/``) is the
reference environment.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_env import AgenticEnvTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="alfworld/alfworld_grpo")
def main(cfg: DictConfig) -> None:
    trainer = AgenticEnvTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        adv_normalization_scope=cfg.get("adv_normalization_scope", "group"),
        normalize_adv_by_std=cfg.get("normalize_adv_by_std", True),
        balance_shards=cfg.get("balance_shards", False),
        eval_interval=cfg.get("eval_interval", 0),
        stop=cfg.get("stop"),
    )
    trainer.train(
        num_rollouts=cfg.get("num_rollouts", 100),
        weight_sync_interval=cfg.get("weight_sync_interval", 1),
        save_interval=cfg.get("save_interval", 0),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=cfg.get("save_mode", "auto"),
    )


if __name__ == "__main__":
    main()
