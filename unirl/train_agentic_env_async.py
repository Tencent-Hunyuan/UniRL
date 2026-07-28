#!/usr/bin/env python
"""UniRL fully-async agentic ENV-reward training entry point (LIN-531).

Drives :class:`unirl.trainer.agentic_env_async.AsyncAgenticEnvTrainer` — the
disaggregated producer/consumer sibling of ``train_agentic_env`` (AgenticEnvTrainer).
Variable-depth episodes (ALFWorld runs 2 vs 15 turns) are the workload the partial-rollout
overlap targets: the async trainer overlaps generation with training on disjoint slabs
and checkpoints the in-flight tail at a turn boundary on each weight sync, instead of a
full barrier that waits for the slowest episode. ``tail_policy=drop`` is required because
a stateful episode is worker-local state and cannot currently resume from a carried
``Sample`` after checkpoint teardown.

Launch (single node, whole 8-GPU node; train_fraction=0.5 -> 4 train + 4 rollout):
  QWEN3_INSTRUCT_PATH=... ALFWORLD_DATA=... DATA_PATH=data/alfworld/train.jsonl \
  python -m unirl.train_agentic_env_async --config-name=alfworld/alfworld_grpo_async num_devices=8

This entrypoint serves every env-reward fully-async recipe; ALFWorld
(``examples/alfworld/``) is the reference environment.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_env_async import AsyncAgenticEnvTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="alfworld/alfworld_grpo_async")
def main(cfg: DictConfig) -> None:
    trainer = AsyncAgenticEnvTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        samples_per_prompt=cfg.sampling.samples_per_prompt,
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
        stop=cfg.get("stop"),
        train_fraction=cfg.get("train_fraction", 0.5),
        oversample_batch_size=cfg.get("oversample_batch_size"),
        buffer_max_staleness=cfg.get("buffer_max_staleness"),
        tail_policy=cfg.get("tail_policy", "drop"),
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
