#!/usr/bin/env python
"""UniRL colocate PARTIAL-ROLLOUT ALFWorld (env-reward agentic) training entry point (LIN-531).

Drives :class:`unirl.trainer.agentic_partial.AgenticEnvPartialTrainer` — the colocate/synchronous
partial-rollout sibling of ``train_alfworld`` (barrier `AgenticEnvTrainer`). Over-samples ALFWorld
episodes, commits the fastest `batch_size` complete GRPO groups, and `abort`s the slow tail at a
turn boundary. Recipes use ``tail_policy: drop`` because ALFWorld is a stateful-episode env (a
carried partial restarts). Keeps all GPUs for generation (colocate) while cutting the straggler —
the sweet spot the fully-async trainer missed on ALFWorld.

Launch (single node, whole 8-GPU node):
  QWEN3_INSTRUCT_PATH=... ALFWORLD_DATA=... DATA_PATH=data/alfworld/train.jsonl \
  python -m unirl.train_partial_alfworld --config-name=alfworld/alfworld_grpo_partial num_devices=8
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.agentic_partial import AgenticEnvPartialTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="alfworld/alfworld_grpo_partial")
def main(cfg: DictConfig) -> None:
    trainer = AgenticEnvPartialTrainer(
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
        balance_shards=cfg.get("balance_shards", False),
        eval_interval=cfg.get("eval_interval", 0),
        stop=cfg.get("stop"),
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
