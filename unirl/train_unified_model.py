#!/usr/bin/env python
"""UniRL v2 HunyuanImage3 training entry point (Hydra-native)."""

from __future__ import annotations

import warnings

import hydra
from omegaconf import DictConfig

from unirl.trainer.unified_model import UnifiedModelTrainer


def _resolve_task_config(cfg: DictConfig):
    if "stage_config" not in cfg:
        return cfg.get("task_config")
    if "task_config" in cfg:
        raise ValueError("Specify only task_config; do not set deprecated stage_config alongside it")
    warnings.warn(
        "`stage_config` is deprecated; rename the recipe key to `task_config`",
        FutureWarning,
        stacklevel=2,
    )
    return cfg.get("stage_config")


@hydra.main(version_base=None, config_path="../examples", config_name="unified_model/hi3_vllmomni")
def main(cfg: DictConfig) -> None:
    trainer = UnifiedModelTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        ar_rollout_cfg=cfg.get("ar_rollout"),
        dit_rollout_cfg=cfg.get("dit_rollout"),
        rollout_cfg=cfg.get("rollout"),
        reward_cfg=cfg.reward,
        ar_algorithm_cfg=cfg.algorithm.ar,
        image_algorithm_cfg=cfg.algorithm.image,
        stack_cfg=cfg.stack,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        task_config=_resolve_task_config(cfg),
        sync_cfg=cfg.get("sync"),
        dump_dir=cfg.get("dump_dir"),
        logging_cfg=cfg.get("logging"),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", True),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", cfg.batch_size),
        eval_cfg_text_scale=float(cfg.get("eval_cfg_text_scale", 4.0)),
        eval_eta=float(cfg.get("eval_eta", 0.0)),
        eval_rewards_cfg=cfg.get("eval_rewards"),
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
