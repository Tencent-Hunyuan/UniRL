#!/usr/bin/env python
"""UniRL diffusion training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.diffusion.DiffusionTrainer`.
The trainer owns the placement scope, sibling Remote wiring, and the
``train_step → train`` loop; this module just maps the loaded Hydra
config blocks to constructor kwargs.

Pairs with ``examples/diffusion/sd3/sd3_trainside.yaml`` (default) and
``examples/diffusion/sd3/sd3_vllmomni.yaml``. Switch with
``--config-name diffusion/sd3/sd3_vllmomni`` on the CLI.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.diffusion import DiffusionTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="diffusion/sd3/sd3_trainside")
def main(cfg: DictConfig) -> None:
    trainer = DiffusionTrainer(
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
        layout=cfg.get("layout", "colocate"),
        train_fraction=cfg.get("train_fraction", 0.5),
        reward_fraction=cfg.get("reward_fraction", 0.0),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", False),
        adv_use_global_std=cfg.get("adv_use_global_std", False),
        eval_interval=cfg.get("eval_interval", 0),
        eval_num_prompts=cfg.get("eval_num_prompts", 64),
        eval_samples_per_prompt=cfg.get("eval_samples_per_prompt", 4),
        eval_chunk_prompts=cfg.get("eval_chunk_prompts", 16),
        eval_cfg_text_scale=cfg.get("eval_cfg_text_scale", 4.0),
        eval_eta=cfg.get("eval_eta", 0.0),
        stage_config=cfg.get("stage_config"),
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
