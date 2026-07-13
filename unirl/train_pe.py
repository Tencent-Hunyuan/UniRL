#!/usr/bin/env python
"""UniRL v2 PE (Prompt Enhancement) joint training entry point.

Thin Hydra wrapper around :class:`unirl.trainer.pe.PETrainer`. The
trainer owns the placement scope, wires the two sibling stacks (diffusion +
ar), composes the PEPipeline, and runs the ``train_step → train`` loop; this
module just maps the loaded Hydra config blocks to constructor kwargs.

Pairs with ``examples/pe/pe_trainside_pickscore.yaml``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.pe import PETrainer


@hydra.main(version_base=None, config_path="../examples", config_name="pe/pe_trainside_pickscore")
def main(cfg: DictConfig) -> None:
    trainer = PETrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        diffusion_cfg=cfg.diffusion,
        ar_cfg=cfg.ar,
        rollout_cfg=cfg.rollout,
        reward_cfg=cfg.reward,
        data_source_cfg=cfg.data_source,
        sampling_cfg=cfg.sampling,
        sync_cfg=cfg.get("sync"),
        logging_cfg=cfg.get("logging"),
        enable_fsdp_offload=cfg.get("enable_fsdp_offload", False),
        pe_cfg=cfg.get("pe"),
        freeze_llm=cfg.get("freeze_llm", False),
        diffusion_group_scope=cfg.get("diffusion_group_scope", "rewrite"),
        eval_interval=int(cfg.get("eval_interval", 0)),
        eval_num_prompts=int(cfg.get("eval_num_prompts", cfg.batch_size)),
        eval_cfg_text_scale=float(cfg.get("eval_cfg_text_scale", 4.0)),
        eval_eta=float(cfg.get("eval_eta", 0.0)),
        eval_rewards_cfg=cfg.get("eval_rewards"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "auto")),
    )


if __name__ == "__main__":
    main()
