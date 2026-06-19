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
        freeze_llm=cfg.get("freeze_llm", False),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        weight_sync_interval=int(cfg.get("weight_sync_interval", 1)),
    )


if __name__ == "__main__":
    main()
