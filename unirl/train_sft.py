#!/usr/bin/env python
"""UniRL SFT training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.sft.SFTTrainer` — supervised
finetuning for any bundled model family (AR cross-entropy / diffusion
flow-matching), selected entirely by the recipe's ``_target_`` dotpaths.
Pairs with ``examples/sft/*.yaml``. Like the other entrypoints, the Ray
cluster is started by the launcher (``ray start --head`` + ``RAY_ADDRESS=auto``).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.sft import SFTTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="sft/qwen3_sft")
def main(cfg: DictConfig) -> None:
    trainer = SFTTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        bundle_cfg=cfg.bundle,
        pipeline_cfg=cfg.pipeline,
        backend_cfg=cfg.backend,
        algorithm_cfg=cfg.algorithm,
        stack_cfg=cfg.stack,
        track_builder_cfg=cfg.track_builder,
        data_source_cfg=cfg.data_source,
        logging_cfg=cfg.get("logging"),
        eval_interval=int(cfg.get("eval_interval", 0)),
        eval_batch_size=int(cfg.get("eval_batch_size", 8)),
        eval_num_samples=int(cfg.get("eval_num_samples", -1)),
    )
    trainer.train(
        num_steps=int(cfg.get("num_steps", 100)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "auto")),
    )


if __name__ == "__main__":
    main()
