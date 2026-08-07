#!/usr/bin/env python
"""WAN Self-Forcing DMD training entry point."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.self_forcing import WAN21SelfForcingTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="self_forcing/wan21_t2v_ucf101_dmd")
def main(cfg: DictConfig) -> None:
    trainer = WAN21SelfForcingTrainer(
        cfg=cfg,
        batch_size=int(cfg.batch_size),
        generator_bundle_cfg=cfg.generator_bundle,
        generator_pipeline_cfg=cfg.generator_pipeline,
        generator_backend_cfg=cfg.generator_backend,
        fake_score_bundle_cfg=cfg.fake_score_bundle,
        fake_score_pipeline_cfg=cfg.fake_score_pipeline,
        fake_score_backend_cfg=cfg.fake_score_backend,
        real_score_bundle_cfg=cfg.real_score_bundle,
        real_score_pipeline_cfg=cfg.real_score_pipeline,
        real_score_backend_cfg=cfg.real_score_backend,
        stack_cfg=cfg.stack,
        track_builder_cfg=cfg.track_builder,
        data_source_cfg=cfg.data_source,
        logging_cfg=cfg.get("logging"),
    )
    trainer.train(
        num_steps=int(cfg.get("num_steps", 300)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "full")),
    )


if __name__ == "__main__":
    main()
