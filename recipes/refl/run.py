#!/usr/bin/env python
"""Hydra entry point for the recipes.refl REFL recipe."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from recipes.refl.trainer import REFLTrainer


@hydra.main(version_base=None, config_path="configs", config_name="wan22_i2v_face_refl")
def main(cfg: DictConfig) -> None:
    trainer = REFLTrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
