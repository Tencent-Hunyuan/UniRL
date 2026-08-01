#!/usr/bin/env python
"""Hydra entry point for the experimental.refl REFL recipe."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from experimental.refl.trainer import REFLTrainer


@hydra.main(version_base=None, config_path="examples", config_name="wan22_i2v_face_refl")
def main(cfg: DictConfig) -> None:
    trainer = REFLTrainer(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    main()
