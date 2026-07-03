#!/usr/bin/env python
"""UniRL ReFL training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.refl.RewardBackpropTrainer` — direct
differentiable-reward backprop (DRaFT-K) for SD3. Pairs with
``examples/diffusion/refl_sd3.yaml``. Like ``train_diffusion``, the Ray cluster
is started by the launcher (``ray start --head`` + ``RAY_ADDRESS=auto``).
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.refl import RewardBackpropTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="diffusion/refl_sd3")
def main(cfg: DictConfig) -> None:
    trainer = RewardBackpropTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        policy_cfg=cfg.policy,
        reward_cfg=cfg.reward,
        data_source_cfg=cfg.data_source,
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        policy_device_fraction=float(cfg.get("policy_device_fraction", 0.75)),
        logging_cfg=cfg.get("logging"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "adapter")),
    )


if __name__ == "__main__":
    main()
