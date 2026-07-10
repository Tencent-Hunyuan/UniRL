#!/usr/bin/env python
"""UniRL SFT / behavior-cloning training entry point (Hydra-native).

Thin wrapper around :class:`unirl.trainer.sft.SFTTrainer` — supervised
finetuning driven by a config-chosen task adapter. The task owns the model
family and its loss, so the same entry point serves both autoregressive
(``examples/sft/qwen3_*.yaml``, next-token cross-entropy) and diffusion
(``examples/sft/sd3_*.yaml``, flow-matching MSE) families. Like ``train_refl``,
the Ray cluster is started by the launcher (``ray start --head`` +
``RAY_ADDRESS=auto``). Pick a recipe on the CLI: ``--config-name sft/<recipe>``.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from unirl.trainer.sft import SFTTrainer


@hydra.main(version_base=None, config_path="../examples", config_name="sft/qwen3_sft_smoke")
def main(cfg: DictConfig) -> None:
    trainer = SFTTrainer(
        cfg=cfg,
        batch_size=cfg.batch_size,
        policy_cfg=cfg.policy,
        data_source_cfg=cfg.data_source,
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        eval_interval=int(cfg.get("eval_interval", 0)),
        eval_num_samples=int(cfg.get("eval_num_samples", 1)),
        logging_cfg=cfg.get("logging"),
    )
    trainer.train(
        num_rollouts=int(cfg.get("num_rollouts", 100)),
        save_interval=int(cfg.get("save_interval", 0)),
        save_dir=cfg.get("save_dir"),
        load_dir=cfg.get("load_dir"),
        save_mode=str(cfg.get("save_mode", "auto")),
    )


if __name__ == "__main__":
    main()
