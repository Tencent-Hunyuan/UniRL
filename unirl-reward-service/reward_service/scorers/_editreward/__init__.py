"""Vendored EditReward inference subset.

Minimal slice of the upstream EditReward repo
(https://github.com/TIGER-AI-Lab/EditReward) needed to run the
EditReward Qwen2.5-VL / MiMo-VL reward model in inference mode.

Training-only parts (trainer, dataset loaders, deepspeed, fire, matplotlib,
tensorboard, datasets) are intentionally stripped. This eliminates ~30
heavy dependencies that conflict with the reward service's base image.

Provenance: TIGER-AI-Lab/EditReward @ commit (2026-06-10)
Upstream files → vendored:
  - model/qwen2_5_vl_trainer.py  → model.py (model class only, no trainer)
  - inference_edit.py            → inferencer.py
  - utils/parser.py              → configs.py
  - dataset/prompts.py           → prompts.py
  - dataset/data_collator_qwen_edit.py → (inlined prompt strings)
  - dataset/utils.py             → vision_process.py
"""

from reward_service.scorers._editreward.inferencer import EditRewardInferencer

__all__ = ["EditRewardInferencer"]
