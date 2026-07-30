"""Inference-only port of the VideoAlign reward model.

This subpackage contains everything needed to *load and forward* the
VideoAlign Qwen2-VL reward checkpoint, but **none** of the training-time
code (no ``VideoVLMRewardTrainer``, no GSB CSV data loader, no Bradley-Terry
loss). The original training pipeline lives at the upstream
``TIGER-AI-Lab/VideoScore`` / ``Tencent-Hunyuan/VideoAlign`` repos; mmrl
ships a vendored snapshot for its own RL training. UniRL does not need any
of that — at REFL rollout time we only call ``forward`` on the reward model
to read out three scalars per (video, prompt) pair.

Files
-----
- :mod:`prompt_template`  — ``build_prompt`` + the four prompt variants used
  by the published checkpoints.
- :mod:`configs`          — ``ModelConfig`` / ``PEFTLoraConfig`` /
  ``TrainingConfig`` (the *inference-relevant* subset of fields; loaded from
  the checkpoint's ``model_config.json``).
- :mod:`reward_model`     — ``Qwen2VLRewardModelBT`` — Qwen2-VL with an
  ``rm_head`` linear projection to (VQ, MQ, TA) scalars.
- :mod:`checkpoint`       — ``load_model_from_checkpoint`` (full / LoRA
  branches, plus the transformers>=5 key-remap).
- :mod:`factory`          — ``create_model_and_processor`` — builds the
  model, processor and (optionally) wraps with PEFT LoRA.
"""

from .checkpoint import load_model_from_checkpoint
from .configs import ModelConfig, PEFTLoraConfig, TrainingConfig
from .factory import create_model_and_processor
from .prompt_template import DIMENSION_DESCRIPTIONS, build_prompt
from .reward_model import Qwen2VLRewardModelBT

__all__ = [
    "DIMENSION_DESCRIPTIONS",
    "ModelConfig",
    "PEFTLoraConfig",
    "Qwen2VLRewardModelBT",
    "TrainingConfig",
    "build_prompt",
    "create_model_and_processor",
    "load_model_from_checkpoint",
]
