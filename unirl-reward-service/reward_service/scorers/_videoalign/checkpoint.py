"""Checkpoint loader — vendored from upstream ``utils.py``.

The upstream module mixes the loader with deepspeed/zero-3 helpers
(``maybe_zero_3``, ``get_peft_state_*``) used only at training time.
We keep just the inference loader plus its private helper.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Tuple

import safetensors
import torch

from reward_service.logging_utils import get_logger

logger = get_logger(__name__)


def _insert_adapter_name_into_state_dict(
    state_dict: Dict[str, torch.Tensor], adapter_name: str, parameter_prefix: str
) -> Dict[str, torch.Tensor]:
    """Remap a raw LoRA state dict so its keys match a ``peft.PeftModel``.

    LoRA weights saved via ``safetensors.torch.save_file`` are keyed
    like ``base_model.model.<...>.lora_A.weight``; the live ``PeftModel``
    expects ``base_model.model.<...>.lora_A.<adapter_name>.weight``.
    Insert the adapter name in place.
    """
    peft_state_dict: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_state_dict[key] = val
        else:
            peft_state_dict[key] = val
    return peft_state_dict


def load_model_from_checkpoint(
    model, checkpoint_dir: str, checkpoint_step: int | None
) -> Tuple[object, str]:
    """Load weights from ``<checkpoint_dir>/checkpoint-<step>/`` into ``model``.

    Two on-disk layouts are supported (matching upstream):

    1. **Full checkpoint**: ``model.pth`` — a torch ``state_dict``.
    2. **LoRA + non-LoRA pair**: ``adapter_model.safetensors`` (LoRA
       deltas) + ``non_lora_state_dict.pth`` (e.g. updated visual
       merger / regression head). Merged into the live model's
       state_dict and reloaded.

    Args:
        model: Live ``Qwen2VLRewardModelBT`` or wrapped ``PeftModel``.
        checkpoint_dir: Directory containing ``checkpoint-<step>/``
            subdirectories.
        checkpoint_step: ``None`` or ``-1`` ⇒ pick the latest step;
            otherwise pick that exact step (falls back to latest with a
            warning if missing).

    Returns:
        ``(model, checkpoint_step_str)`` — the same model object (for
        chaining) and the loaded step as a string.
    """
    checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    checkpoint_paths.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no checkpoint-* subdirectories found under {checkpoint_dir}"
        )

    if checkpoint_step is None or checkpoint_step == -1:
        checkpoint_path = checkpoint_paths[0]
        logger.info("checkpoint step not provided, using latest: %s", checkpoint_path)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
        if checkpoint_path not in checkpoint_paths:
            checkpoint_path = checkpoint_paths[0]
            logger.warning(
                "checkpoint step %s not found, falling back to latest: %s",
                checkpoint_step, checkpoint_path,
            )
        else:
            logger.info("loading checkpoint step %s: %s", checkpoint_step, checkpoint_path)

    loaded_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")

    if os.path.exists(full_ckpt):
        model_state_dict = torch.load(full_ckpt, map_location="cpu")
        model.load_state_dict(model_state_dict)
        return model, loaded_step

    if not (os.path.exists(lora_ckpt) and os.path.exists(non_lora_ckpt)):
        raise FileNotFoundError(
            f"checkpoint {checkpoint_path} contains neither model.pth nor "
            f"adapter_model.safetensors + non_lora_state_dict.pth"
        )

    lora_state_dict = safetensors.torch.load_file(lora_ckpt)
    non_lora_state_dict = torch.load(non_lora_ckpt, map_location="cpu")
    lora_state_dict = _insert_adapter_name_into_state_dict(
        lora_state_dict, adapter_name="default", parameter_prefix="lora_"
    )

    merged = model.state_dict()
    merged.update(non_lora_state_dict)
    merged.update(lora_state_dict)
    model.load_state_dict(merged)
    return model, loaded_step
