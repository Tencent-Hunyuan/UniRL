"""Checkpoint loader for VideoAlign reward checkpoints.

Supports both layouts produced by the upstream trainer:

1. **Full state dict** (``save_full_model=True``) — ``checkpoint-K/model.pth``.
   Loaded directly into the freshly-constructed
   :class:`Qwen2VLRewardModelBT`.

2. **LoRA split** (the default for the published VideoAlign release) —
   ``checkpoint-K/adapter_model.safetensors`` (LoRA weights) plus
   ``checkpoint-K/non_lora_state_dict.pth`` (rm_head + special-token
   embeddings + any other ``requires_grad`` non-LoRA tensor). Both pieces
   are merged into the PEFT-wrapped model.

Includes the transformers>=5 compatibility shim
``base_model.model.model.*`` → ``base_model.model.model.language_model.*``
when the target model uses the new Qwen2-VL submodule layout but the
checkpoint was saved against the old one. The remap is a no-op when both
sides agree.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def _insert_adapter_name_into_state_dict(
    state_dict: Dict[str, torch.Tensor],
    adapter_name: str,
    parameter_prefix: str,
) -> Dict[str, torch.Tensor]:
    """Rewrite raw LoRA keys to match peft's ``{module}.lora_{A,B}.{adapter}.*`` layout."""
    peft_model_state_dict: Dict[str, torch.Tensor] = {}
    for key, val in state_dict.items():
        if parameter_prefix in key:
            suffix = key.split(parameter_prefix)[1]
            if "." in suffix:
                suffix_to_replace = ".".join(suffix.split(".")[1:])
                key = key.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
            else:
                key = f"{key}.{adapter_name}"
            peft_model_state_dict[key] = val
        else:
            peft_model_state_dict[key] = val
    return peft_model_state_dict


def _pick_checkpoint_dir(checkpoint_dir: str, checkpoint_step: Optional[int]) -> str:
    """Return the absolute path of the ``checkpoint-<step>`` subdir to load.

    ``checkpoint_step is None`` or ``-1`` selects the largest step found.
    A specific step falls back to the latest with a warning if missing.
    """
    candidates = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
    if not candidates:
        raise FileNotFoundError(
            f"No 'checkpoint-*' subdirectory under {checkpoint_dir!r}. "
            "Make sure you point ``reward_model_path`` at the directory that "
            "contains ``model_config.json`` alongside one or more ``checkpoint-K`` dirs."
        )
    candidates.sort(key=lambda x: int(x.split("-")[-1]), reverse=True)

    if checkpoint_step is None or checkpoint_step == -1:
        chosen = candidates[0]
        logger.info("Using latest VideoAlign checkpoint: %s", chosen)
        return chosen

    explicit = os.path.join(checkpoint_dir, f"checkpoint-{checkpoint_step}")
    if explicit in candidates:
        logger.info("Using requested VideoAlign checkpoint: %s", explicit)
        return explicit

    chosen = candidates[0]
    logger.warning(
        "Requested VideoAlign checkpoint-%s not found; falling back to latest %s",
        checkpoint_step,
        chosen,
    )
    return chosen


def load_model_from_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    checkpoint_step: Optional[int],
) -> Tuple[torch.nn.Module, str]:
    """Load VideoAlign weights into ``model`` from a ``checkpoint-K`` subdir.

    Args:
        model: an already-constructed reward model (PEFT-wrapped or not).
        checkpoint_dir: parent dir containing ``checkpoint-*`` subdirs.
        checkpoint_step: which step to load; ``None`` / ``-1`` → latest.

    Returns:
        ``(model, step_str)`` — the same model object (mutated in place by
        ``load_state_dict``) and the step number of the loaded checkpoint
        as a string (e.g. ``"3000"``), for downstream logging.
    """
    checkpoint_path = _pick_checkpoint_dir(checkpoint_dir, checkpoint_step)
    loaded_step = checkpoint_path.split("checkpoint-")[-1].split("/")[0]

    full_ckpt = os.path.join(checkpoint_path, "model.pth")
    lora_ckpt = os.path.join(checkpoint_path, "adapter_model.safetensors")
    non_lora_ckpt = os.path.join(checkpoint_path, "non_lora_state_dict.pth")

    def _remap_qwen_layout(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        new_state_dict: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("base_model.model.model.language_model.") or key.startswith(
                "base_model.model.model.visual."
            ):
                new_state_dict[key] = value
            elif key.startswith("base_model.model.model"):
                new_state_dict["base_model.model.model.language_model" + key[len("base_model.model.model") :]] = value
            elif key.startswith("base_model.model.visual"):
                new_state_dict["base_model.model.model.visual" + key[len("base_model.model.visual") :]] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    if os.path.exists(full_ckpt):
        model_state_dict = torch.load(full_ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(_remap_qwen_layout(model_state_dict))
    else:
        if not os.path.exists(lora_ckpt) or not os.path.exists(non_lora_ckpt):
            raise FileNotFoundError(
                f"Neither {full_ckpt!r} nor the LoRA pair "
                f"({lora_ckpt!r} + {non_lora_ckpt!r}) was found under {checkpoint_path!r}."
            )

        lora_state_dict = _remap_qwen_layout(safetensors.torch.load_file(lora_ckpt))
        non_lora_state_dict = _remap_qwen_layout(torch.load(non_lora_ckpt, map_location="cpu"))

        lora_state_dict = _insert_adapter_name_into_state_dict(
            lora_state_dict,
            adapter_name="default",
            parameter_prefix="lora_",
        )

        model_state_dict = model.state_dict()
        model_state_dict.update(non_lora_state_dict)
        model_state_dict.update(lora_state_dict)
        model.load_state_dict(model_state_dict)

    return model, loaded_step


__all__ = ["load_model_from_checkpoint"]
