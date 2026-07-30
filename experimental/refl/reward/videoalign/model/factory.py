"""Build the VideoAlign reward model + Qwen2-VL processor.

Inference-only counterpart to ``create_model_and_processor`` in
``mmrl/recipes/rewards/videoalign/vendor/videoalign/train_reward.py``.
Specifically:

- Drops the ``trl.get_kbit_device_map`` / ``trl.get_quantization_config``
  dependency — we never 4-/8-bit quantise a reward model at REFL time.
- Drops the optimiser / loss / dataset side entirely.
- Keeps the optional LoRA wrapping and the optional ``<|VQ_reward|>`` /
  ``<|MQ_reward|>`` / ``<|TA_reward|>`` special-token registration, since
  both are needed to construct a *load-able* parameter graph for the
  published checkpoints.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
from transformers import AutoProcessor

from .configs import ModelConfig, PEFTLoraConfig, TrainingConfig
from .reward_model import Qwen2VLRewardModelBT

logger = logging.getLogger(__name__)


def _find_target_linear_names(
    model: torch.nn.Module,
    num_lora_modules: int = -1,
    lora_namespan_exclude: Optional[list] = None,
) -> list:
    """Discover LoRA target modules by introspecting the model graph.

    Mirrors the upstream selection rule: every ``nn.Linear`` / ``nn.Embedding``
    whose qualified name doesn't contain any excluded namespace keyword.
    ``num_lora_modules > 0`` truncates to the last N modules (the upstream
    knob for cheaply LoRA-tuning only the top of the network).
    """
    lora_namespan_exclude = lora_namespan_exclude or []
    if isinstance(lora_namespan_exclude, str):
        lora_namespan_exclude = [lora_namespan_exclude]

    target_classes = (torch.nn.Linear, torch.nn.Embedding)
    out = []
    for name, module in model.named_modules():
        if any(keyword in name for keyword in lora_namespan_exclude):
            continue
        if isinstance(module, target_classes):
            out.append(name)

    if num_lora_modules > 0:
        out = out[-num_lora_modules:]
    return out


def create_model_and_processor(
    model_config: ModelConfig,
    peft_lora_config: PEFTLoraConfig,
    training_args: TrainingConfig,
    cache_dir: Optional[str] = None,
) -> Tuple[torch.nn.Module, AutoProcessor, object]:
    """Build the Qwen2-VL reward model + matching processor.

    Args:
        model_config: backbone configuration (matches the saved
            ``model_config`` block in ``model_config.json``).
        peft_lora_config: LoRA wiring (matches ``peft_lora_config``).
        training_args: dtype / flash-attn knobs only; the dataset/optimiser
            fields of :class:`TrainingConfig` are not used here.
        cache_dir: optional HF cache directory.

    Returns:
        ``(model, processor, peft_config)`` — same tuple shape as the
        upstream helper. ``peft_config`` is ``None`` when LoRA is disabled.
    """
    # Resolve the torch dtype string to a real dtype.
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Build processor + (optional) special tokens.
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        padding_side="right",
        cache_dir=cache_dir,
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    # Build the reward model. Quantisation is intentionally not supported
    # here — the reward path expects full-precision (or bf16/fp16) weights.
    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=("flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"),
        cache_dir=cache_dir,
        revision=getattr(model_config, "model_revision", "main"),
    )
    # transformers 5.x forwards unknown from_pretrained kwargs to the model
    # ctor (4.x absorbed config fields like use_cache) — set it on the config.
    model.config.use_cache = bool(training_args.gradient_checkpointing)

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    # Optional LoRA wrapping — required to *load* a LoRA-split checkpoint.
    if peft_lora_config.lora_enable:
        # peft is an optional dep for the non-LoRA inference path; import
        # lazily so users who only ever load full-state-dict ckpts don't
        # need it installed.
        from peft import LoraConfig, get_peft_model

        namespan_exclude = list(peft_lora_config.lora_namespan_exclude or [])
        if isinstance(peft_lora_config.lora_namespan_exclude, str):
            namespan_exclude = [peft_lora_config.lora_namespan_exclude]
        # Mirror upstream: when vision_lora is off, exclude the visual tower.
        if not peft_lora_config.vision_lora and "visual" not in namespan_exclude:
            namespan_exclude.append("visual")

        target_modules = _find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=namespan_exclude,
        )
        peft_config = LoraConfig(
            target_modules=target_modules,
            r=peft_lora_config.lora_r,
            lora_alpha=peft_lora_config.lora_alpha,
            lora_dropout=peft_lora_config.lora_dropout,
            task_type=peft_lora_config.lora_task_type,
            use_rslora=peft_lora_config.use_rslora,
            bias="none",
            modules_to_save=peft_lora_config.lora_modules_to_save,
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


__all__ = ["create_model_and_processor"]
