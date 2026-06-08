"""Build ``(model, processor, peft_config)`` for VideoReward inference.

Vendored from upstream ``train_reward.create_model_and_processor`` with:

* The TRL helpers ``get_kbit_device_map`` / ``get_quantization_config``
  replaced by inline equivalents (we only need them for the
  ``load_in_8bit`` / ``load_in_4bit`` paths and we'd rather not depend
  on TRL at inference time).
* The training-only side effects (gradient_checkpointing → use_cache
  toggle, dtype casting via ``training_args.bf16/fp16``) preserved so a
  checkpoint trained with these flags loads identically.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers import AutoProcessor

from reward_service.scorers._videoalign.configs import (
    MinimalTrainingArgs,
    ModelConfig,
    PEFTLoraConfig,
)
from reward_service.scorers._videoalign.model import Qwen2VLRewardModelBT


def _resolve_torch_dtype(name):
    if name in (None, "auto"):
        return name
    return getattr(torch, name)


def _build_quantization_config(model_config: ModelConfig):
    """Return a ``BitsAndBytesConfig`` if 4/8-bit loading was requested,
    else ``None``. Mirrors what TRL's ``get_quantization_config`` did
    upstream — re-implemented here to keep TRL out of the inference
    dependency closure.
    """
    if not (model_config.load_in_8bit or model_config.load_in_4bit):
        return None

    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_8bit=model_config.load_in_8bit,
        load_in_4bit=model_config.load_in_4bit,
        bnb_4bit_quant_type=model_config.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=model_config.use_bnb_nested_quant,
    )


def _kbit_device_map():
    """When loading quantized weights, ``device_map="auto"`` lets HF
    accelerate place them on the visible CUDA devices."""
    if torch.cuda.is_available():
        return {"": torch.cuda.current_device()}
    return None


def create_model_and_processor(
    model_config: ModelConfig,
    peft_lora_config: PEFTLoraConfig,
    training_args: MinimalTrainingArgs,
    cache_dir: Optional[str] = None,
) -> Tuple[Qwen2VLRewardModelBT, object, object | None]:
    """Build the reward model + processor + (optional) PEFT config.

    The order of operations matches upstream so a checkpoint trained on
    one of the public configs lands at exactly the same weight tensors
    after loading.
    """
    torch_dtype = _resolve_torch_dtype(model_config.torch_dtype)
    quantization_config = _build_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        # When gradient checkpointing was on at training time, use_cache must
        # be off to avoid silent recompute mismatches. Inference always wants
        # use_cache=True for KV caching.
        use_cache=False if training_args.gradient_checkpointing else True,
    )

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

    attn_implementation = "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
    model = Qwen2VLRewardModelBT.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        cache_dir=cache_dir,
        **model_kwargs,
    )
    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    peft_config = None
    if peft_lora_config.lora_enable:
        # LoRA branch: wrap the base model with PEFT so the loaded
        # adapter weights find their target modules.
        from peft import LoraConfig, get_peft_model

        target_modules = _find_target_linear_names(
            model,
            num_lora_modules=peft_lora_config.num_lora_modules,
            lora_namespan_exclude=peft_lora_config.lora_namespan_exclude or [],
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

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


def _find_target_linear_names(model, num_lora_modules: int = -1, lora_namespan_exclude=None):
    """Return module-name list of every ``nn.Linear`` / ``nn.Embedding``
    not in ``lora_namespan_exclude`` (mirrors upstream)."""
    if lora_namespan_exclude is None:
        lora_namespan_exclude = []
    linear_cls = torch.nn.Linear
    embedding_cls = torch.nn.Embedding
    names: list[str] = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            names.append(name)
    if num_lora_modules > 0:
        names = names[-num_lora_modules:]
    return names
