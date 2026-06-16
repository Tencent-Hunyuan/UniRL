"""EditReward inferencer — inference-only, no training dependencies.

Vendored from EditReward/inference_edit.py with training imports removed.
Uses the vendored configs, model, prompts, and vision_process modules.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import torch
import safetensors

from reward_service.scorers._editreward.vision_process import process_vision_info
from reward_service.scorers._editreward.prompts import (
    INSTRUCTION_EDIT_FOLLOWING,
    INSTRUCTION_EDIT_QUALITY,
    INSTRUCTION_EDIT_OVERALL,
    INSTRUCTION_EDIT_OVERALL_DETAILED,
    PROMPT_WITH_SPECIAL_TOKEN,
    PROMPT_WITHOUT_SPECIAL_TOKEN,
)
from reward_service.scorers._editreward.configs import (
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    DataConfig,
    parse_args_with_yaml,
)

_MODEL_CONFIG_PATH = Path(__file__).parent / "config"
_DEFAULT_CONFIG_FILENAME = "EditReward-Qwen2.5-7B-VL.yaml"


def _resolve_config_path(config_path):
    if config_path is None:
        return str(_MODEL_CONFIG_PATH / _DEFAULT_CONFIG_FILENAME)

    candidate_paths = [
        Path(config_path),
        _MODEL_CONFIG_PATH.parent / config_path,
        _MODEL_CONFIG_PATH / Path(config_path).name,
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return str(candidate)

    raise FileNotFoundError(
        f"Config not found: {config_path}. "
        f"Checked {[str(path) for path in candidate_paths]}"
    )


def _create_model_and_processor(model_config, peft_lora_config, training_args, differentiable=False):
    """Create model and processor for inference (no LoRA, no quantization)."""
    from transformers import AutoProcessor
    from reward_service.scorers._editreward.model import Qwen2_5_VLRewardModelBT_MultiHead

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path, padding_side="right"
    )

    special_token_ids = None
    if model_config.use_special_tokens:
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

    try:
        import flash_attn
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    if training_args.disable_flash_attn2:
        attn_impl = "sdpa"

    model = Qwen2_5_VLRewardModelBT_MultiHead.from_pretrained(
        model_config.model_name_or_path,
        output_dim=model_config.output_dim,
        reward_token=model_config.reward_token,
        special_token_ids=special_token_ids,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        rm_head_type=model_config.rm_head_type,
        rm_head_kwargs=model_config.rm_head_kwargs,
        pooling_strategy=model_config.pooling_strategy,
    )

    if model_config.use_special_tokens:
        model.resize_token_embeddings(len(processor.tokenizer))

    if training_args.bf16:
        model.to(torch.bfloat16)
    if training_args.fp16:
        model.to(torch.float16)

    if model.rm_head_type in ("ranknet_multi_head", "ranknet_multi_head_regression"):
        for h in model.rm_heads.values():
            h.to(torch.float32)
    elif model.rm_head is not None:
        model.rm_head.to(torch.float32)

    # No LoRA for inference
    peft_config = None

    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    return model, processor, peft_config


class EditRewardInferencer:
    def __init__(
        self,
        config_path=None,
        checkpoint_path=None,
        device="cuda",
        differentiable=False,
        reward_dim="dim1",
        rm_head_type="ranknet_multi_head",
    ):
        config_path = _resolve_config_path(config_path)

        (data_config, training_args, model_config, peft_lora_config), config_path = (
            parse_args_with_yaml(
                (DataConfig, TrainingConfig, ModelConfig, PEFTLoraConfig),
                config_path,
                is_train=False,
            )
        )
        training_args.output_dir = os.path.join(
            training_args.output_dir, config_path.split("/")[-1].split(".")[0]
        )

        model, processor, _ = _create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            differentiable=differentiable,
        )

        self.device = device
        self.use_special_tokens = model_config.use_special_tokens
        self.reward_dim = reward_dim
        self.rm_head_type = rm_head_type

        # Load checkpoint
        full_ckpt = os.path.join(checkpoint_path, "model.pth")
        full_ckpt_safetensors = os.path.join(checkpoint_path, "model.safetensors")

        if os.path.exists(full_ckpt):
            state_dict = torch.load(full_ckpt, map_location="cpu")
        elif os.path.exists(full_ckpt_safetensors):
            import safetensors.torch
            state_dict = safetensors.torch.load_file(full_ckpt_safetensors, device="cpu")
        else:
            raise ValueError(f"Checkpoint not found at {checkpoint_path}")

        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        self.model = model
        self.processor = processor
        self.model.to(self.device)
        self.data_config = data_config

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("Empty inputs")
        return inputs

    def prepare_batch(self, image_src, image_paths, prompts):
        max_pixels = 256 * 28 * 28
        min_pixels = 256 * 28 * 28

        def _build_messages(prompts, image_src, image_paths, reward_dim):
            message_list = []
            for text, src, img in zip(prompts, image_src, image_paths):
                if reward_dim == "dim1":
                    base_prompt = INSTRUCTION_EDIT_FOLLOWING.format(text_prompt=text)
                elif reward_dim == "dim2":
                    base_prompt = INSTRUCTION_EDIT_QUALITY.format(text_prompt=text)
                elif reward_dim == "overall":
                    base_prompt = INSTRUCTION_EDIT_OVERALL.format(text_prompt=text)
                elif reward_dim == "overall_detail":
                    base_prompt = INSTRUCTION_EDIT_OVERALL_DETAILED.format(text_prompt=text)
                else:
                    raise ValueError(f"Unknown reward_dim: {reward_dim}")

                final_text = (
                    base_prompt + PROMPT_WITH_SPECIAL_TOKEN
                    if self.use_special_tokens
                    else base_prompt + PROMPT_WITHOUT_SPECIAL_TOKEN
                )

                out_message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": src, "min_pixels": min_pixels, "max_pixels": max_pixels},
                            {"type": "image", "image": img, "min_pixels": max_pixels, "max_pixels": max_pixels},
                            {"type": "text", "text": final_text},
                        ],
                    }
                ]
                message_list.append(out_message)
            return message_list

        def _build_batch(prompts, image_src, image_paths, reward_dim):
            messages = _build_messages(prompts, image_src, image_paths, reward_dim)
            image_inputs, _ = process_vision_info(messages)
            batch = self.processor(
                text=self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ),
                images=image_inputs,
                padding=True,
                return_tensors="pt",
                videos_kwargs={"do_rescale": True},
            )
            batch = self._prepare_inputs(batch)
            return batch, image_inputs

        if self.rm_head_type == "ranknet_multi_head":
            batch_dim1, image_inputs_1 = _build_batch(prompts, image_src, image_paths, reward_dim="dim1")
            batch_dim2, image_inputs_2 = _build_batch(prompts, image_src, image_paths, reward_dim="dim2")
            return {
                "batch_dim1": batch_dim1,
                "batch_dim2": batch_dim2,
                "image_src": image_src,
                "image_paths": image_paths,
                "prompts": prompts,
                "image_inputs_dim1": image_inputs_1,
                "image_inputs_dim2": image_inputs_2,
            }
        else:
            batch, image_inputs = _build_batch(prompts, image_src, image_paths, reward_dim=self.reward_dim)
            return {
                "batch": batch,
                "image_src": image_src,
                "image_paths": image_paths,
                "prompts": prompts,
                "image_inputs": image_inputs,
            }

    def reward(self, prompts, image_src, image_paths):
        batch = self.prepare_batch(image_src, image_paths, prompts)
        rewards = self.model(return_dict=True, **batch)["logits"]
        return rewards
