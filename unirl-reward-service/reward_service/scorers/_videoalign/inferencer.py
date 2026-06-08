"""High-level VideoReward inference facade.

Vendored from upstream ``inference.VideoVLMRewardInference`` (renamed
``VideoRewardInferencer`` to disambiguate from huggingface inference
APIs). Loads the model + processor, prepares chat batches from
(video_path, prompt) pairs, runs the forward pass, and unpacks logits
into per-dimension reward dicts with optional normalization.

Stripped from upstream:

* ``import pdb`` and dead-code branches.
* `pandas`/`json` imports unused by the inference path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Iterable, List

import torch

from reward_service.scorers._videoalign.builder import create_model_and_processor
from reward_service.scorers._videoalign.checkpoint import load_model_from_checkpoint
from reward_service.scorers._videoalign.configs import (
    DataConfig,
    MinimalTrainingArgs,
    ModelConfig,
    PEFTLoraConfig,
    load_configs_from_json,
)
from reward_service.scorers._videoalign.prompt_template import build_prompt
from reward_service.scorers._videoalign.vision_process import process_vision_info


class VideoRewardInferencer:
    """Inference-only counterpart of upstream ``VideoVLMRewardInference``.

    Args:
        load_from_pretrained: Directory holding ``model_config.json`` plus
            one or more ``checkpoint-<step>/`` subfolders.
        load_from_pretrained_step: ``None`` / ``-1`` ⇒ latest step;
            otherwise the exact step to load.
        device: Torch device string (``"cuda"`` / ``"cuda:0"`` / ``"cpu"``).
        dtype: ``torch.bfloat16`` / ``torch.float16`` / ``torch.float32``.
            Drives the ``bf16`` / ``fp16`` flags on the minimal training
            args so weights end up cast identically to upstream.
        disable_flash_attn2: When True, force ``attn_implementation="sdpa"``;
            useful on machines without flash-attn or when the venv's
            flash-attn build mismatches the active torch ABI.
    """

    def __init__(
        self,
        load_from_pretrained: str,
        load_from_pretrained_step: int | None = -1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        disable_flash_attn2: bool = False,
    ) -> None:
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_dict, _, model_dict, peft_dict, inference_config = load_configs_from_json(config_path)

        data_config = DataConfig(**data_dict)
        model_config = ModelConfig(**model_dict)
        peft_lora_config = PEFTLoraConfig(**peft_dict)

        training_args = MinimalTrainingArgs(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=disable_flash_attn2,
            bf16=(dtype == torch.bfloat16),
            fp16=(dtype == torch.float16),
            output_dir="",
        )

        model, processor, _peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )

        model, _step = load_model_from_checkpoint(
            model, load_from_pretrained, load_from_pretrained_step
        )
        model.eval()
        model.to(device)

        self.model = model
        self.processor = processor
        self.device = device
        self.data_config = data_config
        self.inference_config = inference_config

    def _norm(self, reward: dict) -> dict:
        """Apply per-dim mean/std rescaling if the checkpoint shipped with
        an ``inference_config`` block. Otherwise return as-is."""
        if self.inference_config is None:
            return reward
        for dim in ("VQ", "MQ", "TA"):
            mean = self.inference_config[f"{dim}_mean"]
            std = self.inference_config[f"{dim}_std"]
            reward[dim] = (reward[dim] - mean) / std
        return reward

    def _to_device(self, data):
        """Recursively move tensors in ``data`` to ``self.device``."""
        if isinstance(data, Mapping):
            return type(data)({k: self._to_device(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._to_device(v) for v in data)
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _build_chat_data(
        self,
        video_paths: List[str],
        prompts: List[str],
        fps: float | None,
        num_frames: int | None,
        max_pixels: int | None,
    ) -> list[list[dict]]:
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels
        sample_type = self.data_config.sample_type
        eval_dim = self.data_config.eval_dim
        prompt_template_type = self.data_config.prompt_template_type

        chat_data: list[list[dict]] = []
        for video_path, prompt in zip(video_paths, prompts):
            video_block: dict[str, Any] = {
                "type": "video",
                "video": f"file://{video_path}",
                "max_pixels": max_pixels,
                "sample_type": sample_type,
            }
            if num_frames is None:
                video_block["fps"] = fps
            else:
                video_block["nframes"] = num_frames
            chat_data.append(
                [
                    {
                        "role": "user",
                        "content": [
                            video_block,
                            {
                                "type": "text",
                                "text": build_prompt(prompt, eval_dim, prompt_template_type),
                            },
                        ],
                    }
                ]
            )
        return chat_data

    def prepare_batch(
        self,
        video_paths: List[str],
        prompts: List[str],
        fps: float | None = None,
        num_frames: int | None = None,
        max_pixels: int | None = None,
    ):
        """Produce a model-ready batch from parallel ``video_paths`` /
        ``prompts`` lists.

        ``fps`` and ``num_frames`` are mutually exclusive — passing
        ``num_frames`` overrides ``fps``. Both default to the values
        captured in the checkpoint's ``DataConfig``.
        """
        if fps is None and num_frames is None:
            fps = self.data_config.fps
            num_frames = self.data_config.num_frames

        chat_data = self._build_chat_data(video_paths, prompts, fps, num_frames, max_pixels)
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        return self._to_device(batch)

    def reward(
        self,
        video_paths: Iterable[str],
        prompts: Iterable[str],
        fps: float | None = None,
        num_frames: int | None = None,
        max_pixels: int | None = None,
        use_norm: bool = True,
    ) -> List[dict[str, float]]:
        """Score each (video_path, prompt) pair.

        Returns one ``{"VQ", "MQ", "TA", "Overall"}`` dict per sample.
        ``Overall`` is the unnormalized sum VQ+MQ+TA (matches upstream).

        Caller must guarantee:

        * ``fps`` XOR ``num_frames`` is set (or both ``None``, in which
          case the checkpoint's defaults apply).
        * ``video_paths[i]`` exists on the local filesystem (this is the
          decord / torchvision contract).
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`fps` and `num_frames` cannot be set at the same time.")

        video_paths = list(video_paths)
        prompts = list(prompts)
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        outputs = self.model(return_dict=True, **batch)
        logits = outputs["logits"]

        rewards: list[dict[str, float]] = []
        for row in logits:
            entry = {
                "VQ": row[0].item(),
                "MQ": row[1].item(),
                "TA": row[2].item(),
            }
            if use_norm:
                entry = self._norm(entry)
            entry["Overall"] = entry["VQ"] + entry["MQ"] + entry["TA"]
            rewards.append(entry)
        return rewards
