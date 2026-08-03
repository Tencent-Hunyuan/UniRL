"""Differentiable forward over a loaded VideoAlign reward model.

Self-contained re-implementation — does NOT touch ``sys.path`` . The Qwen2-VL reward
backbone, its prompt template and the checkpoint loader all live under
:mod:`experimental.refl.reward.videoalign.model`.

Public API
----------
``VideoRewardWrapper(checkpoint_dir, device, ...) -> .forward_scores(...)``

The ``forward_scores`` signature is preserved verbatim from the mmrl
wrapper so the same callsite — ``self.model.forward_scores(per_sample_videos,
prompts, use_norm=...)`` — keeps returning ``{"VQ": Tensor[B], "MQ":
Tensor[B], "TA": Tensor[B], "Overall": Tensor[B]}`` with autograd-live
graphs when the input videos have ``grad_fn`` set.

Gradient flow
-------------
The Qwen2-VL vision encoder is fully differentiable w.r.t. its pixel input
through the *fast* (tensor-native) image processor, which transformers 5.6
loads by default — the slow PIL-based variant routes through ``numpy`` and
silently cuts the graph.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from typing import Dict, List, Optional

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from .model import (
    ModelConfig,
    PEFTLoraConfig,
    TrainingConfig,
    build_prompt,
    create_model_and_processor,
    load_model_from_checkpoint,
)

logger = logging.getLogger(__name__)


def _load_configs_from_json(config_path: str):
    """Parse ``model_config.json`` into the four typed configs.

    Drops trainer-only fields that may carry absolute filesystem paths
    (they would break on a different machine and are never read at
    inference).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    data_config = dict(config_dict["data_config"])
    data_config.pop("meta_data", None)
    data_config.pop("data_dir", None)

    return (
        data_config,
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict.get("inference_config", None),
    )


class VideoRewardWrapper:
    """Frozen VideoAlign reward model with a differentiable forward.

    Constructor responsibilities:

    1. Parse ``model_config.json`` into typed configs.
    2. Build the Qwen2-VL reward model + processor via
       :func:`create_model_and_processor`.
    3. Load weights from ``checkpoint-K`` via
       :func:`load_model_from_checkpoint`.
    4. Move to the requested device + dtype, ``eval()`` + freeze
       parameters (RL gradients flow *through* the activations, not into
       the reward weights).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        use_norm: bool = True,
        resize_height: int = 336,
        resize_width: int = 588,
        micro_batch_size: int = 1,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.use_norm = use_norm
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.micro_batch_size = max(1, int(micro_batch_size))

        config_path = os.path.join(checkpoint_dir, "model_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"VideoRewardWrapper: expected ``model_config.json`` at "
                f"{config_path!r}. Make sure ``reward_model_path`` points at "
                "the directory containing both ``model_config.json`` and "
                "one or more ``checkpoint-K`` subdirs (the layout produced "
                "by the upstream VideoAlign trainer)."
            )

        data_config_dict, model_config_dict, peft_lora_config_dict, inference_config = _load_configs_from_json(
            config_path
        )

        self.prompt_template_type = data_config_dict.get("prompt_template_type", "none")
        self.eval_dim = data_config_dict.get("eval_dim", "VQ")
        self.inference_config = inference_config
        self.build_prompt = build_prompt

        model_config = ModelConfig(**model_config_dict)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config_dict)
        training_args = TrainingConfig(
            load_from_pretrained=checkpoint_dir,
            load_from_pretrained_step=-1,
            gradient_checkpointing=False,
            disable_flash_attn2=True,
            bf16=(dtype == torch.bfloat16),
            fp16=(dtype == torch.float16),
            output_dir="",
        )

        model, processor, _ = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
        )
        model, _ = load_model_from_checkpoint(model, checkpoint_dir, -1)

        model.to(self.device)
        model.eval()
        model.requires_grad_(False)

        self.model = model
        self.processor = processor
        self.data_config = data_config_dict

        logger.info(
            "VideoRewardWrapper loaded: ckpt=%s device=%s dtype=%s resize=%dx%d micro_bs=%d use_norm=%s template=%s",
            checkpoint_dir,
            self.device,
            self.dtype,
            self.resize_height,
            self.resize_width,
            self.micro_batch_size,
            self.use_norm,
            self.prompt_template_type,
        )

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("VideoReward inputs must not be empty.")
        return inputs

    @staticmethod
    def _pixels_neg1_to_255(video: torch.Tensor) -> torch.Tensor:
        """[-1, 1] float → [0, 255] float — stays differentiable."""
        return ((video * 0.5 + 0.5).clamp(0, 1) * 255.0).float()

    def _resize_for_reward(self, video_tchw: torch.Tensor) -> torch.Tensor:
        return TF.resize(  # type: ignore[no-any-return]
            video_tchw,
            [self.resize_height, self.resize_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
    ):
        """Build a Qwen2-VL processor batch from ``[T,C,H,W]`` videos + prompts.

        Auto-transposes ``[C,T,H,W]`` callers when ``C==3``.
        """
        chat_data = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "file://dummy_path"},
                        {
                            "type": "text",
                            "text": self.build_prompt(
                                prompt,
                                self.eval_dim,
                                self.prompt_template_type,
                            ),
                        },
                    ],
                }
            ]
            for prompt in prompts
        ]

        processed: List[torch.Tensor] = []
        for video in video_tensors:
            if video.dim() != 4:
                raise ValueError(f"Expected video tensor shape [T,C,H,W], got {tuple(video.shape)}")
            if video.shape[0] == 3 and video.shape[1] > 3:
                video = video.permute(1, 0, 2, 3)
            video = self._pixels_neg1_to_255(video)
            video = self._resize_for_reward(video)
            processed.append(video)

        batch = self.processor(
            text=self.processor.apply_chat_template(
                chat_data,
                tokenize=False,
                add_generation_prompt=True,
            ),
            images=None,
            videos=processed,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        return self._prepare_inputs(batch)

    def _norm(
        self,
        vq: torch.Tensor,
        mq: torch.Tensor,
        ta: torch.Tensor,
    ):
        if self.inference_config is None:
            return vq, mq, ta
        vq = (vq - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
        mq = (mq - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
        ta = (ta - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
        return vq, mq, ta

    def forward_scores(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        use_norm: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Differentiable scoring → per-sample (VQ, MQ, TA, Overall) scalars.

        Args:
            video_tensors: list of ``(T, C, H, W)`` tensors, pixels in
                ``[-1, 1]``. May carry ``grad_fn`` (REFL path).
            prompts: per-sample user prompt text.
            use_norm: override ``self.use_norm`` for this call.

        Returns:
            ``{"VQ": Tensor[B], "MQ": Tensor[B], "TA": Tensor[B], "Overall": Tensor[B]}``.
            Gradients flow back into ``video_tensors`` when they require grad.
        """
        if len(video_tensors) != len(prompts):
            raise ValueError("video_tensors and prompts must have the same batch size.")
        use_norm = self.use_norm if use_norm is None else use_norm

        all_vq, all_mq, all_ta = [], [], []
        for start in range(0, len(video_tensors), self.micro_batch_size):
            end = start + self.micro_batch_size
            batch = self.prepare_batch_from_frames(video_tensors[start:end], prompts[start:end])
            batch.pop("mm_token_type_ids", None)
            logits = self.model(**batch, return_dict=True)["logits"]
            vq, mq, ta = logits[:, 0], logits[:, 1], logits[:, 2]
            if use_norm:
                vq, mq, ta = self._norm(vq, mq, ta)
            all_vq.append(vq)
            all_mq.append(mq)
            all_ta.append(ta)

        vq = torch.cat(all_vq, dim=0)
        mq = torch.cat(all_mq, dim=0)
        ta = torch.cat(all_ta, dim=0)
        return {"VQ": vq, "MQ": mq, "TA": ta, "Overall": vq + mq + ta}


__all__ = ["VideoRewardWrapper"]
