"""PickScore reward scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend


class PickScoreRewardScorer(LocalRewardBackend):
    """PickScore image-text alignment reward."""

    canonical_model_name = "pickscore"

    def __init__(self, *, config: "PickScoreSpec", base_device: str) -> None:
        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
            processor_id=config.processor_id,
            model_id=config.model_id,
        )

    def _load_model(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError("transformers is required for PickScore")

        processor_path = self.model_kwargs.get("processor_id", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        model_path = self.model_kwargs.get("model_id", "yuvalkirstain/PickScore_v1")

        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)
        self.model = self.model.to(dtype=torch.float32)

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        all_rewards: List[float] = []

        def _extract_tensor(output):
            if isinstance(output, torch.Tensor):
                return output
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                return output.pooler_output
            if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                return output.last_hidden_state[:, 0]
            if isinstance(output, (tuple, list)):
                return output[0]
            raise TypeError(f"Unexpected output format: {type(output)}")

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]

            image_inputs = self.processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

            text_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

            with torch.no_grad():
                image_embs = self.model.get_image_features(**image_inputs)
                image_embs = _extract_tensor(image_embs)
                image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

                text_embs = self.model.get_text_features(**text_inputs)
                text_embs = _extract_tensor(text_embs)
                text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

                logit_scale = self.model.logit_scale.exp()
                scores = logit_scale * (text_embs @ image_embs.T)
                scores = scores.diag() / 26
                all_rewards.extend(scores.cpu().tolist())

        return all_rewards


@dataclass
class PickScoreSpec(BaseRewardComponentSpec):
    """Typed config for the PickScore reward component."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
