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

        # Differentiable image preprocessing (for compute_rewards_differentiable):
        # replicate the CLIP image processor's resize/center-crop/normalize on a
        # grad-carrying tensor (the PIL processor path detaches). Mirrors clip.py.
        import torch.nn as nn
        import torchvision.transforms as T

        def _get_size(size):
            if isinstance(size, int):
                return (size, size)
            if "height" in size and "width" in size:
                return (size["height"], size["width"])
            if "shortest_edge" in size:
                return size["shortest_edge"]
            raise ValueError(f"Invalid size: {size}")

        ip = self.processor.image_processor
        ip_cfg = ip.to_dict()
        resize = (
            T.Resize(_get_size(ip_cfg.get("size")), interpolation=T.InterpolationMode.BICUBIC, antialias=True)
            if ip_cfg.get("do_resize")
            else nn.Identity()
        )
        crop = T.CenterCrop(_get_size(ip_cfg.get("crop_size"))) if ip_cfg.get("do_center_crop") else nn.Identity()
        normalise = T.Normalize(mean=ip.image_mean, std=ip.image_std) if ip_cfg.get("do_normalize") else nn.Identity()
        self._clip_tform = T.Compose([resize, crop, normalise])

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

    def compute_rewards_differentiable(
        self,
        images_tensor: torch.Tensor,
        prompts: List[str],
        records=None,
    ) -> torch.Tensor:
        """Differentiable PickScore: image tensor ``[B, C, H, W]`` in ``[0, 1]``
        → ``[B]`` reward with ``grad_fn``. Reuses the frozen CLIP module; only the
        image path keeps grad (text + logit_scale are constants)."""

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

        if images_tensor.ndim != 4:
            raise ValueError(
                f"PickScore.compute_rewards_differentiable: expected [B, C, H, W], got {tuple(images_tensor.shape)}"
            )
        images_tensor = images_tensor.to(device=self.device, dtype=torch.float32)
        pixel_values = self._clip_tform(images_tensor)

        scores: List[torch.Tensor] = []
        for i in range(0, len(prompts), self.batch_size):
            px = pixel_values[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]

            text_inputs = self.processor(
                text=batch_prompts,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )
            text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

            # Image path keeps grad; text/logit_scale are constants.
            image_embs = _extract_tensor(self.model.get_image_features(pixel_values=px))
            image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
            with torch.no_grad():
                text_embs = _extract_tensor(self.model.get_text_features(**text_inputs))
                text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
                logit_scale = self.model.logit_scale.exp()

            # Per-pair alignment = diag(text @ image.T); avoid the BxB matmul.
            batch_scores = logit_scale * (text_embs * image_embs).sum(dim=-1) / 26
            scores.append(batch_scores)

        return torch.cat(scores, dim=0)


@dataclass
class PickScoreSpec(BaseRewardComponentSpec):
    """Typed config for the PickScore reward component."""

    batch_size: int = 8
    device: str = "auto"
    processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_id: str = "yuvalkirstain/PickScore_v1"
