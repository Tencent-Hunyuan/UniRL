"""HPSv2 reward scorer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from PIL import Image

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend


class HPSv2RewardScorer(LocalRewardBackend):
    """HPSv2 image-text alignment reward."""

    canonical_model_name = "hpsv2"

    def __init__(self, *, config: "HPSv2Spec", base_device: str) -> None:
        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
            open_clip_path=config.open_clip_path,
            checkpoint_path=config.checkpoint_path,
        )

    def _load_model(self) -> None:
        try:
            from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        except ImportError:
            raise ImportError("hpsv2 is required for HPSv2 reward")

        open_clip_path = self.model_kwargs["open_clip_path"]
        checkpoint_path = self.model_kwargs["checkpoint_path"]

        model, _, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            open_clip_path,
            precision="amp",
            device=self.device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            image_mean=None,
            image_std=None,
            light_augmentation=True,
            aug_cfg={},
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["state_dict"])
        self._hpsv2_tokenizer = get_tokenizer("ViT-H-14")
        self._hpsv2_preprocess_val = preprocess_val
        self.model = model.to(self.device)
        self.model.eval()

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        if images is None:
            raise ValueError("HPSv2 requires generated images.")
        if len(images) != len(prompts):
            raise ValueError(
                f"HPSv2 requires one prompt per image; got {len(prompts)} prompts for {len(images)} images."
            )
        all_rewards: List[float] = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]
            try:
                pil_images = [
                    image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(image).convert("RGB")
                    for image in batch_images
                ]
                image_input = torch.stack(
                    [self._hpsv2_preprocess_val(image) for image in pil_images],
                    dim=0,
                ).to(device=self.device, non_blocking=True)
                text_input = self._hpsv2_tokenizer(list(batch_prompts)).to(
                    device=self.device,
                    non_blocking=True,
                )

                device_type = torch.device(self.device).type
                with (
                    torch.no_grad(),
                    torch.amp.autocast(
                        device_type,
                        enabled=device_type == "cuda",
                    ),
                ):
                    outputs = self.model(image_input, text_input)
                    image_features = outputs["image_features"]
                    text_features = outputs["text_features"]
                    hps_scores = torch.diagonal(image_features @ text_features.T)
                all_rewards.extend(float(value) for value in hps_scores.float().cpu().tolist())
            except Exception as exc:
                raise RuntimeError(
                    f"HPSv2 reward scoring failed for batch rows [{i}:{i + len(batch_images)}]."
                ) from exc

        return all_rewards


@dataclass
class HPSv2Spec(BaseRewardComponentSpec):
    """Typed config for the HPSv2 reward component."""

    batch_size: int = 8
    device: str = "auto"
    open_clip_path: str = "./hps_ckpt/open_clip_pytorch_model.bin"
    checkpoint_path: str = "./hps_ckpt/HPS_v2.1_compressed.pt"

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"HPSv2Spec.batch_size must be a positive integer, got {self.batch_size!r}.")
        for name in ("device", "open_clip_path", "checkpoint_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"HPSv2Spec.{name} must be a non-empty string, got {value!r}.")
