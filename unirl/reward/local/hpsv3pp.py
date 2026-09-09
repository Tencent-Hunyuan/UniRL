"""HPSv3++ reward scorer."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend


class HPSv3PPRewardScorer(LocalRewardBackend):
    """HPSv3++ reward (Qwen3-VL-8B + FiLM) — the RankNet head outputs ``[mu, sigma]``."""

    canonical_model_name = "hpsv3pp"

    def __init__(self, *, config: "HPSv3PPSpec", base_device: str) -> None:
        self.rl_iteration = float(config.rl_iteration)
        self.score_scale = float(config.score_scale)
        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
            repo_path=config.repo_path,
            config_path=config.config_path,
            checkpoint_path=config.checkpoint_path,
        )

    def _load_model(self) -> None:
        repo_path = str(self.model_kwargs.get("repo_path") or "").strip()
        if repo_path:
            repo_path = os.path.abspath(os.path.expanduser(repo_path))
            if repo_path not in sys.path:
                sys.path.insert(0, repo_path)

        try:
            from hpsv3.inference import HPSv3RewardInferencer
        except ImportError as exc:
            raise ImportError(
                "HPSv3++ requires the HPSv3-PlusPlus source repo on the import path. "
                "Clone https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus, run "
                "`pip install -r requirements.txt` (transformers==4.57.0 for Qwen3-VL), "
                "and set the reward Spec's `repo_path` to that checkout (or add it to PYTHONPATH)."
            ) from exc

        config_path = str(self.model_kwargs.get("config_path") or "").strip()
        if not config_path and repo_path:
            config_path = os.path.join(repo_path, "hpsv3", "config", "train_stage2.yaml")
        if not config_path or not os.path.isfile(config_path):
            raise FileNotFoundError(
                "HPSv3++ needs the film_hybrid YAML (hpsv3/config/train_stage2.yaml). "
                f"Set the reward Spec's `config_path` (resolved to {config_path!r})."
            )

        checkpoint_path = str(self.model_kwargs.get("checkpoint_path") or "").strip()
        if not checkpoint_path:
            import huggingface_hub

            checkpoint_path = huggingface_hub.hf_hub_download(
                "Junjun2333/HPSv3-PlusPlus", "hpsv3++.pth", repo_type="model"
            )

        self._hpsv3pp_inferencer = HPSv3RewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self.device,
        )
        self.model = self._hpsv3pp_inferencer.model

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        all_rewards: List[float] = []

        for i in range(0, len(images), self.batch_size):
            batch_images = images[i : i + self.batch_size]
            batch_prompts = prompts[i : i + self.batch_size]

            pil_images = [
                img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(img).convert("RGB")
                for img in batch_images
            ]

            with torch.no_grad():
                rewards = self._hpsv3pp_inferencer.reward(
                    prompts=batch_prompts,
                    image_paths=pil_images,
                    iter_step=self.rl_iteration,
                )
                if not torch.is_tensor(rewards):
                    rewards = torch.as_tensor(rewards)
                mu = rewards[:, 0] if rewards.ndim == 2 else rewards
                if self.score_scale != 1.0:
                    mu = mu / self.score_scale
                all_rewards.extend(mu.float().cpu().tolist())

        return all_rewards


@dataclass
class HPSv3PPSpec(BaseRewardComponentSpec):
    """Typed config for the HPSv3++ reward component."""

    batch_size: int = 8
    device: str = "auto"
    repo_path: str = ""
    config_path: str = ""
    checkpoint_path: str = ""
    rl_iteration: float = 0.0
    score_scale: float = 1.0
