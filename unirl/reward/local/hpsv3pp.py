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
    """HPSv3++ capability- and RL-iteration-aware reward (Qwen3-VL-8B + FiLM).

    HPSv3++ (``model_type: film_hybrid``) scores (prompt, image) pairs with a
    Qwen3-VL-8B-Instruct backbone, a Capability Encoder (model capability is
    inferred from the image, not passed in), a FiLM head fed an explicit
    RL-iteration scalar, and a 3-layer RankNet head that outputs ``[mu, sigma]``
    per image. We use ``mu`` (index 0) as the scalar reward.

    Unlike the original HPSv3 (PyPI ``hpsv3``, Qwen2-VL-7B), v3++ is run from
    the source repo (no PyPI package). Point ``repo_path`` at a checkout of
    https://github.com/PlantPotatoOnMoon/HPSv3-PlusPlus (prepended to
    ``sys.path`` so its ``hpsv3`` package wins over any PyPI ``hpsv3``),
    ``config_path`` at its ``hpsv3/config/train_stage2.yaml`` (the film_hybrid
    config; defaults to ``<repo_path>/hpsv3/config/train_stage2.yaml``), and
    ``checkpoint_path`` at ``hpsv3++.pth`` (empty auto-downloads
    ``Junjun2333/HPSv3-PlusPlus``).

    ``rl_iteration`` is the explicit RL-iteration condition, a normalized scalar
    in ``[0, 1]``. Use ``0.0`` (the default) for preference scoring / ranking.
    As the reward inside T2I RL fine-tuning the paper ramps it linearly from
    0.3 -> 1.0 over training; this scorer applies a fixed value, so set it to
    the desired constant (per-step ramping would need training-step plumbing
    into the reward backend).

    Reference: HPSv3++: Scaling Reward Models Across the Full Spectrum of
    Diffusion Model Capabilities.
    """

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
                # reward() infers capability from the image; iter_step is the
                # explicit RL-iteration condition. Returns [B, 2] (mu, sigma).
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
    """Typed config for the HPSv3++ reward component.

    HPSv3++ runs from the HPSv3-PlusPlus source repo (no PyPI package), so set
    ``repo_path`` to that checkout. ``config_path`` defaults to
    ``<repo_path>/hpsv3/config/train_stage2.yaml``; empty ``checkpoint_path``
    auto-downloads ``Junjun2333/HPSv3-PlusPlus/hpsv3++.pth``.

    ``score_scale`` is a *divisor* applied to the raw mu (which is ~7-11): the
    reward is ``mu / score_scale``. Use ~15.0 to land it in ~0-1 (parity with
    the ``hpsv3`` scorer); leave at 1.0 only if downstream wants the raw scale.
    """

    batch_size: int = 8
    device: str = "auto"
    repo_path: str = ""
    config_path: str = ""
    checkpoint_path: str = ""
    rl_iteration: float = 0.0
    score_scale: float = 1.0
