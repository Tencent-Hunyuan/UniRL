"""EditReward scorer — multi-dimensional reward for instruction-guided image editing.

Wraps the EditRewardInferencer (Qwen2.5-VL-7B based) from the EditReward
package. Evaluates how well an edited image follows the editing instruction
(dim1: instruction following) and visual quality (dim2: visual quality).

Input convention:
    history[0] = (prompt, source_image)   — source image before editing
    history[1] = (prompt, edited_image)   — edited image after editing

The text prompt (editing instruction) is taken from history[0][0].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

if TYPE_CHECKING:
    from PIL import Image


class EditRewardScorer(BaseScorer):
    """Multi-head edit reward scorer backed by EditRewardInferencer."""

    name = "editreward"
    sub_metric_names = ("edit_following", "edit_quality")

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        rm_head_type: str = "ranknet_multi_head",
    ) -> None:
        import os

        from reward_service.scorers._editreward import EditRewardInferencer

        self._device = device if torch.cuda.is_available() else "cpu"
        self._rm_head_type = rm_head_type

        # If checkpoint_path looks like a HF repo ID (not a local dir),
        # download it via huggingface_hub first.
        if not os.path.isdir(checkpoint_path) and "/" in checkpoint_path:
            from huggingface_hub import snapshot_download
            checkpoint_path = snapshot_download(repo_id=checkpoint_path)

        self.inferencer = EditRewardInferencer(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=self._device,
            reward_dim="dim1",
            rm_head_type=rm_head_type,
        )

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        results: list[dict[str, float]] = []

        for item in items:
            try:
                result = self._score_single(item)
            except Exception:
                result = {k: float("nan") for k in self.sub_metric_names}
            results.append(result)

        return results

    def _score_single(self, item: ScoreItem) -> dict[str, float]:
        """Score a single item.

        Expects item.history to have at least 2 turns:
            history[0] = (prompt, source_image)
            history[1] = (prompt, edited_image)
        """
        if len(item.history) < 2:
            raise ValueError(
                f"EditReward requires 2 history turns (source + edited), got {len(item.history)}"
            )

        prompt, source_image = item.history[0]
        _, edited_image = item.history[1]

        if source_image is None or edited_image is None:
            raise ValueError("Both source and edited images must be provided")

        # EditRewardInferencer.reward() accepts paths or PIL images
        # (process_vision_info handles PIL.Image directly)
        rewards = self.inferencer.reward(
            prompts=[prompt],
            image_src=[source_image],
            image_paths=[edited_image],
        )

        # rewards shape depends on rm_head_type:
        # ranknet_multi_head: tensor of shape (batch, num_heads) or list of tensors
        if self._rm_head_type == "ranknet_multi_head":
            # Multi-head returns scores for both dims
            if isinstance(rewards, torch.Tensor):
                if rewards.dim() >= 2 and rewards.shape[-1] >= 2:
                    return {
                        "edit_following": float(rewards[0, 0].item()),
                        "edit_quality": float(rewards[0, 1].item()),
                    }
                else:
                    return {
                        "edit_following": float(rewards[0].item()),
                        "edit_quality": float("nan"),
                    }
            elif isinstance(rewards, (list, tuple)):
                scores = [float(r[0].item()) if torch.is_tensor(r) else float(r) for r in rewards]
                return {
                    "edit_following": scores[0] if len(scores) > 0 else float("nan"),
                    "edit_quality": scores[1] if len(scores) > 1 else float("nan"),
                }
        else:
            # Single-head: only one score
            if isinstance(rewards, torch.Tensor):
                val = float(rewards[0, 0].item()) if rewards.dim() >= 2 else float(rewards[0].item())
            else:
                val = float(rewards[0])
            return {
                "edit_following": val,
                "edit_quality": float("nan"),
            }

        # Fallback
        return {k: float("nan") for k in self.sub_metric_names}

    def close(self) -> None:
        if hasattr(self, "inferencer"):
            del self.inferencer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


register("editreward", EditRewardScorer)
