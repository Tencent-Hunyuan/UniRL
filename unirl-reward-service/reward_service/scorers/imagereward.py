"""ImageReward scorer backed by the official `image-reward` pip package.

The official API only exposes `inference_rank(prompt, [image])` which
returns (ranking, scores); we iterate per-item since the underlying
model does not batch across different prompts.

Weights layout (local checkpoint): `RM.load(name)` accepts either
  - a model name like "ImageReward-v1.0" (auto-download from HF), or
  - a direct path to the `.pt` file.

When loading from a local directory that contains both `ImageReward.pt`
and `med_config.json`, pass `weights_path=<dir>` and the scorer will
resolve both files; alternatively pass the `.pt` file directly with
`med_config_path` pointing at `med_config.json`.
"""

from __future__ import annotations

from pathlib import Path

import torch

from reward_service.scorers._common import resolve_dtype
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register


def _resolve_checkpoint_and_config(
    model_name: str,
    weights_path: str | None,
    med_config_path: str | None,
) -> tuple[str, str | None]:
    """Return (name_or_file, med_config_path) suitable for RM.load().

    - If weights_path is a directory containing ImageReward.pt + med_config.json,
      autofill both from it.
    - If weights_path is a .pt file, use it directly; med_config stays as given.
    - If weights_path is None, fall through to model_name (HF auto-download).
    """
    if weights_path is None:
        return model_name, med_config_path

    p = Path(weights_path)
    if p.is_dir():
        pt = p / "ImageReward.pt"
        cfg = p / "med_config.json"
        if not pt.exists():
            raise FileNotFoundError(f"{pt} not found under {p}")
        resolved_cfg = str(cfg) if cfg.exists() else med_config_path
        return str(pt), resolved_cfg
    return str(p), med_config_path


class ImageRewardScorer(BaseScorer):
    name = "imagereward"
    sub_metric_names = ("imagereward",)

    def __init__(
        self,
        model_name: str = "ImageReward-v1.0",
        weights_path: str | None = None,
        med_config_path: str | None = None,
        dtype: str = "float32",
        device: str = "cuda",
    ) -> None:
        import ImageReward as RM

        torch_dtype = resolve_dtype(dtype)
        self.device = device if torch.cuda.is_available() else "cpu"
        ckpt, med_cfg = _resolve_checkpoint_and_config(model_name, weights_path, med_config_path)
        load_kwargs: dict = {"device": self.device}
        if med_cfg is not None:
            load_kwargs["med_config"] = med_cfg
        self.model = RM.load(ckpt, **load_kwargs).eval().to(dtype=torch_dtype)
        self.model.requires_grad_(False)

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []
        out: list[dict[str, float]] = []
        for item in items:
            text, image = item.history[-1]
            _, reward = self.model.inference_rank(text, [image])
            value = reward[0] if isinstance(reward, (list, tuple)) else reward
            out.append({"imagereward": float(value)})
        return out


register("imagereward", ImageRewardScorer)
