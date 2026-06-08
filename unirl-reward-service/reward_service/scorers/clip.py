"""CLIP cosine-similarity scorer.

Based on flow_grpo/clip_scorer.py but adapted to BaseScorer batch shape.
Returns a single sub-metric `clip`, the diagonal of logits_per_image/30
(the /30 is kept identical to flow_grpo for comparability).
"""

from __future__ import annotations

import torch

from reward_service.scorers._common import resolve_dtype, resolve_model_path, split_last_turn
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register


class ClipScorer(BaseScorer):
    name = "clip"
    sub_metric_names = ("clip",)

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        weights_path: str | None = None,
        dtype: str = "float32",
        device: str = "cuda",
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        torch_dtype = resolve_dtype(dtype)
        path = resolve_model_path(model_name, weights_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(path).to(self.device, dtype=torch_dtype).eval()
        self.model.requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(path)

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []
        prompts, images = split_last_turn(items)

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=77,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        scores = (outputs.logits_per_image.diagonal() / 30.0).float().cpu().tolist()
        return [{"clip": float(s)} for s in scores]


register("clip", ClipScorer)
