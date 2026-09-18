"""PickScore v1 scorer.

Mirrors flow_grpo/pickscore_scorer.py: PickScore_v1 CLIP model with its
H-14 processor, loaded from the same directory (or HF id) as the model
weights. Scales scores by 1/26 (matching flow_grpo) to bring them
roughly into 0-1 range. Single sub-metric `pickscore`.

Follows the same "weights_path (local) else model_name (HF fallback)"
convention as clip.py: if weights_path is given, both the model and its
CLIPProcessor are loaded from that directory; otherwise both fall back
to model_name via HF hub. The PickScore_v1 HF repo (and any local mirror
of it) ships the CLIP processor files — preprocessor_config.json,
tokenizer.json, vocab.json, merges.txt — alongside the weights, so no
separate processor path is needed.
"""

from __future__ import annotations

import torch

from reward_service.scorers._common import resolve_dtype, resolve_model_path, split_last_turn
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register


def _projected(out) -> torch.Tensor:
    """Projected CLIP embeddings across transformers versions.

    transformers < 5 returns the projected features tensor directly;
    >= 5 returns the full ``BaseModelOutputWithPooling`` with the projected
    features written back into ``pooler_output``.
    """
    if torch.is_tensor(out):
        return out
    return out.pooler_output


class PickScoreScorer(BaseScorer):
    name = "pickscore"
    sub_metric_names = ("pickscore",)
    supports_grad = True

    def __init__(
        self,
        model_name: str = "yuvalkirstain/PickScore_v1",
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

    def score_tensors(self, images: torch.Tensor, prompts: list[str]) -> torch.Tensor:
        """Differentiable PickScore over a [B,3,H,W] float batch in [0,1].

        The whole image path is torch-native (bicubic resize + CLIP
        normalization + CLIP forward) so gradients flow from the returned
        scores back to ``images`` — a PIL/numpy hop anywhere here would break
        the graph silently (the slow-processor lesson). Grad-or-not is the
        caller's context; weights stay frozen either way.
        """
        import torchvision.transforms.functional as TF

        ip = getattr(self.processor, "image_processor", None)
        mean = list(getattr(ip, "image_mean", None) or (0.48145466, 0.4578275, 0.40821073))
        std = list(getattr(ip, "image_std", None) or (0.26862954, 0.26130258, 0.27577711))
        size = int((getattr(ip, "crop_size", None) or {"height": 224})["height"])

        x = images.to(self.device)
        x = TF.resize(x, [size, size], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        x = (x - x.new_tensor(mean).view(1, 3, 1, 1)) / x.new_tensor(std).view(1, 3, 1, 1)
        x = x.to(dtype=next(self.model.parameters()).dtype)

        text_inputs = self.processor(
            text=list(prompts), padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = _projected(self.model.get_image_features(pixel_values=x))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = _projected(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        scores = self.model.logit_scale.exp() * (text_embs * image_embs).sum(dim=-1)
        return (scores / 26.0).float()

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []
        prompts, images = split_last_turn(items)

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}

        image_embs = _projected(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = _projected(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = (scores.diag() / 26.0).float().cpu().tolist()
        return [{"pickscore": float(s)} for s in scores]


register("pickscore", PickScoreScorer)
