"""HPSv2 / HPSv2.1 scorer backed by the official `hpsv2` pip package.

Model loading (architecture + checkpoint + tokenizer) is done once in
``__init__``; ``score()`` reuses the cached model for inference only.

The official ``hpsv2.img_score.score()`` reloads the checkpoint from disk
on every call — we avoid this by performing the load once and keeping the
model in GPU memory.

Inference logic mirrors the official ``img_score.score()`` per-item loop
exactly: ``preprocess_val(PIL.Image)`` → ``unsqueeze(0)`` → ``to(device)``
→ ``model(image, text)`` → ``features @ features.T`` → ``diagonal()[0]``.
This guarantees numerical equivalence with the upstream implementation.

Local checkpoint layout (matches the official HF release):
  <weights_dir>/HPS_v2.1_compressed.pt
  <weights_dir>/HPS_v2_compressed.pt
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

from reward_service.scorers._common import split_last_turn
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register


_CHECKPOINT_FILENAMES = {
    "v2.0": "HPS_v2_compressed.pt",
    "v2.1": "HPS_v2.1_compressed.pt",
}


class HPSv2Scorer(BaseScorer):
    name = "hpsv2"
    sub_metric_names = ("hpsv2",)

    def __init__(
        self,
        hps_version: str = "v2.1",
        weights_path: str | None = None,
        device: str = "cuda",
    ) -> None:
        """Initialise model, tokenizer and preprocessing — loaded once.

        Args:
            hps_version: ``"v2.0"`` or ``"v2.1"``.
            weights_path: Directory containing the checkpoint ``.pt`` file,
                or a direct path to the checkpoint. When *None*, falls back
                to the default ``HPS_ROOT`` / HF download.
            device: Torch device string (``"cuda"`` / ``"cpu"``).
        """
        if hps_version not in _CHECKPOINT_FILENAMES:
            raise ValueError(
                f"unknown hps_version: {hps_version!r}. "
                f"expected one of {list(_CHECKPOINT_FILENAMES)}"
            )

        self.hps_version = hps_version
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        checkpoint_path = self._resolve_checkpoint_path(weights_path, hps_version)

        # Populate HPS_ROOT so any auxiliary assets (benchmark prompts) hpsv2
        # tries to load land in the same directory rather than ~/.cache.
        if weights_path:
            os.environ.setdefault("HPS_ROOT", weights_path)

        # ── Build model architecture (no pretrained weights yet) ──
        from hpsv2.img_score import create_model_and_transforms, get_tokenizer

        model, _preprocess_train, preprocess_val = create_model_and_transforms(
            "ViT-H-14",
            pretrained=None,
            precision="amp",
            device=str(self._device),
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

        # ── Load checkpoint weights ──
        if checkpoint_path is None:
            # Fallback: let hpsv2 download via huggingface_hub (online only).
            import huggingface_hub
            from hpsv2.utils import hps_version_map

            checkpoint_path = huggingface_hub.hf_hub_download(
                "xswu/HPSv2", hps_version_map[hps_version]
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])

        model = model.to(self._device)
        model.eval()
        model.requires_grad_(False)

        self._model = model
        self._preprocess_val = preprocess_val
        self._tokenizer = get_tokenizer("ViT-H-14")

    @staticmethod
    def _resolve_checkpoint_path(weights_path: str | None, hps_version: str) -> str | None:
        """Resolve the checkpoint ``.pt`` file from *weights_path*.

        Returns *None* when *weights_path* is not given, signalling the
        caller should fall back to online download.
        """
        if weights_path is None:
            return None
        p = Path(weights_path)
        if p.is_file():
            return str(p)
        filename = _CHECKPOINT_FILENAMES[hps_version]
        ckpt = p / filename
        if not ckpt.exists():
            raise FileNotFoundError(f"{filename} not found under {p}")
        return str(ckpt)

    # ── Inference (mirrors official img_score.score logic exactly) ────────

    def _score_single(self, image, prompt: str) -> float:
        """Score one (image, prompt) pair — logic mirrors GitHub main ``img_score.score``.

        The per-item loop, ``torch.no_grad``, ``torch.cuda.amp.autocast``,
        ``unsqueeze(0)``, ``features @ features.T``, ``diagonal()[0]`` are
        kept identical to the upstream implementation so that numerical
        output is bit-for-bit equivalent.
        """
        with torch.no_grad():
            image_tensor = self._preprocess_val(image).unsqueeze(0).to(
                device=self._device, non_blocking=True
            )
            text_tokens = self._tokenizer([prompt]).to(
                device=self._device, non_blocking=True
            )
            with torch.cuda.amp.autocast():
                outputs = self._model(image_tensor, text_tokens)
                image_features = outputs["image_features"]
                text_features = outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                hps_score = torch.diagonal(logits_per_image).cpu().numpy()
        return float(hps_score[0])

    def _batch_score(
        self,
        images: list,
        prompts: list[str],
    ) -> list[float]:
        """Score a batch of (image, prompt) pairs.

        Iterates per-item to match the upstream ``img_score.score()``
        behaviour exactly (single-image forward pass per item).

        Args:
            images: List of PIL.Image.Image objects.
            prompts: Corresponding text prompts (same length as *images*).

        Returns:
            List of float scores, one per (image, prompt) pair.
        """
        if len(images) != len(prompts):
            raise ValueError(
                f"images ({len(images)}) and prompts ({len(prompts)}) must have same length"
            )
        return [self._score_single(img, prompt) for img, prompt in zip(images, prompts)]

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        """Score items using the cached model — no checkpoint reload."""
        if not items:
            return []
        prompts, images = split_last_turn(items)
        scores = self._batch_score(images, prompts)
        return [{"hpsv2": s} for s in scores]


register("hpsv2", HPSv2Scorer)
