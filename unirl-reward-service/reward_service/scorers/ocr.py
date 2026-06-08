"""OCR reward scorer using GOT-OCR-2.0-hf.

Evaluates text rendering quality in generated images by:
1. Running OCR on the image using GOT-OCR-2.0-hf (a vision-language model)
2. Extracting the target text from the prompt (text between quotes)
3. Computing reward based on edit distance between OCR output and target text

The reward formula (same as flow_grpo):
    reward = 1 - min(edit_distance, len(target)) / len(target)

This gives:
    - 1.0 if the recognized text exactly matches the target
    - 0.0 if completely wrong (edit distance >= target length)
    - Partial credit for partially correct text

Dependencies (isolated via envs/ocr.txt + Ray runtime_env):
    - transformers (AutoModelForImageTextToText, AutoProcessor)
    - python-Levenshtein (edit distance computation)

Reference: https://github.com/yifan123/flow_grpo/blob/main/flow_grpo/ocr.py
Model: https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf
"""

from __future__ import annotations

import torch
from PIL import Image

from reward_service.logging_utils import get_logger
from reward_service.scorers._common import resolve_dtype, resolve_model_path
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.ocr_common import (
    compute_ocr_reward as _compute_ocr_reward,
    extract_target_text as _extract_target_text,
)
from reward_service.scorers.registry import register

logger = get_logger(__name__)


class OCRScorer(BaseScorer):
    """OCR reward scorer using GOT-OCR-2.0-hf for text recognition.

    Extracts text rendered in generated images and computes a reward
    based on edit-distance similarity to the target text (extracted from
    quotes in the prompt).
    """

    name = "ocr"
    sub_metric_names = ("ocr",)

    def __init__(
        self,
        model_name_or_path: str = "stepfun-ai/GOT-OCR-2.0-hf",
        weights_path: str | None = None,
        max_new_tokens: int = 4096,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ) -> None:
        # transformers lives in the per-scorer venv (envs/ocr.txt); import
        # here so the main process stays dependency-free (see clip.py).
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.max_new_tokens = max_new_tokens
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        model_path = resolve_model_path(model_name_or_path, weights_path)
        logger.info("Loading GOT-OCR model: %s on %s", model_path, self.device)

        torch_dtype = resolve_dtype(dtype)

        self._processor = AutoProcessor.from_pretrained(model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=str(self.device),
        )
        self._model.eval()
        logger.info("GOT-OCR model loaded successfully")

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        """Compute OCR reward for each item.

        Target text is extracted from quotes in the prompt. The reward is
        based on normalized edit distance between OCR output and target.
        """
        if not items:
            return []

        results: list[dict[str, float]] = []
        for item in items:
            text, image = item.history[-1]
            target_text = _extract_target_text(text)
            try:
                recognized_text = self._run_ocr(image)
            except Exception as e:
                # Score this item NaN (not 0.0) so an inference failure stays
                # distinguishable from a legitimate "no text rendered" reward:
                # the consumer treats non-finite as a per-item failure (see the
                # BaseScorer.score contract) rather than a real 0.0. Per-item
                # isolation — one bad image does not fail the whole batch, which
                # a raise here would.
                logger.error("OCR inference failed; scoring NaN for this item: %s", e)
                results.append({"ocr": float("nan")})
                continue
            reward = _compute_ocr_reward(recognized_text, target_text)
            results.append({"ocr": float(reward)})

        return results

    def _run_ocr(self, image: Image.Image) -> str:
        """Run GOT-OCR inference on a single image and return recognized text.

        Raises on inference failure; ``score`` turns that into a NaN reward
        for the affected item rather than a misleading 0.0.
        """
        inputs = self._processor(image, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        generate_ids = self._model.generate(
            **inputs,
            do_sample=False,
            tokenizer=self._processor.tokenizer,
            stop_strings="<|im_end|>",
            max_new_tokens=self.max_new_tokens,
        )

        # Decode only the generated tokens (skip input tokens)
        input_len = inputs["input_ids"].shape[1]
        text = self._processor.decode(
            generate_ids[0, input_len:], skip_special_tokens=True
        )
        return text.strip()

    def close(self) -> None:
        """Release model resources."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


register("ocr", OCRScorer)
