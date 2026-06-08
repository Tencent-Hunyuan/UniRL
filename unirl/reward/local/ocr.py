"""OCR reward scorer."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List

from PIL import Image
from tqdm import tqdm

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)


class OCRRewardScorer(LocalRewardBackend):
    """OCR reward for text rendering tasks."""

    canonical_model_name = "ocr"

    def __init__(self, *, config: "OCRSpec", base_device: str) -> None:
        # base_device unused: paddleocr forces CPU init.
        del base_device
        super().__init__(lang=config.lang)

    def _load_model(self) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr is required for OCR reward. Install with: pip install paddleocr")

        try:
            from Levenshtein import distance as levenshtein_distance
        except ImportError:
            raise ImportError(
                "python-Levenshtein is required for OCR reward. Install with: pip install python-Levenshtein"
            )

        import paddle

        # PaddleOCR v3 removed `use_gpu=` and can aggressively reserve GPU memory
        # during initialization. Force CPU init so colocate mode does not contend
        # with training/SGLang allocations.
        paddle.set_device("cpu")
        self._ocr_reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=self.model_kwargs.get("lang", "en"),
        )
        self._levenshtein_distance = levenshtein_distance
        self.model = "ocr"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        import numpy as np

        images = request.images
        prompts: List[str] = []
        for idx, raw_prompt in enumerate(request.prompts):
            parts = raw_prompt.split('"')
            if len(parts) < 3:
                raise ValueError(f"OCR reward prompt at index {idx} must contain quoted target text.")
            target_text = parts[1].replace(" ", "").lower()
            if not target_text:
                raise ValueError(f"OCR reward prompt at index {idx} has empty quoted target text.")
            prompts.append(target_text)

        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have the same length")

        rewards: List[float] = []
        rank = int(os.environ.get("RANK", 0))
        progress = tqdm(
            zip(images, prompts),
            desc="Computing OCR rewards",
            disable=(rank != 0),
            total=len(prompts),
        )
        for sample_idx, (img, prompt) in enumerate(progress):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                result = self._run_ocr(img)
                recognized_text = self._extract_recognized_text(result)

                recognized_text = recognized_text.replace(" ", "").lower()
                if prompt in recognized_text:
                    dist = 0
                else:
                    dist = self._levenshtein_distance(recognized_text, prompt)
                if dist > len(prompt):
                    dist = len(prompt)
            except Exception:
                logger.warning(
                    "OCR reward scoring failed for sample %d; assigning worst-case distance.",
                    sample_idx,
                    exc_info=True,
                )
                dist = len(prompt)

            rewards.append(1 - dist / len(prompt))

        return rewards

    def _run_ocr(self, img):
        predict_fn = getattr(self._ocr_reader, "predict", None)
        if callable(predict_fn):
            return predict_fn(img)
        return self._ocr_reader.ocr(img, cls=False)

    def _extract_recognized_text(self, result) -> str:
        texts: List[str] = []
        if isinstance(result, list):
            for page in result:
                if isinstance(page, dict):
                    rec_texts = page.get("rec_texts")
                    if isinstance(rec_texts, list):
                        texts.extend(str(text) for text in rec_texts if text)
                    continue
                if not isinstance(page, list):
                    continue
                for line in page:
                    if not isinstance(line, (list, tuple)) or len(line) < 2:
                        continue
                    candidate = line[1]
                    if isinstance(candidate, (list, tuple)) and candidate:
                        text = candidate[0]
                        if isinstance(text, str) and text:
                            texts.append(text)
        return "".join(texts)


@dataclass
class OCRSpec(BaseRewardComponentSpec):
    """Typed config for the OCR (PaddleOCR) reward component.

    OCR is forced to CPU by paddleocr's runtime and processes one image at a
    time, so neither ``device`` nor ``batch_size`` appear here.
    """

    lang: str = "en"
