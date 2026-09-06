"""OCR reward scorer."""

from __future__ import annotations

import logging
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import List

from PIL import Image
from tqdm import tqdm

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest, RewardResponse

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)


class OCRRewardScorer(LocalRewardBackend):
    """OCR reward for text rendering tasks."""

    canonical_model_name = "ocr"

    def __init__(self, *, config: "OCRSpec", base_device: str) -> None:
        del base_device
        super().__init__(lang=config.lang)

    def _load_model(self) -> None:
        self._levenshtein_distance = None
        try:
            from Levenshtein import distance as levenshtein_distance
        except ImportError:
            logger.info("python-Levenshtein is unavailable; using the pure-Python OCR edit-distance fallback.")
        else:
            self._levenshtein_distance = levenshtein_distance

        try:
            from paddleocr import PaddleOCR
        except ImportError:
            raise ImportError("paddleocr is required for OCR reward. Install with: pip install paddleocr")

        import paddle

        paddle.set_device("cpu")
        self._ocr_reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang=self.model_kwargs.get("lang", "en"),
        )
        self.model = "ocr"

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Score OCR once and expose order-sensitive diagnostics for every sample."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{type(self).__name__}.compute_rewards called before _load_model "
                f"completed (model_name={self.model_name!r}, batch_size={request.batch_size})."
            )
        start = time.time()
        rewards, component_rewards = self._compute_rewards_and_components(request)
        return RewardResponse(
            rewards=rewards,
            component_rewards=component_rewards,
            successes=[True] * len(rewards),
            errors=[None] * len(rewards),
            compute_time=time.time() - start,
        )

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        rewards, _ = self._compute_rewards_and_components(request)
        return rewards

    def _compute_rewards_and_components(self, request: RewardRequest) -> tuple[List[float], dict[str, List[float]]]:
        import numpy as np

        images = request.images
        if images is None:
            raise ValueError("OCR reward requires generated images")

        prompts: List[str] = []
        for idx, raw_prompt in enumerate(request.prompts):
            target_text = self._extract_target_text(raw_prompt)
            if not target_text:
                raise ValueError(f"OCR reward prompt at index {idx} has empty quoted target text.")
            prompts.append(target_text)

        if len(images) != len(prompts):
            raise ValueError("Images and prompts must have the same length")

        rewards: List[float] = []
        component_rewards: dict[str, List[float]] = {
            "edit_similarity": [],
            "substring_match": [],
            "exact_match": [],
            "counter_iou": [],
        }
        rank = int(os.environ.get("RANK", 0))
        progress = tqdm(
            zip(images, prompts),
            desc="Computing OCR rewards",
            disable=(rank != 0),
            total=len(prompts),
        )
        for sample_idx, (img, target) in enumerate(progress):
            if isinstance(img, Image.Image):
                img = np.array(img)

            try:
                result = self._run_ocr(img)
                prediction = self._normalize_text(self._extract_recognized_text(result))
                metrics = self._score_text(prediction, target)
            except Exception:
                logger.warning(
                    "OCR reward scoring failed for sample %d; assigning zero reward.",
                    sample_idx,
                    exc_info=True,
                )
                metrics = {
                    "edit_similarity": 0.0,
                    "substring_match": 0.0,
                    "exact_match": 0.0,
                    "counter_iou": 0.0,
                }

            rewards.append(metrics["edit_similarity"])
            for name, value in metrics.items():
                component_rewards[name].append(value)

        return rewards, component_rewards

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

    @classmethod
    def _extract_target_text(cls, prompt: str) -> str:
        match = re.search(r'["“”]([^"“”]+)["“”]', str(prompt))
        if match is None:
            raise ValueError("OCR reward prompt must contain quoted target text.")
        return cls._normalize_text(match.group(1))

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(re.findall(r"\w", str(text).casefold(), flags=re.UNICODE))

    @staticmethod
    def _python_levenshtein_distance(left: str, right: str) -> int:
        """Dependency-free Levenshtein fallback for the short OCR target strings."""
        if len(left) < len(right):
            left, right = right, left
        previous = list(range(len(right) + 1))
        for row, left_char in enumerate(left, start=1):
            current = [row]
            for col, right_char in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[col] + 1,
                        previous[col - 1] + (left_char != right_char),
                    )
                )
            previous = current
        return previous[-1]

    def _score_text(self, prediction: str, target: str) -> dict[str, float]:
        substring_match = float(bool(target) and target in prediction)
        exact_match = float(prediction == target)
        if substring_match:
            edit_similarity = 1.0
        else:
            distance_fn = self._levenshtein_distance or self._python_levenshtein_distance
            distance = min(int(distance_fn(prediction, target)), len(target))
            edit_similarity = 1.0 - distance / len(target)
        return {
            "edit_similarity": float(edit_similarity),
            "substring_match": substring_match,
            "exact_match": exact_match,
            "counter_iou": self._counter_iou(prediction, target),
        }

    @staticmethod
    def _counter_iou(prediction: str, target: str) -> float:
        pred, gold = Counter(prediction), Counter(target)
        union = sum((pred | gold).values())
        return float(sum((pred & gold).values()) / union) if union else 1.0


@dataclass
class OCRSpec(BaseRewardComponentSpec):
    """Typed config for the OCR (PaddleOCR) reward component."""

    lang: str = "en"
