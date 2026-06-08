"""GSM8K-style numeric exact-match reward scorer for text-RL QA tasks.

Sibling of :mod:`mc_exact_match` (which is letter-only A-D). Here the answer is
a number: extraction prefers the GSM8K ``#### <answer>`` marker, then falls back
to the last numeric value in the text; the prediction is string-compared to the
ground-truth answer (commas stripped). Used by the AR DRPO Qwen3 recipe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

# GSM8K "#### <number>" final-answer marker (highest priority).
_HASH_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")
# Any integer / decimal (fallback: take the last one in the text).
_NUMERIC_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def _extract_answer(text: str) -> str:
    """Extract the predicted numeric answer.

    Priority: ``#### <number>`` marker → last numeric value in the text.
    """
    if not text:
        return ""
    hash_match = _HASH_ANSWER_PATTERN.search(text)
    if hash_match:
        return hash_match.group(1).replace(",", "")
    nums = _NUMERIC_PATTERN.findall(text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


class GSM8KExactMatchRewardScorer(LocalRewardBackend):
    """Numeric exact-match reward for GSM8K-style QA (1.0 if the extracted
    answer equals the ground truth, else 0.0)."""

    canonical_model_name = "gsm8k_exact_match"
    input_kind = "text"

    def __init__(self, *, config: "GSM8KExactMatchSpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "gsm8k_exact_match"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("GSM8KExactMatchRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = str(meta["answer"]).strip().replace(",", "")
            predicted = _extract_answer(text)
            rewards.append(1.0 if predicted == gt else 0.0)
        return rewards


@dataclass
class GSM8KExactMatchSpec(BaseRewardComponentSpec):
    """Config for the GSM8K numeric exact-match scorer."""
