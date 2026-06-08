"""Multiple-choice exact-match reward scorer for VLM QA tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

_ANSWER_PATTERN = re.compile(
    r"(?:(?:answer|option)\s*(?:is|:)\s*)\(?([A-D])\)?",
    re.IGNORECASE,
)

_STANDALONE_LETTER = re.compile(r"\b([A-D])\b")


def _normalize_answer(answer: str) -> str:
    """Normalize answer to A/B/C/D letter.

    Handles: "A"/"B"/"C"/"D" → "A"/"B"/"C"/"D"
                 "1"/"2"/"3"/"4" → "A"/"B"/"C"/"D"
    """
    a = answer.strip().upper()
    # Numeric → letter
    if len(a) == 1 and a in "1234":
        return chr(ord("A") + ord(a) - ord("1"))
    # Already a letter
    if len(a) == 1 and a in "ABCD":
        return a
    return a  # fallback


def _extract_answer_letter(text: str) -> str:
    text = text.strip()
    # Handle numeric answers: "1"→"A", "2"→"B", "3"→"C", "4"→"D"
    if len(text) == 1 and text in "1234":
        return chr(ord("A") + ord(text) - ord("1"))
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    match = _ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    matches = _STANDALONE_LETTER.findall(text)
    if matches:
        return matches[-1].upper()
    return ""


class MCExactMatchRewardScorer(LocalRewardBackend):
    """Multiple-choice exact-match reward for VLM QA tasks."""

    canonical_model_name = "mc_exact_match"
    input_kind = "text"

    def __init__(self, *, config: "MCExactMatchSpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "mc_exact_match"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MCExactMatchRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = _normalize_answer(str(meta["answer"]))
            predicted = _extract_answer_letter(text)
            rewards.append(1.0 if predicted == gt else 0.0)

        return rewards


@dataclass
class MCExactMatchSpec(BaseRewardComponentSpec):
    """Config for the MC exact-match scorer."""
