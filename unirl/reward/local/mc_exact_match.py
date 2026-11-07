"""Multiple-choice exact-match reward scorer for VLM QA tasks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

_ANSWER_TAG = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)

_ANSWER_PATTERN = re.compile(
    r"(?:(?:answer|option)\s*(?:is|:)\s*)[\(\[]?\s*([A-D])\s*[\)\]]?",
    re.IGNORECASE,
)

_STANDALONE_LETTER = re.compile(r"\b([A-D])\b")


def _extract_answer_letter_graded(text: str, *, require_phrase: bool = False) -> Tuple[str, float]:
    """Return the extracted letter and its format-quality weight."""
    text = text.strip()

    tag_matches = _ANSWER_TAG.findall(text)
    if tag_matches:
        return tag_matches[-1].upper(), 1.0

    phrase_matches = _ANSWER_PATTERN.findall(text)
    if require_phrase:
        return (phrase_matches[-1].upper(), 0.5) if phrase_matches else ("", 0.0)
    if phrase_matches:
        return phrase_matches[-1].upper(), 0.5

    if len(text) == 1 and text in "1234":
        return chr(ord("A") + ord(text) - ord("1")), 0.5
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper(), 0.5

    matches = _STANDALONE_LETTER.findall(text)
    if matches:
        return matches[-1].upper(), 0.5

    return "", 0.0


class MCExactMatchRewardScorer(LocalRewardBackend):
    """Multiple-choice exact-match reward for VLM QA tasks."""

    canonical_model_name = "mc_exact_match"
    input_kind = "text"

    def __init__(self, *, config: "MCExactMatchSpec", base_device: str) -> None:
        del base_device
        super().__init__()
        self.graded_format_reward = bool(config.graded_format_reward)
        self.require_answer_tag = bool(config.require_answer_tag)
        self.require_answer_phrase = bool(config.require_answer_phrase)

    def _load_model(self) -> None:
        pass

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MCExactMatchRewardScorer requires request.texts (generated answers).")
        metadata_list = [None] * len(generated) if request.metadata is None else request.metadata
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            raw_gt = str(meta["answer"]).strip().upper()
            gt = chr(ord("A") + ord(raw_gt) - ord("1")) if len(raw_gt) == 1 and raw_gt in "1234" else raw_gt
            if self.require_answer_tag:
                matches = _ANSWER_TAG.findall(text.strip())
                predicted = matches[-1].upper() if matches else ""
                rewards.append(1.0 if predicted == gt else 0.0)
            else:
                predicted, fmt_weight = _extract_answer_letter_graded(text, require_phrase=self.require_answer_phrase)
                if predicted != gt:
                    rewards.append(0.0)
                else:
                    rewards.append(fmt_weight if self.graded_format_reward else 1.0)

        return rewards


@dataclass
class MCExactMatchSpec(BaseRewardComponentSpec):
    """Configure exact match and optional graded answer formatting."""

    graded_format_reward: bool = False
    require_answer_tag: bool = False
    require_answer_phrase: bool = False

    def __post_init__(self) -> None:
        enabled = sum((self.graded_format_reward, self.require_answer_tag, self.require_answer_phrase))
        if enabled > 1:
            raise ValueError(
                "MCExactMatchSpec graded_format_reward, require_answer_tag, "
                "and require_answer_phrase are mutually exclusive"
            )
