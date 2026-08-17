r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend


class MathVerifyRewardScorer(LocalRewardBackend):
    r"""Numeric/symbolic reward via HuggingFace ``math-verify`` (1.0 match / 0.0)."""

    canonical_model_name = "math_verify"
    input_kind = "text"

    def __init__(self, *, config: "MathVerifySpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "math_verify"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        from math_verify import parse, verify

        generated = request.texts
        if generated is None:
            raise ValueError("MathVerifyRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = str(meta["answer"]).strip()
            try:
                ok = bool(
                    verify(
                        parse("\\boxed{" + gt + "}", parsing_timeout=None),
                        parse(text or "", parsing_timeout=None),
                        timeout_seconds=None,
                    )
                )
            except Exception:
                ok = False
            rewards.append(1.0 if ok else 0.0)
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
