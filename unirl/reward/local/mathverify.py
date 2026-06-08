r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify).

The AdaSPO/DRPO paper grades competition-math answers with the ``math-verify``
library (Appendix D). Unlike :class:`MathBoxedRewardScorer` (a custom ``\boxed``
regex + sympy matcher that REQUIRES a ``\boxed{}`` / ``####`` and scores 0
otherwise), math-verify also extracts a final answer from free-form text — e.g. a
base model that reasons to the correct answer without wrapping it in ``\boxed{}``
— and handles richer symbolic equivalence. So it does NOT under-count
correct-but-unboxed generations, which is common for Qwen3-*-Base + thinking and
otherwise artificially depresses the reward / inflates zero-advantage groups.

Reward = 1.0 if ``math_verify.verify(parse(ground_truth), parse(response))`` else
0.0. Requires the ``math-verify`` package (``pip install math-verify``; pulls
``latex2sympy2-extended``) in the node venv. The import is lazy (only the workers
that actually score need the dependency).
"""

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
        # Lazy import so the dependency is only needed where scoring runs.
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
                # Gold wrapped in \boxed{} (math-verify's canonical gold form); the
                # prediction is parsed free-form (math-verify finds the final answer
                # via \boxed{} else the last expression). verify(gold, target).
                ok = bool(verify(parse("\\boxed{" + gt + "}"), parse(text or "")))
            except Exception:
                ok = False
            rewards.append(1.0 if ok else 0.0)
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
