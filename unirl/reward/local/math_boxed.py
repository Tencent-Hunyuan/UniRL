r"""Boxed-answer numeric-match reward scorer for math RL (competition-style).

Sibling of :mod:`gsm8k_exact_match`. That scorer was authored for GSM8K's
``#### <number>`` convention and falls back to "last number in the text" — both
mismatch how Qwen3 (and datasets like DAPO-Math) actually emit answers: a
``\boxed{...}`` at the end of the solution. This scorer:

1. extracts the LAST ``\boxed{...}`` (balanced braces; tolerant of truncation),
   taking the part after the last ``=`` when the box holds ``... = <answer>``;
2. compares it to the ground truth NUMERICALLY via sympy (so ``34.0 == 34`` and
   ``3/2 == 1.5``), with string- and float-equality fallbacks;
3. falls back to ``#### <num>`` when no box is present, but deliberately NOT to
   the last number in the text (a false-positive source on rambling / truncated
   output); with no parseable boxed/#### answer the reward is 0.

No new dependencies — uses sympy (already in the env). Used by the Qwen3-4B-Base
DAPO-Math DRPO recipe (long boxed-answer generations).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

_HASH_ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)")


def _last_boxed(text: str) -> Optional[str]:
    r"""Return the content of the last ``\boxed{...}`` (balanced braces), or None.

    Returns None if there is no ``\boxed`` or its braces never close (generation
    truncated mid-box) — both correctly score as "no answer".
    """
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    open_brace = text.find("{", idx)
    if open_brace == -1:
        return None
    depth = 0
    for j in range(open_brace, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : j]
    return None  # unbalanced → truncated


def _extract_prediction(text: str) -> str:
    r"""Extract the model's final answer string.

    Priority: last ``\boxed{...}`` (part after last ``=``) → ``#### <num>``.
    Deliberately NO "last number in the text" fallback: the model reliably emits
    ``\boxed{}``, and grabbing a trailing number from rambling / truncated text is
    a false-positive source. No parseable answer → "" → reward 0.
    """
    if not text:
        return ""
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.rsplit("=", 1)[-1].strip() if "=" in boxed else boxed.strip()
    hash_match = _HASH_ANSWER_PATTERN.search(text)
    if hash_match:
        return hash_match.group(1).replace(",", "")
    return ""


def _normalize_latex(s: str) -> str:
    r"""Light LaTeX → sympy-parsable normalization (no latex2sympy dependency)."""
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\s*\{[^{}]*\}", "", s)
    for tok in ("\\left", "\\right", "\\,", "\\!", "\\;", "\\ ", "\\quad", "\\qquad"):
        s = s.replace(tok, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    # \frac{a}{b} -> (a)/(b)  (one level; nested fracs are rare in final answers)
    for _ in range(3):
        s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("^", "**")
    s = s.replace(",", "").replace(" ", "").replace("%", "")
    return s


def _values_equal(pred: str, gt: str) -> bool:
    """True if pred and gt denote the same number.

    Exact rational comparison first (so 18+ digit integers don't collapse under
    float rounding), then sympy for latex-ish forms, then a tolerant float as a
    last resort.
    """
    if not pred:
        return False
    if pred.strip() == gt.strip():
        return True
    np, ng = _normalize_latex(pred), _normalize_latex(gt)
    if not np:
        return False
    from fractions import Fraction

    try:  # exact: integers, decimals, "a/b"
        return Fraction(np) == Fraction(ng)
    except (ValueError, ZeroDivisionError):
        pass
    try:  # symbolic: normalized forms like "((3)/(2))" or a bare "2**3"
        import sympy

        return bool(sympy.simplify(sympy.sympify(np) - sympy.sympify(ng)) == 0)
    except Exception:
        logger.debug("Sympy comparison failed for pred=%r gt=%r.", pred, gt, exc_info=True)
    try:
        return abs(float(np) - float(ng)) < 1e-9
    except (ValueError, TypeError):
        return False


class MathBoxedRewardScorer(LocalRewardBackend):
    r"""Numeric reward for ``\boxed{}``-style math answers (1.0 if the boxed value
    equals the ground truth, else 0.0)."""

    canonical_model_name = "math_boxed"
    input_kind = "text"

    def __init__(self, *, config: "MathBoxedSpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "math_boxed"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MathBoxedRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards: List[float] = []
        for text, meta in zip(generated, metadata_list):
            if meta is None or "answer" not in meta:
                rewards.append(0.0)
                continue
            gt = str(meta["answer"]).strip()
            predicted = _extract_prediction(text)
            rewards.append(1.0 if _values_equal(predicted, gt) else 0.0)
        return rewards


@dataclass
class MathBoxedSpec(BaseRewardComponentSpec):
    r"""Config for the ``\boxed{}`` numeric-match scorer."""
