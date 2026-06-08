"""Text-extraction and edit-distance reward helpers for the OCR scorer.

Kept separate from ocr.py so the pure-Python logic is unit-testable
without importing transformers, and so the heavy Levenshtein dependency
stays out of module top level (imported lazily inside compute_ocr_reward).
"""

from __future__ import annotations

import re

# Target text is the first quoted span in the prompt. Precompiled at module
# scope, matching the _SCORE_PATTERN convention in unified_reward.py.
_DOUBLE_QUOTED = re.compile(r'"([^"]+)"')
_SINGLE_QUOTED = re.compile(r"'([^']+)'")


def extract_target_text(prompt: str) -> str:
    """Extract the target text from the prompt (text between quotes).

    Examples:
        'A sign that says "Hello World"' -> 'Hello World'
        'New York Skyline with "Hello World" written with fireworks' -> 'Hello World'

    If no quoted text is found, returns the full prompt as fallback.
    """
    match = _DOUBLE_QUOTED.search(prompt)
    if not match:
        match = _SINGLE_QUOTED.search(prompt)
    if match:
        return match.group(1)
    return prompt


def compute_ocr_reward(recognized_text: str, target_text: str) -> float:
    """Compute OCR reward using normalized Levenshtein distance.

    Matches the logic in flow_grpo/ocr.py:
    - Normalize both texts (lowercase, remove spaces)
    - If target is a substring of recognized text, perfect match
    - Otherwise compute edit distance
    - Cap penalty at len(target) to avoid over-penalizing extra characters
    - Return 1 - distance / len(target)
    """
    # python-Levenshtein is a per-scorer venv dep (envs/ocr.txt); import
    # lazily so this main-process-importable helper carries no heavy dep.
    from Levenshtein import distance

    recognized_norm = recognized_text.replace(" ", "").lower()
    target_norm = target_text.replace(" ", "").lower()

    if not target_norm:
        return 1.0  # No target text = trivially satisfied

    # If target is a substring of recognized text, it's a perfect match
    if target_norm in recognized_norm:
        dist = 0
    else:
        dist = distance(recognized_norm, target_norm)

    # Cap distance at target length
    if dist > len(target_norm):
        dist = len(target_norm)

    return 1 - dist / len(target_norm)
