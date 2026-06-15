"""``<think>...</think>`` tag parsing (Qwen3-style thinking models)."""

from __future__ import annotations

import re
from typing import Tuple

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>(.*)$", re.DOTALL)


def split_thinking_tags(text: str) -> Tuple[str, str]:
    """Split LLM output into ``(content, reasoning_content)``.

    Handles both closed ``<think>...</think>`` and unclosed ``<think>...`` (when
    ``max_new_tokens`` cuts off before the closing tag).
    """
    matches = _THINK_PATTERN.findall(text)
    if matches:
        reasoning = "\n".join(matches)
        content = _THINK_PATTERN.sub("", text).strip()
        return content, reasoning

    unclosed = _UNCLOSED_THINK_PATTERN.search(text)
    if unclosed:
        reasoning = unclosed.group(1).strip()
        content = text[: unclosed.start()].strip()
        return content, reasoning

    return text.strip(), ""


__all__ = ["split_thinking_tags"]
