"""Shared PE (Prompt Enhancement) text post-processing helpers.

The LLM rewriter emits free-form chat text; both the sglang
:class:`~unirl.rollout.engine.composed.engine.ComposedRolloutEngine` and the
trainside :class:`~unirl.models.pe.pipeline.PEPipeline` must turn that raw
output into the clean rewritten prompt the diffusion child conditions on. These
two helpers are the single source of truth for that extraction so the two PE
paths stay byte-identical.

Lives under ``unirl/models/pe/`` (not the rollout engine) so the trainside
pipeline can import it without a ``models -> rollout`` dependency edge — the
dependency only ever runs ``rollout -> models``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def extract_pe_text(raw_text: str, marker: str) -> str:
    """Return the substring after the LAST occurrence of ``marker``.

    Pre-strips an optional ``<think>...</think>`` reasoning preamble (Qwen3
    chat output), then takes everything after the last ``marker`` and
    removes a wrapping pair of quotes. Returns ``""`` when the marker is
    absent so the caller can fall back to the original user prompt.
    """
    text = (raw_text or "").strip()
    if not text:
        return ""

    think_close = text.rfind("</think>")
    if think_close != -1:
        text = text[think_close + len("</think>") :].strip()
        if not text:
            return ""

    marker_idx = text.rfind(marker)
    if marker_idx == -1:
        return ""

    pe_text = text[marker_idx + len(marker) :].strip()
    if len(pe_text) >= 2 and pe_text[0] == pe_text[-1] and pe_text[0] in ('"', "'"):
        pe_text = pe_text[1:-1].strip()
    return pe_text


def postprocess_pe_texts(
    raw_texts: List[str],
    *,
    user_prompts: List[str],
    samples_per_prompt: int,
    marker: str,
    max_chars: Optional[int] = None,
) -> Tuple[List[str], Dict[str, int]]:
    """Run marker extraction + truncation + empty-fallback over PE outputs.

    ``raw_texts`` is PE-major over ``[P*N]``; the user prompt for slot
    ``k`` is ``user_prompts[k // samples_per_prompt]``. Off-format / empty
    outputs fall back to the original user prompt so the diffusion child
    never conditions on blank text. Returns ``(cleaned_texts, stats)`` where
    ``stats`` counts ``empty`` / ``truncated`` / ``fallback`` slots.
    """
    cleaned: List[str] = []
    stats = {"empty": 0, "truncated": 0, "fallback": 0}
    for k, raw in enumerate(raw_texts):
        pe = extract_pe_text(raw, marker)
        if not pe:
            stats["empty"] += 1
        if max_chars is not None and len(pe) > int(max_chars):
            pe = pe[: int(max_chars)]
            stats["truncated"] += 1
        if not pe.strip():
            idx = k // max(1, samples_per_prompt)
            pe = user_prompts[idx] if idx < len(user_prompts) else ""
            stats["fallback"] += 1
        cleaned.append(pe)
    return cleaned, stats


__all__ = ["extract_pe_text", "postprocess_pe_texts"]
