"""Shared, dependency-light primitives for trustworthy TTS rewards."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import torch

from unirl.types.reward import RewardResponse

_CJK_LANGUAGES = frozenset({"zh", "yue", "ja", "ko"})
_SPACE_RE = re.compile(r"\s+")


class RewardUnavailableError(RuntimeError):
    """Raised when a required reward model or reference is unavailable."""


def language_root(language: str) -> str:
    return str(language or "").strip().lower().replace("_", "-").split("-", 1)[0]


def normalize_tts_text(text: str, *, language: str = "") -> str:
    """Normalize Unicode, digits, punctuation, and spacing deterministically.

    NFKC canonicalizes full-width Latin letters and decimal digits. Punctuation,
    symbols, and control characters become spaces for whitespace-tokenized
    languages and are removed between CJK characters by the final whitespace
    collapse. Number values are intentionally not rewritten into words: that is
    locale-dependent and would make the reference depend on a hidden NLP model.
    """

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    chars: List[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in {"P", "S", "C"}:
            chars.append(" ")
        else:
            chars.append(char)
    normalized = _SPACE_RE.sub(" ", "".join(chars)).strip()
    if language_root(language) in _CJK_LANGUAGES:
        # Spaces are not lexical boundaries for character-error languages.
        normalized = normalized.replace(" ", "")
    return normalized


def metric_for_language(language: str, cer_languages: Iterable[str]) -> str:
    roots = {language_root(item) for item in cer_languages}
    return "cer" if language_root(language) in roots else "wer"


def metric_tokens(text: str, *, metric: str, language: str = "") -> List[str]:
    normalized = normalize_tts_text(text, language=language)
    if metric == "cer":
        return list(normalized.replace(" ", ""))
    if metric == "wer":
        return normalized.split()
    raise ValueError(f"Unsupported edit metric {metric!r}; expected 'wer' or 'cer'.")


@dataclass(frozen=True)
class EditCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_units: int
    hypothesis_units: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_units == 0:
            return 0.0 if self.hypothesis_units == 0 else 1.0
        return float(self.errors) / float(self.reference_units)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "substitutions": self.substitutions,
            "deletions": self.deletions,
            "insertions": self.insertions,
            "errors": self.errors,
            "reference_units": self.reference_units,
            "hypothesis_units": self.hypothesis_units,
            "rate": self.rate,
        }


def edit_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Levenshtein distance with a stable S/D/I traceback.

    Tie-breaking is substitution, deletion, insertion. The distance is
    unaffected by this choice; deterministic raw counts make evaluation diffs
    reproducible.
    """

    n, m = len(reference), len(hypothesis)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    operation = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        distance[i][0] = i
        operation[i][0] = "D"
    for j in range(1, m + 1):
        distance[0][j] = j
        operation[0][j] = "I"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                distance[i][j] = distance[i - 1][j - 1]
                operation[i][j] = "M"
                continue
            candidates = (
                (distance[i - 1][j - 1] + 1, "S"),
                (distance[i - 1][j] + 1, "D"),
                (distance[i][j - 1] + 1, "I"),
            )
            distance[i][j], operation[i][j] = min(candidates, key=lambda item: item[0])

    substitutions = deletions = insertions = 0
    i, j = n, m
    while i or j:
        op = operation[i][j]
        if op == "M":
            i -= 1
            j -= 1
        elif op == "S":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "D":
            deletions += 1
            i -= 1
        elif op == "I":
            insertions += 1
            j -= 1
        else:  # only reachable at [0, 0]
            break
    return EditCounts(substitutions, deletions, insertions, n, m)


def score_edit_metric(reference: str, hypothesis: str, *, metric: str, language: str = "") -> Dict[str, Any]:
    ref_units = metric_tokens(reference, metric=metric, language=language)
    hyp_units = metric_tokens(hypothesis, metric=metric, language=language)
    counts = edit_counts(ref_units, hyp_units)
    return {
        "metric": metric,
        "reference_normalized": normalize_tts_text(reference, language=language),
        "hypothesis_normalized": normalize_tts_text(hypothesis, language=language),
        **counts.as_dict(),
    }


def unavailable_response(batch_size: int, *, component: str, reason: str) -> RewardResponse:
    """A dry-run diagnostic response that cannot enter training."""

    message = f"{component} unavailable: {reason}"
    return RewardResponse(
        rewards=[0.0] * batch_size,
        successes=[False] * batch_size,
        errors=[message] * batch_size,
        details=[{"status": "unavailable", "component": component, "reason": reason} for _ in range(batch_size)],
    )


def mono_waveform(waveform: torch.Tensor) -> torch.Tensor:
    wav = waveform.detach().float().cpu()
    if wav.ndim == 2:
        # Audios convention is [L, C], while some callers still provide [C, L].
        wav = wav.mean(dim=1 if wav.shape[0] >= wav.shape[1] else 0)
    elif wav.ndim != 1:
        wav = wav.reshape(-1)
    return wav.contiguous()


def resample_waveform(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    wav = mono_waveform(waveform)
    if int(source_rate) == int(target_rate):
        return wav
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Sample rates must be positive, got {source_rate} -> {target_rate}.")
    target_length = max(1, round(wav.numel() * float(target_rate) / float(source_rate)))
    return torch.nn.functional.interpolate(
        wav.view(1, 1, -1),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).view(-1)


__all__ = [
    "EditCounts",
    "RewardUnavailableError",
    "edit_counts",
    "language_root",
    "metric_for_language",
    "metric_tokens",
    "mono_waveform",
    "normalize_tts_text",
    "resample_waveform",
    "score_edit_metric",
    "unavailable_response",
]
