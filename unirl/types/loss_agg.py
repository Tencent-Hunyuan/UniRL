"""Stdlib-only ``loss_agg_mode`` vocabulary, importable by recipe checks that run without torch."""

from __future__ import annotations

from enum import Enum
from typing import Any


class LossAggMode(str, Enum):
    """AR token-loss reductions; values are the YAML ``loss_agg_mode`` strings (verl ``agg_loss`` names)."""

    TOKEN_MEAN = "token-mean"
    SEQ_MEAN_TOKEN_MEAN = "seq-mean-token-mean"
    SEQ_MEAN_TOKEN_SUM_NORM = "seq-mean-token-sum-norm"


def parse_loss_agg_mode(value: Any, *, owner: str) -> LossAggMode:
    """Parse a config ``loss_agg_mode``; raises ``ValueError`` naming ``owner`` and the accepted values."""
    try:
        return LossAggMode(value)
    except ValueError:
        accepted = ", ".join(repr(mode.value) for mode in LossAggMode)
        raise ValueError(f"{owner}: loss_agg_mode must be one of {accepted}; got {value!r}") from None


__all__ = ["LossAggMode", "parse_loss_agg_mode"]
