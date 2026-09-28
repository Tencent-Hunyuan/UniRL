"""Helpers for normalizing distributed method return values."""

from __future__ import annotations


def rank_zero_bool(value: object, *, name: str) -> bool:
    """Normalize a rank-zero distributed return without list truthiness."""
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise RuntimeError(f"{name} returned {len(value)} rank-zero values; expected exactly one")
        value = value[0]
    if type(value) is not bool:
        raise TypeError(f"{name} returned {type(value).__name__}, expected bool")
    return value


__all__ = ["rank_zero_bool"]
