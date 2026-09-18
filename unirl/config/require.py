"""One-line precondition helper for dataclass validation."""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` if ``condition`` is falsy."""
    if not condition:
        raise ValueError(message)


__all__ = ["require"]
