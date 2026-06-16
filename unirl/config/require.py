"""One-line precondition helper for dataclass validation.

Replaces the two-line ``if X: raise ValueError(Y)`` idiom with a single call
that expresses the positive invariant::

    require(self.num_gpus >= 1, f"SGLangDiffusionEngineConfig.num_gpus must be >= 1; got {self.num_gpus!r}")

Stdlib-only so light config modules (``logging_config``, ``resume_config``,
...) can import it without pulling in ``torch`` / ``omegaconf``.
"""

from __future__ import annotations


def require(condition: bool, message: str) -> None:
    """Raise ``ValueError(message)`` if ``condition`` is falsy."""
    if not condition:
        raise ValueError(message)


__all__ = ["require"]
