"""Bundle — identity contract for a model's collection of related modules."""

from __future__ import annotations

from unirl.distributed.group.remote import Remote


class Bundle(Remote):
    """Collection of related modules a ``Pipeline``'s stages dispatch against."""


__all__ = ["Bundle"]
