"""Bundle — identity contract for a model's collection of related modules.

A ``Bundle`` is the typed container a ``Pipeline``'s stages call into to
access the model's transformer / VAE / text encoders / scheduler / etc.
This Protocol is intentionally empty: concrete bundles add accessors for
the modules they own, and lifecycle concerns (LoRA, FSDP wrap, adapter
switching, autocast) live outside the bundle.
"""

from __future__ import annotations

from unirl.distributed.group.remote import Remote


class Bundle(Remote):
    """Collection of related modules (transformer, VAE, encoders, scheduler, …)
    that a ``Pipeline``'s stages dispatch against."""


__all__ = ["Bundle"]
