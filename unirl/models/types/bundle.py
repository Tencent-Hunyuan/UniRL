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

    # FSDP wrap-policy hint read by the training backend (like ``no_split_modules``):
    # True iff the trainable model's forward — sampling AND replay — is driven
    # entirely through its root module, so an FSDP *root wrap*'s pre-forward
    # all-gather covers every parameter the forward touches. Diffusion bundles
    # whose stage runs the whole transformer forward opt in (set True) to enable
    # ``forward_prefetch`` (which requires a root wrap). Autoregressive bundles MUST
    # leave this False: they apply ``lm_head`` / ``embed`` *outside* the root
    # forward (e.g. chunked over the time dim to bound the ``[B, T, vocab]`` logits
    # transient), so a root wrap would leave those shards un-gathered at the direct
    # call sites and the forward would read sharded weights. Conservative default —
    # opt in per bundle only after verifying the stage drives a single root forward.
    supports_fsdp_root_wrap: bool = False


__all__ = ["Bundle"]
