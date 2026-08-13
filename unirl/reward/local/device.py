"""Shared device resolver for reward Specs."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_REMOTE_HINT = (
    "the local reward backend does not support pinning a specific GPU ordinal — "
    "each reward worker auto-resolves to its own assigned card. Use device='cuda' "
    "(or 'auto'). To give the reward its own dedicated GPU(s), set the trainer's "
    "reward_fraction (places the reward role on its own slab in the same job), or "
    "switch to the remote reward backend (unirl.reward.remote.RemoteRewardBackend)."
)


def resolve_device(spec_device: str, base_device: str) -> str:
    """Resolve a Spec's ``device`` against the cluster-level ``base_device``."""
    chosen = _resolve_one(spec_device)
    if chosen == "auto":
        chosen = _resolve_one(base_device)
    if chosen == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    if chosen == "cuda" and not torch.cuda.is_available():
        logger.warning("Reward Spec requested cuda but CUDA is not available; falling back to cpu.")
        return "cpu"
    return chosen


def _resolve_one(value: str) -> str:
    pref = str(value or "").strip().lower()
    if pref in {"cpu", "cuda", "auto"}:
        return pref
    if pref.startswith("cuda:"):
        raise ValueError(f"Reward Spec device={value!r}: {_REMOTE_HINT}")
    logger.warning(
        "Unknown device pref %r; falling back to cpu.",
        value,
    )
    return "cpu"


__all__ = ["resolve_device"]
