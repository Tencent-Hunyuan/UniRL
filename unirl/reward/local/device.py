"""Shared device resolver for reward Specs.

A Spec's ``device: str`` field accepts ``"cpu"``, ``"cuda"``, or ``"auto"``.
``"auto"`` defers to the backend's ``base_device`` (also ``"cpu"``/``"cuda"``/
``"auto"``); when that itself is ``"auto"``, fall back to ``"cuda"`` if available
else ``"cpu"``. This lets per-component overrides win where set, while keeping a
single cluster-level default.

Device selection here is intentionally coarse (cpu / cuda / auto), NOT a
GPU-pinning knob. An explicit ordinal like ``"cuda:1"`` cannot be honored safely
in the distributed local path (every DP worker would pile onto the same physical
card, and the ordinal mis-maps across nodes / ``CUDA_VISIBLE_DEVICES``), so it is
**rejected**. To give a (possibly heavy) reward model its OWN dedicated GPU(s), do
NOT pin an ordinal — instead either:
  * set the trainer's ``reward_fraction`` so the reward role is placed on its own
    disjoint GPU slab in the SAME job (each reward worker still resolves
    ``device="cuda"``/``"auto"`` to its assigned card via ``CUDA_VISIBLE_DEVICES``);
    validated for an 8B VLM reward (Qwen3-VL) at 8- and 16-GPU; or
  * use the remote reward backend (``unirl.reward.remote.RemoteRewardBackend``),
    which owns its own device pool independent of the trainer.
"""

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
    """Resolve a Spec's ``device`` against the cluster-level ``base_device``.

    Precedence: explicit ``cpu``/``cuda`` on the spec wins; ``auto`` falls
    through to ``base_device``; if that is also ``auto``, pick ``cuda`` when
    available else ``cpu``. ``cuda`` requested without availability falls back to
    ``cpu`` with a warning. An explicit ``cuda:<index>`` ordinal raises
    ``ValueError`` — local rewards do not pin GPUs (see module docstring); use the
    remote backend instead.
    """
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
    # Explicit CUDA ordinal (e.g. "cuda:0" / "cuda:1"): reject loudly rather than
    # silently dropping to CPU. Local rewards have no safe GPU-pinning path; point
    # the user at the remote backend.
    if pref.startswith("cuda:"):
        raise ValueError(f"Reward Spec device={value!r}: {_REMOTE_HINT}")
    logger.warning(
        "Unknown device pref %r; falling back to cpu.",
        value,
    )
    return "cpu"


__all__ = ["resolve_device"]
