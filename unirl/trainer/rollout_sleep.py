"""Scheduling helpers for rollout engines whose deep sleep discards weights."""

from __future__ import annotations


def must_preserve_rollout_weights(
    *,
    uses_gpu_streaming: bool,
    next_phase_syncs: bool,
    has_next_phase: bool,
) -> bool:
    """Return whether the current sleep must retain weights for the next phase."""
    return bool(uses_gpu_streaming and has_next_phase and not next_phase_syncs)


__all__ = ["must_preserve_rollout_weights"]
