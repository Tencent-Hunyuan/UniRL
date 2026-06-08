"""Typed media references for multimodal rollout inputs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MediaRef:
    """A lightweight reference to one per-sample media input.

    The reference is intentionally small and serializable. Actual media loading
    happens on the actor/sampler side so the driver does not move large tensors
    through Ray.
    """

    modality: str
    role: str
    uri: str


__all__ = ["MediaRef"]
