"""Config for the trainside (in-process) rollout engine."""

from __future__ import annotations

from dataclasses import dataclass

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class TrainsideEngineConfig(BaseEngineConfig):
    """No static fields — pipeline/policy are runtime handles, not cfg leaves."""

    pass


__all__ = ["TrainsideEngineConfig"]
