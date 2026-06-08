"""Config for the trainside (in-process) rollout engine.

Empty dataclass — the engine's only runtime deps (``pipeline``, ``policy``)
are Python handles owned by the train actor, injected when the actor
constructs :class:`TrainsideRolloutEngine`.

Recipes wire ``rollout: {_target_: ...TrainsideRolloutEngine, ...}`` directly;
a ``_target_`` ending in ``TrainsideRolloutEngine`` is what
:func:`unirl.config.validation.is_direct_sampling` keys off.
"""

from __future__ import annotations

from dataclasses import dataclass

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class TrainsideEngineConfig(BaseEngineConfig):
    """No static fields — pipeline/policy are runtime handles, not cfg leaves.

    Having no cfg leaves lets :class:`TrainsideRolloutEngine.__init__` keep
    its keyword-only ``(pipeline, policy)`` signature without accepting a
    vestigial ``config`` parameter.
    """

    pass


__all__ = ["TrainsideEngineConfig"]
