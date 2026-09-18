"""In-process rollout engine adapter for direct-sampling mode."""

from unirl.rollout.engine.trainside.config import TrainsideEngineConfig
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine

__all__ = ["TrainsideEngineConfig", "TrainsideRolloutEngine"]
