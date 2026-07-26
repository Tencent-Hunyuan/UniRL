"""In-process rollout engine adapter for direct-sampling mode.

Exposes a materialized ``models`` ``Pipeline`` as a
:class:`unirl.rollout.engine.base.BaseRolloutEngine`. Used when the
training Policy itself is the sampler (direct sampling, on-policy RL) —
the rollout runs in the same Ray actor / Python process / GPU as
training, so no worker subprocess and no weight sync are needed.

Selected by pointing a recipe's ``rollout`` block at
``TrainsideRolloutEngine``. Living in this package is what marks the engine
direct-sampling for :func:`unirl.config.contracts.is_direct_sampling`, which
gates the recipe contracts that follow from it (no ``sync`` section, no
offload, no separate device slab).
"""

from unirl.rollout.engine.trainside.config import TrainsideEngineConfig
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine

__all__ = ["TrainsideEngineConfig", "TrainsideRolloutEngine"]
