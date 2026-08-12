"""In-process rollout engine adapter for direct-sampling mode.

Exposes a materialized ``models`` ``Pipeline`` as a
:class:`unirl.rollout.engine.base.BaseRolloutEngine`. Used when the
training Policy itself is the sampler (direct sampling, on-policy RL) —
the rollout runs in the same Ray actor / Python process / GPU as
training, so no worker subprocess and no weight sync are needed.

Selected via ``cfg.rollout.engine: trainside`` (the registration in
``config.py`` wires ``_target_`` to ``TrainsideRolloutEngine``). The
presence of this target is the canonical signal for
:func:`unirl.config.validation.is_direct_sampling`.
"""

from unirl.rollout.engine.trainside.config import TrainsideEngineConfig
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine

__all__ = ["TrainsideEngineConfig", "TrainsideRolloutEngine"]
