"""AsyncAgenticEnvTrainer — fully-async agentic RL with ENV-SOURCED rewards (LIN-531).

The disaggregated producer/consumer sibling of
:class:`~unirl.trainer.agentic_env.AgenticEnvTrainer` — the same relation
:class:`~unirl.trainer.agentic_async.AsyncAgenticTrainer` has to the barrier
:class:`~unirl.trainer.agentic.AgenticTrainer`. For interactive environments
(ALFWorld, Sokoban, WebShop, …) where the reward is the simulator's per-trajectory
return rather than a grade of a final ``<answer>``.

Only the reward SOURCE differs from ``AsyncAgenticTrainer``, and it comes from the
shared :class:`~unirl.trainer.agentic_env._EnvRewardSource` mixin. Everything else —
disaggregated slabs, the resident agentic drive, ``NCCLWeightSync``, the bounded
staleness buffer, and the turn-boundary partial-rollout quiesce — is inherited
unchanged, so the variable-depth ALFWorld trajectories are exactly the workload the
partial-rollout overlap targets.
"""

from __future__ import annotations

import logging

from unirl.trainer.agentic_async import AsyncAgenticTrainer
from unirl.trainer.agentic_env import _EnvRewardSource

logger = logging.getLogger(__name__)


class AsyncAgenticEnvTrainer(_EnvRewardSource, AsyncAgenticTrainer):
    """Fully-async agentic trainer whose reward is the environment's per-trajectory return.

    The recipe still carries a (built-but-unused) ``reward`` backend so the shared
    construction path is happy; this trainer never calls it. The reconstructed
    answer-request that ``AsyncAgenticTrainer._train_on_groups`` passes to
    ``_rewards_and_groups`` is ignored (env reward rides the trajectories).
    """


__all__ = ["AsyncAgenticEnvTrainer"]
