"""AgenticEnvTrainer — agentic multi-turn RL with ENV-SOURCED rewards (LIN-519).

A thin subclass of :class:`~unirl.trainer.agentic.AgenticTrainer` for
interactive ENVIRONMENTS (ALFWorld, Sokoban, WebShop, …) where the reward is the
environment's return per trajectory — the simulator's task-success / shaped signal,
already attached to each trajectory's last generated Part by the
:class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine` — not a grade of a
final ``<answer>``.

Only the reward SOURCE differs from the base ``AgenticTrainer``: this reads the
per-trajectory reward off the frontier instead of building a scoring Sample and
calling the reward backend (``RewardService`` even rejects precomputed rewards). The
GRPO tail — group-relative advantage → ``Part.concat`` of every assistant turn → one
on-policy ``train_track`` — is inherited unchanged, so ``ratio≈1`` and the
learn-from-all-turns behavior hold exactly as validated for the deep-research task.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from unirl.distributed.tensor import hydrate
from unirl.trainer.agentic import AgenticTrainer
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class AgenticEnvTrainer(AgenticTrainer):
    """Agentic trainer whose reward is the environment's per-trajectory return.

    The recipe still carries a (built-but-unused) ``reward`` backend so the shared
    ``ARTrainer`` construction path is happy; this trainer never calls it.
    """

    def _rewards_and_groups(
        self, sample: Sample, trajs: List[Sample], rollout_id: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """Read the env-sourced scalar reward the engine attached to each
        trajectory's last generated Part; group by the shared root id. No reward
        backend, no scoring Sample."""
        del sample, rollout_id  # env reward is already on the trajectories
        values: List[float] = []
        group_ids: List[str] = []
        missing = 0
        for tr in trajs:
            gens = tr.gen_parts()
            reward: Optional[torch.Tensor] = gens[-1].rewards if gens else None
            if reward is not None:
                values.append(float(hydrate(reward).to(torch.float32).flatten()[0].item()))
            else:
                # Gen-less trajectory = engine-marked failure (the fault hit before the
                # first turn). NaN, not 0.0, so _group_advantages drops it from the
                # group's mean/std and gives it zero advantage — scoring an
                # infrastructure fault as a genuine miss biases every sibling.
                values.append(float("nan"))
                missing += 1
            group_ids.append(tr.parts[0].sample_ids[0])
        if missing:
            logger.warning(
                "AgenticEnvTrainer: %d/%d trajectories had no env reward (marked failed).",
                missing,
                len(trajs),
            )
        return torch.tensor(values, dtype=torch.float32), group_ids


__all__ = ["AgenticEnvTrainer"]
