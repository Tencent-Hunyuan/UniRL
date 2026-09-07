"""Update-level sample ordering composed with a micro-batch planner."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner.types import MicroPlanner, Plan
from unirl.types.sample import Part

Arrangement = Tuple[Part, Plan]


class UpdatePlanner:
    """Arrange update membership, then delegate micro-batching to ``micro_planner``."""

    def __init__(
        self,
        micro_planner: MicroPlanner,
        *,
        shuffle_updates: bool = False,
        shuffle_seed: int | None = None,
    ) -> None:
        self.micro_planner = micro_planner
        self.shuffle_updates = bool(shuffle_updates)
        self.shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else 0
        self._shuffle_count = 0

    def validate(self, algorithm: StageAlgorithm) -> None:
        self.micro_planner.validate(algorithm)

    def arrange(self, part: Part, *, num_updates: int, micro_batch_size: int) -> Arrangement:
        """Arrange one part, shuffling before contiguous update partitioning when enabled."""
        return self.arrange_many(
            (part,),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
        )[0]

    def arrange_many(
        self,
        parts: Sequence[Part],
        *,
        num_updates: int,
        micro_batch_size: int,
    ) -> Tuple[Arrangement, ...]:
        """Arrange aligned parts with the same per-rollout permutation seed."""
        ordered = tuple(parts)
        should_shuffle = bool(ordered) and self.shuffle_updates and int(num_updates) > 1
        if should_shuffle:
            seed = (self.shuffle_seed + self._shuffle_count) & 0xFFFFFFFF
            permutations = {
                size: self._permutation(size, seed=seed) for size in {int(part.batch_size) for part in ordered}
            }
            ordered = tuple(
                part if int(part.batch_size) <= 1 else part.select(permutations[int(part.batch_size)])
                for part in ordered
            )
        arrangements = tuple(
            self.micro_planner.arrange(
                part,
                num_updates=num_updates,
                micro_batch_size=micro_batch_size,
            )
            for part in ordered
        )
        if should_shuffle:
            self._shuffle_count += 1
        return arrangements

    @staticmethod
    def _permutation(total_size: int, *, seed: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(total_size, generator=generator)
