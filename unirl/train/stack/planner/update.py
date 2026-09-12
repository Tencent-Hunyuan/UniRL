"""Update-level sample ordering composed with a micro-batch planner."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner.types import MicroPlanner, Plan
from unirl.types.sample import Part
from unirl.types.sample_id import parent_id

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

    def validate(self, algorithm: StageAlgorithm) -> None:
        self.micro_planner.validate(algorithm)

    def arrange(
        self,
        part: Part,
        *,
        num_updates: int,
        micro_batch_size: int,
        shuffle_step: int | None = None,
    ) -> Arrangement:
        """Arrange one part, shuffling before contiguous update partitioning when enabled."""
        return self.arrange_many(
            (part,),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
            shuffle_step=shuffle_step,
        )[0]

    def arrange_many(
        self,
        parts: Sequence[Part],
        *,
        num_updates: int,
        micro_batch_size: int,
        shuffle_step: int | None = None,
    ) -> Tuple[Arrangement, ...]:
        """Arrange lineage-aligned parts with one permutation rooted at the coarsest part."""
        ordered = tuple(parts)
        should_shuffle = bool(ordered) and self.shuffle_updates and int(num_updates) > 1
        if should_shuffle:
            if shuffle_step is None:
                raise ValueError("shuffle_updates=True requires a stable shuffle_step (for example, rollout_id).")
            seed = (self.shuffle_seed + int(shuffle_step)) & 0xFFFFFFFF
            ordered = tuple(
                part.select(permutation)
                for part, permutation in zip(ordered, self._aligned_permutations(ordered, seed=seed))
            )
        return tuple(
            self.micro_planner.arrange(
                part,
                num_updates=num_updates,
                micro_batch_size=micro_batch_size,
            )
            for part in ordered
        )

    @classmethod
    def _aligned_permutations(cls, parts: Sequence[Part], *, seed: int) -> Tuple[torch.Tensor, ...]:
        sizes = tuple(int(part.batch_size) for part in parts)
        coarse_index = min(range(len(parts)), key=sizes.__getitem__)
        coarse = parts[coarse_index]
        coarse_size = sizes[coarse_index]
        if coarse_size < 1:
            raise ValueError("Cannot shuffle an empty Part.")
        coarse_permutation = cls._permutation(coarse_size, seed=seed)

        permutations = []
        for part, size in zip(parts, sizes):
            if size % coarse_size:
                raise ValueError(
                    "Aligned update shuffle requires each Part batch size to be an integer multiple "
                    f"of the smallest size ({coarse_size}); got {sizes}."
                )
            fanout = size // coarse_size
            if part is not coarse:
                cls._validate_lineage_expansion(coarse, part, fanout=fanout)
            offsets = torch.arange(fanout)
            permutations.append((coarse_permutation[:, None] * fanout + offsets).reshape(-1))
        return tuple(permutations)

    @staticmethod
    def _validate_lineage_expansion(coarse: Part, expanded: Part, *, fanout: int) -> None:
        if len(coarse.sample_ids) != int(coarse.batch_size) or len(expanded.sample_ids) != int(expanded.batch_size):
            raise ValueError("Aligned update shuffle across unequal Part sizes requires sample_ids.")
        for index, ancestor in enumerate(coarse.sample_ids):
            descendants = expanded.sample_ids[index * fanout : (index + 1) * fanout]
            if not all(UpdatePlanner._is_descendant(sample_id, ancestor) for sample_id in descendants):
                raise ValueError(
                    "Aligned update shuffle requires each larger Part to be a contiguous lineage expansion "
                    f"of the smallest Part; {descendants!r} are not all descendants of {ancestor!r}."
                )

    @staticmethod
    def _is_descendant(sample_id: str, ancestor_id: str) -> bool:
        current = sample_id
        while current != ancestor_id:
            parent = parent_id(current)
            if parent is None:
                return False
            current = parent
        return True

    @staticmethod
    def _permutation(total_size: int, *, seed: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(seed)
        return torch.randperm(total_size, generator=generator)
