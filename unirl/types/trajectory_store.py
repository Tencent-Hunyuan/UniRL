"""
Trajectory — compact trajectory storage as a Batch dataclass.
TrajectoryBuilder — builder for collecting latents during denoising loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set, Tuple

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field, shared_field


def compute_trajectory_positions(sde_indices: Set[int], num_steps: int) -> List[int]:
    """Return sorted positions needed for ``(x_t, x_{t+1})`` pairs at SDE boundaries.

    For each SDE step index ``i`` in *sde_indices*, both position ``i`` and
    ``i + 1`` are required.  Results are clamped to ``[0, num_steps]``.

    >>> compute_trajectory_positions({0, 2, 4}, 5)
    [0, 1, 2, 3, 4, 5]
    >>> compute_trajectory_positions({3}, 5)
    [3, 4]
    """
    positions: Set[int] = set()
    for i in sde_indices:
        positions.add(max(0, min(i, num_steps)))
        positions.add(max(0, min(i + 1, num_steps)))
    return sorted(positions)


@dataclass
class Trajectory(Batch):
    """Compact trajectory storage with index map.

    Attributes:
        data: Dense trajectory tensor [B, K, ...] where K <= T+1
            (only collected positions).
        index_map: 1-D LongTensor of size ``total_positions`` where
            ``index_map[i]`` gives the compact index into ``data``
            for original position ``i``, or -1 if not stored.
        total_positions: Total number of positions in the full
            trajectory (T+1).
    """

    data: torch.Tensor = field(kind=FieldKind.CONCAT)
    index_map: torch.Tensor = shared_field()
    total_positions: int = shared_field()

    # ---- direct factories ---------------------------------------------------

    @classmethod
    def from_full(cls, trajectories: torch.Tensor) -> Trajectory:
        """Wrap full trajectories [B, T+1, ...] with an identity index map."""
        t_plus_1 = int(trajectories.shape[1])
        index_map = torch.arange(t_plus_1, dtype=torch.long)
        return cls(data=trajectories, index_map=index_map, total_positions=t_plus_1)

    @classmethod
    def from_clean_latents(
        cls,
        clean_latents: torch.Tensor,
        total_positions: int = 1,
    ) -> Trajectory:
        """DiffusionNFT path: store only clean latents as a single-position trajectory."""
        data = clean_latents.unsqueeze(1)  # [B, 1, ...]
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        index_map[total_positions - 1] = 0
        return cls(data=data, index_map=index_map, total_positions=total_positions)

    @classmethod
    def from_selective(
        cls,
        trajectories: torch.Tensor,
        collected_positions: List[int],
        total_positions: int,
    ) -> Trajectory:
        """Selective storage: K positions out of T+1."""
        if int(trajectories.shape[1]) != len(collected_positions):
            raise ValueError(
                f"trajectories dim-1 ({trajectories.shape[1]}) != len(collected_positions) ({len(collected_positions)})"
            )
        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, orig_pos in enumerate(collected_positions):
            if 0 <= orig_pos < total_positions:
                index_map[orig_pos] = compact_idx
        return cls(data=trajectories, index_map=index_map, total_positions=total_positions)

    # ---- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return int(self.data.shape[0])

    @property
    def num_stored(self) -> int:
        return int(self.data.shape[1])

    @property
    def device(self) -> torch.device:
        return self.data.device

    @property
    def is_full(self) -> bool:
        return self.num_stored == self.total_positions

    @property
    def is_clean_latents_only(self) -> bool:
        return self.num_stored == 1

    @property
    def is_selective(self) -> bool:
        return not self.is_full and not self.is_clean_latents_only

    @property
    def clean_latents(self) -> torch.Tensor:
        last_stored = int((self.index_map >= 0).nonzero(as_tuple=False)[-1].item())
        compact_idx = int(self.index_map[last_stored].item())
        return self.data[:, compact_idx]

    # ---- position access ----------------------------------------------------

    def has_position(self, pos: int) -> bool:
        if pos < 0 or pos >= self.total_positions:
            return False
        return int(self.index_map[pos].item()) >= 0

    def get_position(self, pos: int) -> torch.Tensor:
        if pos < 0 or pos >= self.total_positions:
            raise IndexError(f"Position {pos} out of range [0, {self.total_positions})")
        compact_idx = int(self.index_map[pos].item())
        if compact_idx < 0:
            raise IndexError(f"Position {pos} was not collected. Stored positions: {self.stored_positions}")
        return self.data[:, compact_idx]

    def get_pair(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_position(pos), self.get_position(pos + 1)

    @property
    def stored_positions(self) -> List[int]:
        return sorted(int(i) for i in range(self.total_positions) if int(self.index_map[i].item()) >= 0)

    # ---- batch ops (Batch provides concat/select/slice via fields) --------

    def select(self, indices: torch.Tensor) -> Trajectory:
        """Select samples by index along the batch dimension."""
        return Trajectory(
            data=self.data.index_select(0, indices.to(self.data.device)),
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def slice_batch(self, start: int, end: int) -> Trajectory:
        return Trajectory(
            data=self.data[start:end].clone(),
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def index_select_batch(self, idx: torch.Tensor) -> Trajectory:
        return self.select(idx)

    def reindex_batch(self, indices: torch.Tensor) -> Trajectory:
        return Trajectory(
            data=self.data[indices],
            index_map=self.index_map,
            total_positions=self.total_positions,
        )

    def cast_dtype(self, dtype: torch.dtype) -> Trajectory:
        if self.data.is_floating_point() and self.data.dtype != dtype:
            return Trajectory(
                data=self.data.to(dtype=dtype),
                index_map=self.index_map,
                total_positions=self.total_positions,
            )
        return self

    # ---- modality detection -------------------------------------------------

    def detect_modality(self) -> str:
        if self.is_full:
            return "video" if int(self.data.ndim) >= 6 else "image"
        return "video" if int(self.data.ndim) >= 5 else "image"


class TrajectoryBuilder:
    """Builder for collecting latents during denoising loops.

    Usage::

        builder = TrajectoryBuilder.for_sde_steps(sde_indices, num_steps)
        builder.add(0, initial_latents)
        for i in range(num_steps):
            ...
            builder.add(i + 1, latents)
        trajectory = builder.finalize()
    """

    def __init__(self, total_steps: int, needed_positions: Set[int]) -> None:
        self._total_steps = total_steps
        self._needed = needed_positions
        self._collected: List[Tuple[int, torch.Tensor]] = []

    @classmethod
    def for_sde_steps(cls, sde_indices: Set[int], total_steps: int) -> TrajectoryBuilder:
        """Create a builder that keeps only positions needed for SDE pairs."""
        needed = set(compute_trajectory_positions(sde_indices, total_steps))
        if not needed:
            needed = {total_steps}
        return cls(total_steps, needed)

    @classmethod
    def full(cls, total_steps: int) -> TrajectoryBuilder:
        """Create a builder that keeps all positions."""
        return cls(total_steps, set(range(total_steps + 1)))

    def add(self, position: int, latents: torch.Tensor) -> None:
        """Record latents at *position*. Silently drops unneeded positions."""
        if position in self._needed:
            self._collected.append((position, latents))

    def finalize(self) -> Trajectory:
        """Freeze collected latents into a Trajectory."""
        if not self._collected:
            raise ValueError(f"finalize() called with no collected positions. needed={sorted(self._needed)}")
        positions = [p for p, _ in self._collected]
        data = torch.stack([t for _, t in self._collected], dim=1)
        total_positions = self._total_steps + 1

        index_map = torch.full((total_positions,), -1, dtype=torch.long)
        for compact_idx, orig_pos in enumerate(positions):
            if 0 <= orig_pos < total_positions:
                index_map[orig_pos] = compact_idx

        return Trajectory(
            data=data,
            index_map=index_map,
            total_positions=total_positions,
        )


# Backward compat alias
TrajectoryStore = Trajectory

__all__ = ["Trajectory", "TrajectoryBuilder", "TrajectoryStore", "compute_trajectory_positions"]
