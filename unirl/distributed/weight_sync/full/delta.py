"""Abstract shard-local delta encoder for sparse full-weight sync."""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparseDelta:
    """Changed positions of one shard: ``indices`` int32 flat offsets, ``values`` the new elements."""

    indices: torch.Tensor  # [nnz] int32 flat offsets into the shard
    values: torch.Tensor  # [nnz] matching the shard dtype


class DeltaWeightEncoder(abc.ABC):
    """Diffs each FSDP shard against a pinned snapshot into sparse deltas for shard-local sync."""

    @abc.abstractmethod
    def seed(self, name: str, shard: torch.Tensor) -> None:
        """Record the first-sync dense baseline for ``name`` (no delta emitted)."""
        raise NotImplementedError

    @abc.abstractmethod
    def encode(self, name: str, shard: torch.Tensor) -> SparseDelta:
        """Bit-exact diff ``shard`` against its snapshot, refresh the snapshot, return changed positions."""
        raise NotImplementedError


__all__ = ["SparseDelta", "DeltaWeightEncoder"]
