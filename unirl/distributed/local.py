"""Local-storage access for tensors that may be distributed."""

from __future__ import annotations

from torch import Tensor


def local_view(tensor: Tensor) -> Tensor:
    """Return a DTensor's local shard, or ``tensor`` itself when not distributed."""
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    return tensor


__all__ = ["local_view"]
