"""RDMA HCA discovery for Mooncake."""

from __future__ import annotations

import functools
import os
from typing import List

_IB_CLASS_DIR = "/sys/class/infiniband"


@functools.lru_cache(maxsize=1)
def list_rdma_bonds() -> List[str]:
    """All ``mlx5_bond_*`` devices under sysfs, sorted."""
    if not os.path.isdir(_IB_CLASS_DIR):
        return []
    entries = sorted(os.listdir(_IB_CLASS_DIR))
    bonds = [n for n in entries if n.startswith("mlx5_bond_")]
    return bonds if bonds else entries


__all__ = ["list_rdma_bonds"]
