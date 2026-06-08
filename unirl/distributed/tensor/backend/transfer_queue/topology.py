"""RDMA HCA discovery for Mooncake.

Walks ``/sys/class/infiniband`` to enumerate available ``mlx5_bond_*`` devices.
Mooncake's ``MC_ENABLE_DEST_DEVICE_AFFINITY=1`` mode then picks the PIX-distance
HCA per process from the active CUDA context — we don't compute PCIe topology
ourselves.

See ``LIN-186/docs/mooncake_-800_diagnosis.md`` for the empirical work behind
this design (probe G6: 3200/3200 with full HCA list + affinity flag).
"""

from __future__ import annotations

import functools
import os
from typing import List

_IB_CLASS_DIR = "/sys/class/infiniband"


@functools.lru_cache(maxsize=1)
def list_rdma_bonds() -> List[str]:
    """All ``mlx5_bond_*`` devices under sysfs, sorted.

    Falls back to any IB device when no bonds exist (rare; normally bonds are
    the canonical RDMA interface on H20/H100 boxes). Returns ``[]`` when sysfs
    has no InfiniBand at all (the caller should raise a clear error since
    Mooncake won't function in that environment).
    """
    if not os.path.isdir(_IB_CLASS_DIR):
        return []
    entries = sorted(os.listdir(_IB_CLASS_DIR))
    bonds = [n for n in entries if n.startswith("mlx5_bond_")]
    return bonds if bonds else entries


__all__ = ["list_rdma_bonds"]
