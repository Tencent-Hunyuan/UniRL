"""Miscellaneous utilities for the distributed controller."""

from __future__ import annotations

import socket
from dataclasses import fields as dc_fields
from typing import Tuple, Type

import torch

from unirl.distributed.tensor.batch import Batch

_CUDA_IPC_HANDLE_BYTES = 66


def cuda_ipc_needs_clone(storage: torch.UntypedStorage) -> tuple:
    """Return (ipc_handle, needs_clone) for a CUDA storage."""
    handle = storage._share_cuda_()
    return handle, len(handle[1]) != _CUDA_IPC_HANDLE_BYTES


class Broadcast:
    """Mark a value as broadcast — it will NOT be split across workers."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return f"Broadcast({self.value!r})"


def collect_leaves(x, leaf_type: Type) -> list:
    """Depth-first collect all instances of leaf_type from an arbitrary structure."""
    result = []
    if isinstance(x, leaf_type):
        result.append(x)
    elif isinstance(x, Batch):
        for f in sorted(dc_fields(x), key=lambda f: f.name):
            v = getattr(x, f.name)
            if v is not None:
                result.extend(collect_leaves(v, leaf_type))
    elif isinstance(x, dict):
        for k in sorted(x.keys()):
            result.extend(collect_leaves(x[k], leaf_type))
    elif isinstance(x, (list, tuple)):
        for v in x:
            result.extend(collect_leaves(v, leaf_type))
    return result


def get_open_port() -> int:
    """Find an available TCP port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_node_ip_and_port(pg, bundle_index: int = 0) -> Tuple[str, int]:
    """Get IP and an open port on the node where a PG bundle landed."""
    import ray
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    @ray.remote(num_cpus=0)
    class _Probe:
        def info(self):
            ip = ray.util.get_node_ip_address()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                port = s.getsockname()[1]
            return ip, port

    probe = _Probe.options(
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=bundle_index,
        ),
    ).remote()
    result = ray.get(probe.info.remote())
    ray.kill(probe)
    return result


def get_node_ip(pg, bundle_index: int = 0) -> str:
    """Get IP of the node where a PG bundle landed."""
    ip, _ = get_node_ip_and_port(pg, bundle_index)
    return ip
