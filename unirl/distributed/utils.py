"""Miscellaneous utilities for the distributed controller.

Network helpers used by DevicePool and RoleProxy, plus the ``Broadcast``
dispatch marker and the generic ``collect_leaves`` tree walker.
"""

from __future__ import annotations

import socket
from dataclasses import fields as dc_fields
from typing import Tuple, Type

import torch

from unirl.distributed.tensor.batch import Batch

# ── CUDA IPC compatibility ──

# cudaMalloc-backed IPC handles always have a 66-byte device handle (CUDA_IPC_HANDLE_SIZE=64
# + 2 bytes padding).  Expandable-segments (VMM) allocations produce shorter handles
# because they use pidfd_getfd for cross-process sharing instead of the legacy IPC path.
_CUDA_IPC_HANDLE_BYTES = 66


def cuda_ipc_needs_clone(storage: torch.UntypedStorage) -> tuple:
    """Return (ipc_handle, needs_clone) for a CUDA storage.

    needs_clone is True when the storage was allocated by the expandable-segments
    (VMM) allocator: its IPC handle is not a standard cudaMalloc handle and cannot
    be opened by other processes on kernels lacking pidfd_getfd (Linux < 5.6).
    The caller should clone to a cudaMalloc-backed allocation before sharing.

    Returns the probe handle so the caller avoids a second _share_cuda_() call.
    """
    handle = storage._share_cuda_()
    return handle, len(handle[1]) != _CUDA_IPC_HANDLE_BYTES


# ── Broadcast wrapper ──


class Broadcast:
    """Mark a value as broadcast — it will NOT be split across workers.

    Usage::

        role.forward(images, lr=Broadcast(0.01), config=Broadcast(cfg_dict))

    Dispatch infers the batch size from the first batch-axis field it finds
    (see ``pytree.infer_batch_size``) and splits every field whose leading
    dim matches it, replicating the rest. Wrap a value in ``Broadcast(...)``
    to force it to be replicated rather than split — e.g. per-rollout
    metadata whose leading dim happens to coincide with the batch size.
    """

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return f"Broadcast({self.value!r})"


# ── Generic tree traversal ──


def collect_leaves(x, leaf_type: Type) -> list:
    """Depth-first collect all instances of leaf_type from an arbitrary structure.

    Traversal rules:
      - isinstance(x, leaf_type) -> append and stop recursing
      - Batch / dict -> recurse into values, keys sorted alphabetically
      - list / tuple  -> recurse in positional order
      - everything else -> skip

    Sorting dict/Batch keys ensures deterministic ordering across call sites,
    which is critical for aligning controller-side collect_leaves(TensorRef)
    with worker-side collect_leaves(torch.Tensor) by index.
    """
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


# ── Network helpers ──


def get_open_port() -> int:
    """Find an available TCP port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_node_ip_and_port(pg, bundle_index: int = 0) -> Tuple[str, int]:
    """Get IP and an open port on the node where a PG bundle landed.

    Both values come from the same worker node, avoiding
    controller-vs-worker mismatch. Uses a lightweight probe actor.
    """
    import ray
    from ray.util.placement_group import PlacementGroupSchedulingStrategy

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
