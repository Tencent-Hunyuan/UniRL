"""CUDA-IPC tensor codec for the managed data plane (parent side).

Wire format (version 1): base64(pickle((rebuild_fn, args))) exactly as
produced by ``torch.multiprocessing.reductions.reduce_tensor`` — the same
mechanism torch itself uses to share CUDA tensors between processes. The
handle is a few dozen bytes; the storage never moves. Constraints enforced
by the handshake, not here: both processes on the same physical device,
IPC-compatible torch builds, tensors from the regular cudaMalloc pool
(expandable_segments and cumem/VMM regions are not exportable).

Lifetime: the PRODUCER must keep the source tensor alive until the consumer
is done with it — the parent retains request tensors until the response
arrives, the child retains gradient tensors until the parent acks (next
request or explicit release). Consumers should ``.clone()`` if they need
the data past that window.

A byte-identical twin of this module lives on the child side
(``reward_service/tensor_ipc.py``): the two ends of the wire share no Python
package, so the codec is duplicated rather than imported across the
environment boundary. Keep both copies in sync.
"""

from __future__ import annotations

import base64
import pickle

import torch


def encode_tensor(tensor: torch.Tensor) -> str:
    """Share a CUDA tensor and return the printable handle blob."""
    if not tensor.is_cuda:
        raise ValueError("cuda_ipc data plane requires CUDA tensors")
    from torch.multiprocessing.reductions import reduce_tensor

    return base64.b64encode(pickle.dumps(reduce_tensor(tensor))).decode("ascii")


def decode_tensor(blob: str) -> torch.Tensor:
    """Open a handle blob produced by :func:`encode_tensor` in another process."""
    rebuild, args = pickle.loads(base64.b64decode(blob.encode("ascii")))
    return rebuild(*args)


def ipc_fingerprint() -> dict:
    """This process's IPC-compatibility facts, exchanged during handshake."""
    dev = torch.cuda.current_device() if torch.cuda.is_available() else None
    props = torch.cuda.get_device_properties(dev) if dev is not None else None
    uuid = str(getattr(props, "uuid", "")) if props is not None else ""
    try:
        allocator = torch.cuda.memory._get_allocator_backend()
    except Exception:
        allocator = "unknown"
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "device_uuid": uuid,
        "allocator": allocator,
    }


def ipc_compatible(mine: dict, theirs: dict) -> tuple[bool, str]:
    """Decide whether two fingerprints may share a cuda_ipc data plane."""
    if not mine.get("device_uuid") or not theirs.get("device_uuid"):
        return False, "no CUDA device on one side"
    if mine["device_uuid"] != theirs["device_uuid"]:
        return False, f"different devices: {mine['device_uuid']} vs {theirs['device_uuid']}"
    mv, tv = mine.get("torch", ""), theirs.get("torch", "")
    if mv.split("+")[0] != tv.split("+")[0]:
        return False, f"torch version skew: {mv} vs {tv}"
    for side, fp in (("local", mine), ("peer", theirs)):
        if fp.get("allocator") == "cudaMallocAsync":
            return False, f"{side} allocator {fp['allocator']} is not IPC-exportable"
    return True, "ok"
