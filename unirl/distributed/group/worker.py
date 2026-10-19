"""Worker — physical GPU Ray actor."""

from __future__ import annotations

import logging
import os
import socket
from typing import Dict, Optional

import ray
import torch
from torch import Tensor

from unirl.distributed.group.remote import RankInfo, Remote
from unirl.distributed.tensor import TensorRef, TensorTransport, TensorTransportRuntime, map_tree
from unirl.distributed.tensor.factory import build_transport
from unirl.distributed.utils import collect_leaves

logger = logging.getLogger(__name__)


class Worker:
    """Physical worker: one per GPU slot."""

    def __init__(
        self,
        device_id: int,
        slot: int = 0,
        nccl_rank: Optional[int] = None,
        world_size: int = 1,
        transport_kind: str = "colocate_store",
        tq_handoff: Optional[dict] = None,
    ) -> None:
        """Ray remote actor entry point. Sets up the device and the transport."""
        self.device_id = device_id
        self.slot = slot
        self.nccl_rank = nccl_rank
        self.world_size = world_size
        self.transport_kind = transport_kind or "colocate_store"

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        from unirl.utils.memory_utils import init_process_snapshot_sampler

        init_process_snapshot_sampler(rank=nccl_rank if nccl_rank is not None else device_id)

        self.worker_id = f"dw{device_id}" if slot == 0 else f"dw{device_id}_s{slot}"

        self.tw = None
        self.tq_handoff = tq_handoff
        self.transport: Optional[TensorTransport] = None

        self._roles: Dict[str, Remote] = {}
        self._reserved_sockets: Dict[int, socket.socket] = {}

        if self.transport_kind in ("colocate_store", "colocate", "transfer_queue", "tq"):
            self.build_and_install_transport()

    def _init_local(self, device_id: int = 0, slot: int = 0, transport=None) -> None:
        """Initialize without GPU/Ray for unit testing."""
        self.device_id = device_id
        self.slot = slot
        self.device = "cpu"
        self.nccl_rank = 0
        self.world_size = 1
        self.transport_kind = "colocate_store"
        self.worker_id = f"dw{device_id}" if slot == 0 else f"dw{device_id}_s{slot}"
        self.tw = None
        self.tq_handoff = None
        self._roles = {}
        self._reserved_sockets = {}
        if transport is None:
            self.build_and_install_transport()
        else:
            self._install_transport(transport)

    def set_tensor_worker(self, tw_handle) -> None:
        """Inject the per-GPU TensorWorker actor handle (gpu backend). Called by DevicePool."""
        self.tw = tw_handle

    def build_and_install_transport(self):
        """Build the configured transport and install it as the process backend."""
        self._install_transport(
            build_transport(
                self.transport_kind,
                worker_id=self.worker_id,
                device=self.device,
                device_id=self.device_id,
                tw=self.tw,
                tq_handoff=self.tq_handoff,
                global_rank=self.nccl_rank,
                world_size=self.world_size,
            )
        )

    def _install_transport(self, transport: TensorTransport) -> None:
        """Install the Worker's transport as the process backend."""
        self.transport = transport
        TensorTransportRuntime.install(transport)

    def reset_zero_copy_buffer_free(self) -> None:
        """Reclaim this process's mooncake zero-copy buffer free-lists (per-rollout)."""
        from unirl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime

        rt = TransferQueueRuntime.current()
        if rt is not None:
            rt.reset_zero_copy_buffer_free()

    def _reserve_port(self) -> int:
        """Bind a socket to an ephemeral port and hold it open."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", 0))
        port = s.getsockname()[1]
        self._reserved_sockets[port] = s
        return port

    def _release_port(self, port: int) -> None:
        """Release a previously reserved port so init_process_group can use it."""
        s = self._reserved_sockets.pop(port, None)
        if s:
            s.close()

    def add_remote(
        self, role_name: str, role_cls, rank_info: RankInfo, init_kwargs: dict = None, dist_env: dict = None
    ) -> None:
        """Register a logical worker role on this device."""
        resolved_kwargs = self._resolve_init_kwargs(init_kwargs or {})
        role = role_cls(**resolved_kwargs)
        role.setup(
            transport=self.transport,
            device=self.device,
            rank_info=rank_info,
            dist_env=dist_env,
            get_sibling=lambda name: self._roles[name],
        )
        self._roles[role_name] = role

    def _resolve_init_kwargs(self, obj):
        """Resolve HandleRefs and nested ``_target_`` blocks for one kwarg tree."""
        from hydra.utils import get_method

        from unirl.distributed.group.handle import HandleRef

        if isinstance(obj, HandleRef):
            try:
                return self._roles[obj.role_name]
            except KeyError:
                raise RuntimeError(
                    f"Cannot resolve sibling Handle '{obj.role_name}' on "
                    f"Worker {self.worker_id}: not registered on this "
                    f"Worker. Likely cause: the sibling lives on a different "
                    f"device slab (separate placement scope) or a different slot."
                )
        if isinstance(obj, dict):
            children = {k: self._resolve_init_kwargs(v) for k, v in obj.items() if k != "_target_"}
            if "_target_" in obj:
                cls = get_method(obj["_target_"])
                return cls(**children)
            return children
        if isinstance(obj, list):
            return [self._resolve_init_kwargs(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._resolve_init_kwargs(v) for v in obj)
        return obj

    def get_rank_info(self, role_name: str) -> RankInfo:
        """Read back rank_info (may have been modified by initialize)."""
        return self._roles[role_name].rank_info

    def call(self, role_name: str, method_name: str, args: tuple, kwargs: dict, grad_mode: bool = False, call_id=None):
        """Generic RPC entry point."""
        role = self._roles[role_name]

        in_metas = self._collect(args, TensorRef) + self._collect(kwargs, TensorRef)
        fetched = self.transport.get_batch({str(i): m for i, m in enumerate(in_metas)})
        in_iter = iter(fetched[str(i)] for i in range(len(in_metas)))

        def resolve(o):
            return next(in_iter) if isinstance(o, TensorRef) else o

        resolved_args = map_tree(args, resolve)
        resolved_kwargs = map_tree(kwargs, resolve)

        if grad_mode:
            tensors = [fetched[str(i)] for i in range(len(in_metas))]
            for t in tensors:
                t.requires_grad_(True)
                t.retain_grad()
            role._grad_inputs[call_id] = tensors

        result = getattr(role, method_name)(*resolved_args, **resolved_kwargs)

        if grad_mode:
            role._grad_outputs[call_id] = collect_leaves(result, Tensor)

        out_tensors = self._collect(result, Tensor)
        stored = self.transport.put_batch({str(i): t for i, t in enumerate(out_tensors)})
        out_iter = iter(stored[str(i)] for i in range(len(out_tensors)))

        def pack(o):
            return next(out_iter) if isinstance(o, Tensor) else o

        return map_tree(result, pack)

    def _collect(self, obj, leaf_type) -> list:
        """Collect leaves of leaf_type in the SAME order ``map_tree`` visits them."""
        out: list = []

        def visit(o):
            if isinstance(o, leaf_type):
                out.append(o)
            return o

        map_tree(obj, visit)
        return out

    def transport_op(self, method: str, *args, **kwargs):
        """Relay a controller-side call into this Worker's transport."""
        allowed = getattr(type(self.transport), "REMOTE_OPS", frozenset())
        if method not in allowed:
            raise AttributeError(f"transport_op: {method!r} is not a remote-callable transport op")
        return getattr(self.transport, method)(*args, **kwargs)

    def setup_global_pg(self) -> None:
        """Initialize the cross-worker transfer group (slot0 only)."""
        self.transport.setup_transfer(self.nccl_rank, self.world_size)

    def teardown(self) -> None:
        """Release everything this actor owns. Idempotent, best effort."""
        for role_name, role in reversed(list(getattr(self, "_roles", {}).items())):
            closer = getattr(role, "shutdown", None) or getattr(role, "close", None)
            if not callable(closer):
                continue
            try:
                closer()
            except Exception:
                logger.exception("Role %s failed to shut down on worker %s", role_name, self.worker_id)
        if hasattr(self, "_roles"):
            self._roles.clear()

        for port in list(getattr(self, "_reserved_sockets", {})):
            self._release_port(port)

    def get_gpu_count(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def get_cuda_visible_devices(self) -> str:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "")

    def get_node_ip(self) -> str:
        return ray.util.get_node_ip_address()
