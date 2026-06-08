"""Worker — physical GPU Ray actor.

Each GPU slot runs exactly one Worker. It owns a ``TensorTransport`` (chosen by
config) and hosts multiple ``Remote`` instances (colocated logical workers).

The Worker is transport-agnostic: ``call()`` does the generic arg/result
tree-walk and routes every tensor through ``transport.get_batch`` (resolve) and
``transport.put_batch`` (pack); GC / NCCL / remote-compute RPCs delegate to the
transport. The transport delegates to the underlying store (in-process
``TensorStore`` for colocate, a per-GPU ``TensorWorker`` actor for gpu, the
queue client for transfer_queue).
"""

from __future__ import annotations

import os
import socket
from typing import Dict, Optional

import ray
import torch
from torch import Tensor

from unirl.distributed.group.remote import RankInfo, Remote
from unirl.distributed.tensor.factory import build_transport
from unirl.distributed.tensor.transport import TensorMeta, TensorTransport, TensorTransportRuntime, map_tree
from unirl.distributed.utils import collect_leaves


class Worker:
    """Physical worker: one per GPU slot.

    In Ray mode, this class is wrapped with @ray.remote(num_gpus=...).
    For unit testing, use _init_local() to skip GPU/Ray setup.
    """

    def __init__(
        self,
        device_id: int,
        slot: int = 0,
        nccl_rank: Optional[int] = None,
        world_size: int = 1,
        transport_kind: str = "colocate_store",
        tq_handoff: Optional[dict] = None,
    ) -> None:
        """Ray remote actor entry point. Sets up the device and the transport.

        Args:
            device_id:      Physical GPU index (same for all slots on a GPU).
            slot:           Slot index on this device (0 = primary, 1+ = colocated).
            nccl_rank:      Global NCCL rank for slot0 workers (= device_id); None for slot1+.
            world_size:     Total number of slot0 workers (= num_devices).
            transport_kind: Which TensorTransport backend to install.
            tq_handoff:     Driver's TransferQueue handoff (transfer_queue backend only);
                            consumed by build_transport to bootstrap this process's client.
        """
        self.device_id = device_id
        self.slot = slot
        self.nccl_rank = nccl_rank
        self.world_size = world_size
        self.transport_kind = transport_kind or "colocate_store"

        # GPU setup: Ray PlacementGroup sets CUDA_VISIBLE_DEVICES
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.worker_id = f"dw{device_id}" if slot == 0 else f"dw{device_id}_s{slot}"

        # Backend dependencies: tw (gpu) is injected after spawn via
        # set_tensor_worker(); tq_handoff (transfer_queue) arrives here in the
        # constructor from DevicePool. Both are consumed by build_transport.
        self.tw = None
        self.tq_handoff = tq_handoff
        # Typed as the base TensorTransport (matches build_transport's return);
        # _install_transport enforces at runtime that it is a WorkerLocalTransport.
        self.transport: Optional[TensorTransport] = None

        self._roles: Dict[str, Remote] = {}
        self._reserved_sockets: Dict[int, socket.socket] = {}

        # colocate + transfer_queue build immediately — their deps are ready at
        # construction (in-process store / the driver handoff). gpu defers until
        # DevicePool injects the shared per-GPU TensorWorker via set_tensor_worker().
        if self.transport_kind in ("colocate_store", "colocate", "transfer_queue", "tq"):
            self.build_and_install_transport()

    def _init_local(self, device_id: int = 0, slot: int = 0, transport=None) -> None:
        """Initialize without GPU/Ray for unit testing.

        Defaults to an in-process colocate transport on CPU; pass ``transport``
        to inject a custom backend (e.g. InMemoryTransport).
        """
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
        """Build the configured transport and install it as the process backend.

        Runs from __init__ for colocate (in-process store) and transfer_queue
        (the driver handoff arrives via the constructor). gpu_store must run
        after set_tensor_worker(), so DevicePool calls it explicitly there.
        """
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
        # No return: the transport is not Ray-serializable (holds locks / actor
        # handles); DevicePool calls this via RPC and must not receive it.

    def _install_transport(self, transport: TensorTransport) -> None:
        """Install the Worker's transport as the process backend.

        Any TensorTransport works — the Worker is backend-blind (it only uses
        get_batch/put_batch/end_call). Worker-local capabilities (incref/decref,
        NCCL, tensor_op/cat/get_cpu) are reached only by the controller, and only
        when the backend is a WorkerLocalTransport.
        """
        self.transport = transport
        TensorTransportRuntime.install(transport)

    def reset_zero_copy_buffer_free(self) -> None:
        """Reclaim this process's mooncake zero-copy buffer free-lists (per-rollout).

        Delegates to the process-global TransferQueueRuntime installed during
        ``build_transport``; a no-op when TQ is not the active backend (``current()``
        is ``None``) or the backend has no zero-copy buffers. The driver fans this
        across all Workers between rollouts (see
        ``DevicePool.reset_transfer_queue_buffers``) so the registered RDMA buffers
        don't exhaust over a run.
        """
        from unirl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime

        rt = TransferQueueRuntime.current()
        if rt is not None:
            rt.reset_zero_copy_buffer_free()

    # ── Port reservation ──

    def _reserve_port(self) -> int:
        """Bind a socket to an ephemeral port and hold it open.

        The port is guaranteed unique across calls on this actor
        (Ray actor is single-threaded). Call _release_port() before
        the user's initialize() so init_process_group can bind it.
        """
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

    # ── Role management ──

    def add_remote(
        self, role_name: str, role_cls, rank_info: RankInfo, init_kwargs: dict = None, dist_env: dict = None
    ) -> None:
        """Register a logical worker role on this device.

        ``init_kwargs`` is walked depth-first before ``role_cls`` is
        constructed:

        - ``HandleRef`` leaves → local sibling ``Remote`` from
          ``self._roles``.
        - Nested ``dict`` with ``_target_`` → ``hydra.utils.instantiate``
          after children are resolved. This is what makes the driver a
          config router and the worker the materializer — any object
          with ``_target_`` gets constructed in this worker's process
          (its CUDA context, its placement).
        - ``dict`` / ``list`` / ``tuple`` without ``_target_`` →
          recurse, preserving structure.

        Args:
            role_name:   Unique name for this role.
            role_cls:    Remote subclass.
            rank_info:   Parallelism rank info.
            init_kwargs: Dict of kwargs forwarded to role_cls.__init__
                         after resolution.
            dist_env:    Group-level dist env vars (written to os.environ once).
        """
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
        """Resolve HandleRefs and nested ``_target_`` blocks for one kwarg tree.

        Children resolve before parents, so a HandleRef *inside* a
        nested ``_target_`` block becomes the local Remote before the
        enclosing target is called.

        Uses ``hydra.utils.get_method`` + direct ``cls(**children)`` rather
        than ``hydra.utils.instantiate`` so that already-resolved Python
        objects (Remote instances, constructed siblings) pass through as
        kwargs without OmegaConf coercion. Interpolations and structural
        Hydra features are expected to have been resolved driver-side via
        ``OmegaConf.to_container(resolve=True)`` already.

        Raises clearly if a referenced sibling is not registered on this
        Worker (e.g., the user passed a Handle from a different slab/slot).
        """
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

    # ── RPC entry point ──

    def call(self, role_name: str, method_name: str, args: tuple, kwargs: dict, grad_mode: bool = False, call_id=None):
        """Generic RPC entry point.

        Resolves inputs (TensorMeta → Tensor via transport.get_batch) and packs
        outputs (Tensor → TensorMeta via transport.put_batch). Non-tensor
        args/kwargs/results pass through unchanged.

        grad_mode and call_id are dedicated parameters (not via kwargs) so
        dispatch internals remain unaware of grad state. When grad_mode=True:
          - resolved input tensors are marked grad-tracked leaves for backward.
          - output tensors are saved (before detach) for backward.
        Saved under role._grad_inputs[call_id] / role._grad_outputs[call_id].
        """
        role = self._roles[role_name]

        # Resolve: collect TensorMeta leaves (tree order), batch-fetch, substitute.
        # Keys are positional indices so get_batch results align with the walk.
        in_metas = self._collect(args, TensorMeta) + self._collect(kwargs, TensorMeta)
        fetched = self.transport.get_batch({str(i): m for i, m in enumerate(in_metas)})
        in_iter = iter(fetched[str(i)] for i in range(len(in_metas)))

        def resolve(o):
            return next(in_iter) if isinstance(o, TensorMeta) else o

        resolved_args = map_tree(args, resolve)
        resolved_kwargs = map_tree(kwargs, resolve)

        try:
            if grad_mode:
                # get_batch returns detached views/copies, so resolved tensors are
                # fresh objects that don't alias store contents — mark them directly.
                tensors = collect_leaves(resolved_args, Tensor) + collect_leaves(
                    tuple(resolved_kwargs.values()), Tensor
                )
                for t in tensors:
                    t.requires_grad_(True)
                    t.retain_grad()
                role._grad_inputs[call_id] = tensors

            result = getattr(role, method_name)(*resolved_args, **resolved_kwargs)

            if grad_mode:
                # Save output tensors BEFORE pack so backward can use grad_fn.
                role._grad_outputs[call_id] = collect_leaves(result, Tensor)

            # Pack: collect tensor leaves (tree order), batch-store, substitute metas.
            out_tensors = self._collect(result, Tensor)
            stored = self.transport.put_batch({str(i): t for i, t in enumerate(out_tensors)})
            out_iter = iter(stored[str(i)] for i in range(len(out_tensors)))

            def pack(o):
                return next(out_iter) if isinstance(o, Tensor) else o

            return map_tree(result, pack)
        finally:
            # Release any per-call transport resources (gpu: close IPC views).
            self.transport.end_call()

    # ── Leaf collection ──

    def _collect(self, obj, leaf_type) -> list:
        """Collect leaves of leaf_type in the SAME order ``map_tree`` visits them.

        Using ``map_tree`` for both collect and substitute guarantees the
        get_batch/put_batch result lists align by index with the substitution pass.
        """
        out: list = []

        def visit(o):
            if isinstance(o, leaf_type):
                out.append(o)
            return o

        map_tree(obj, visit)
        return out

    # ── Transport relay (the Worker is the transport's addressable proxy) ──

    def transport_op(self, method: str, *args, **kwargs):
        """Relay a controller-side call into this Worker's transport.

        The transport is a plain in-process object with no Ray address, so
        TensorHandle GC/compute and Handle NCCL routing reach it through the
        Worker actor. Restricted to the transport's REMOTE_OPS allowlist so the
        relay can't be turned into an arbitrary-call gadget; a GLOBAL transport
        (no REMOTE_OPS) cannot be relayed into.
        """
        allowed = getattr(type(self.transport), "REMOTE_OPS", frozenset())
        if method not in allowed:
            raise AttributeError(f"transport_op: {method!r} is not a remote-callable transport op")
        return getattr(self.transport, method)(*args, **kwargs)

    def setup_global_pg(self) -> None:
        """Initialize the cross-worker transfer group (slot0 only).

        Not routed through transport_op: it injects Worker identity (rank /
        world_size) that the controller does not pass.
        """
        self.transport.setup_transfer(self.nccl_rank, self.world_size)

    # ── Info queries ──

    def get_gpu_count(self) -> int:
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def get_cuda_visible_devices(self) -> str:
        return os.environ.get("CUDA_VISIBLE_DEVICES", "")

    def get_node_ip(self) -> str:
        return ray.util.get_node_ip_address()
