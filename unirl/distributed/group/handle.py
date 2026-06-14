"""Handle — controller-side SPMD handle for a group of logical workers.

Handle owns a set of device slots (by device_ids), registers a Remote
on each Worker, and binds @distributed-decorated methods as handle functions.

Cross-GPU TensorHandle transfer is handled automatically: when a shard contains
TensorHandle from a foreign worker, _ensure_local() triggers NCCL send/recv
before execution. Users never call NCCL directly.

Usage:
    pool = DevicePool(num_gpus=8)
    pool.setup()

    # Basic
    handle = pool.create_remote(DiffusionRemote, device_ids=[0,1,2,3])
    handle.initialize(model_path="/models/sd", tp_size=2)

    # With constructor args
    handle = pool.create_remote(ScalerRemote, device_ids=[0,1,2,3], init_kwargs={"scale": 3.0})

    # Separated: tensor transfer is automatic
    actor = pool.create_remote(ActorRemote, device_ids=[0,1,2,3])
    reward = pool.create_remote(RewardRemote, device_ids=[4,5,6,7])
    samples = actor.rollout(prompts=prompts)
    rewards = reward.score(samples)  # auto NCCL from gpu 0-3 to gpu 4-7
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

import ray

from unirl.distributed.group.dispatch import (
    DISPATCH_MODE_REGISTRY,
    DISTRIBUTED_CONFIG_ATTR,
    Dispatch,
    Execute,
    resolve_backward_dispatch_mode,
)
from unirl.distributed.group.remote import RankInfo, Remote
from unirl.distributed.tensor import TensorRef, WorkerLocalTransport, map_tree
from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle
from unirl.distributed.tensor.grad_context import (
    RPCBackwardNode,
    current_grad_context,
)
from unirl.distributed.tensor.pytree import infer_batch_size
from unirl.distributed.utils import collect_leaves

if TYPE_CHECKING:
    from unirl.distributed.group.device_pool import DevicePool


# ── Module-level counter for unique role_name generation ─────────────────────
_role_name_counter: Dict[str, int] = {}


def _owning_class(role_cls) -> Type[Remote]:
    """Return the class for Handle method-binding and role naming.

    ``role_cls`` may be a class (normal case), a bound classmethod
    factory (e.g. ``SD3Bundle.from_config``), or a plain function.
    For classmethods we use the owning class so ``_bind_methods`` finds
    its ``@distributed`` methods and ``_make_role_name`` produces a
    meaningful base name. For everything else, fall back to ``role_cls``
    itself.
    """
    import inspect

    if inspect.ismethod(role_cls) and isinstance(role_cls.__self__, type):
        return role_cls.__self__
    return role_cls


def _make_role_name(role_cls) -> str:
    """Generate a unique role_name from the worker class name.

    Always appends counter suffix for deterministic names.
    """
    base = _owning_class(role_cls).__name__
    count = _role_name_counter.get(base, 0)
    _role_name_counter[base] = count + 1
    return f"{base}_{count}"


def reset_role_name_counter() -> None:
    """Reset the role name counter. For testing only."""
    _role_name_counter.clear()


@dataclass(frozen=True)
class HandleRef:
    """Serializable marker for a Handle.

    When a ``Handle`` is passed as a kwarg to ``remote(...)``, the framework
    substitutes a ``HandleRef`` so the Worker can resolve it to the local
    ``Remote`` instance with this ``role_name`` (looked up in
    ``Worker._roles``) before constructing the new role.

    Only resolves on the same Worker as the referenced role — i.e. when the
    sibling lives on the same device slab and slot.
    """

    role_name: str


class Handle:
    """Controller-side SPMD handle.

    Creates logical workers on Workers and binds @distributed methods.

    Args:
        role_cls:      Remote subclass to register.
        pool:          DevicePool managing Workers.
        device_ids:    Explicit GPU indices. If None, auto-allocate via n_gpus.
        n_gpus:        Number of GPUs to auto-allocate (used when device_ids=None).
        role_name:     Optional role name. If None, auto-generated from class name.
        init_kwargs:   Dict of kwargs forwarded to role_cls.__init__.
    """

    def __init__(
        self,
        role_cls: Type[Remote],
        pool: DevicePool,
        device_ids: Optional[List[int]] = None,
        n_gpus: Optional[int] = None,
        role_name: Optional[str] = None,
        init_kwargs: Optional[Dict[str, Any]] = None,
        slot_id: int = 0,
    ) -> None:  # noqa: D107 (args documented in class docstring)
        self.role_cls = role_cls
        self.pool = pool
        self.role_name = role_name or _make_role_name(role_cls)
        self.slot_id = slot_id

        # GPU allocation
        if device_ids is not None:
            self.device_ids = list(device_ids)  # support range, tuple, etc.
        elif n_gpus is not None:
            self.device_ids = pool.allocate(n_gpus)
        else:
            raise ValueError("Must provide device_ids or n_gpus")

        self.world_size = len(self.device_ids)
        self.workers = pool.get_workers(self.device_ids, slot=slot_id)

        # worker_ids for this group (used in _ensure_local)
        self.worker_ids = [f"dw{d}" if slot_id == 0 else f"dw{d}_s{slot_id}" for d in self.device_ids]

        # Reserve a port on rank 0's node for this group's sub-PG.
        # Held by socket until initialize() releases it.
        self._group_port = ray.get(self.workers[0]._reserve_port.remote())
        self._group_master_addr = ray.get(self.workers[0].get_node_ip.remote())

        self._dist_env_base = {
            "MASTER_ADDR": self._group_master_addr,
            "MASTER_PORT": str(self._group_port),
            "WORLD_SIZE": str(self.world_size),
            "GROUP_NAME": self.role_name,
        }

        # Register role on each Worker with dist_env
        self.rank_infos = [
            RankInfo(rank=i, world_size=self.world_size, dp_rank=i, dp_size=self.world_size)
            for i in range(self.world_size)
        ]
        ray.get(
            [
                w.add_remote.remote(
                    self.role_name,
                    role_cls,
                    self.rank_infos[i],
                    init_kwargs=init_kwargs or {},
                    dist_env={"RANK": str(i), **self._dist_env_base},
                )
                for i, w in enumerate(self.workers)
            ]
        )

        # Bind @distributed methods as handle functions
        self._bind_methods(role_cls)

        # Counter for unique call_id generation within enable_grad contexts.
        # Single-threaded training loop assumption: no concurrent handle calls.
        self._grad_call_counter = count()

    @property
    def dp_size(self) -> int:
        """Number of data-parallel groups."""
        return self.rank_infos[0].dp_size if self.rank_infos else self.world_size

    # ── User-facing initialize ──

    def initialize(self, *args, **kwargs) -> None:
        """Call role.initialize(*args, **kwargs) on all workers.

        Releases the reserved port first so init_process_group can bind it,
        then reads back (possibly modified) rank_infos.
        """
        # Release port so init_process_group can use it
        ray.get(self.workers[0]._release_port.remote(self._group_port))

        # Forward to all workers via generic call
        ray.get([w.call.remote(self.role_name, "initialize", args, kwargs) for w in self.workers])

        # Read back rank_infos (user may have modified them in initialize)
        self.rank_infos = ray.get([w.get_rank_info.remote(self.role_name) for w in self.workers])

    # ── Method binding ──

    def _bind_methods(self, role_cls) -> None:
        """Scan role_cls for @distributed methods and create handle functions.

        For classmethod ``role_cls`` (e.g. ``SD3Bundle.from_config``)
        we scan the owning class instead — the constructed instance is
        of that class, so its ``@distributed`` methods are the ones
        callers will dispatch through this Handle.
        """
        role_cls = _owning_class(role_cls)
        for name in dir(role_cls):
            method = getattr(role_cls, name, None)
            if method is None:
                continue
            config = getattr(method, DISTRIBUTED_CONFIG_ATTR, None)
            if config is None:
                continue

            fns = DISPATCH_MODE_REGISTRY[config["dispatch_mode"]]
            dispatch_fn = fns["dispatch_fn"]
            collect_fn = fns["collect_fn"]

            if config["execute_mode"] == Execute.ALL:
                execute_fn = self._execute_all
            else:
                execute_fn = self._execute_rank_zero

            bound = self._make_handle_fn(name, config["dispatch_mode"], dispatch_fn, collect_fn, execute_fn)
            setattr(self, name, bound)

    def _make_handle_fn(
        self,
        method_name: str,
        dispatch_mode: Dispatch,
        dispatch_fn: Callable,
        collect_fn: Callable,
        execute_fn: Callable,
    ) -> Callable:
        """Create handle method: dispatch → localize → execute → collect → rebind.

        When a GradContext is active, wraps the call to record input/output
        TensorMetas and append an RPCBackwardNode for later auto-backward.
        grad_mode and call_id are passed as dedicated parameters to Worker.call
        (not via kwargs) so dispatch internals remain unaware of grad state.
        """

        def handle_fn(*args, **kwargs):
            ctx = current_grad_context()

            # ── enable_grad: validate backward support, record input TensorMetas ──
            call_id = None
            input_metas = []
            bwd_dispatch_mode = None
            if ctx is not None:
                bwd_dispatch_mode = resolve_backward_dispatch_mode(method_name, dispatch_mode, self.rank_infos)
                call_id = f"{method_name}_{next(self._grad_call_counter)}"
                input_metas = collect_leaves(args, TensorRef) + collect_leaves(tuple(kwargs.values()), TensorRef)

            batch_size = infer_batch_size(args, kwargs)
            # Only DP_SCATTER/DP_SCATTER_HEAD split the per-sample batch by dp_size, so only
            # they require divisibility; BROADCAST/SCATTER must not be rejected (main #202).
            if (
                dispatch_mode in (Dispatch.DP_SCATTER, Dispatch.DP_SCATTER_HEAD)
                and batch_size is not None
                and batch_size % self.dp_size != 0
            ):
                raise ValueError(f"batch_size={batch_size} not divisible by dp_size={self.dp_size}")

            shards = dispatch_fn(self, args, kwargs, batch_size)
            # Locality + cross-worker transfer is the transport's policy: its
            # localize makes every ref resolvable on its dst worker (GLOBAL =
            # identity; worker-local = NCCL/IPC routing). It needs controller
            # topology + per-shard dst identity, passed directly.
            transport_cls = self.pool.transport_cls
            worker_local = issubclass(transport_cls, WorkerLocalTransport)
            shards = transport_cls.localize(shards, self.pool, self.device_ids, self.worker_ids)
            # grad_mode/call_id passed as dedicated args, not mixed into kwargs
            refs = execute_fn(method_name, shards, grad_mode=ctx is not None, call_id=call_id)
            results = ray.get(refs)

            # Rebind before collect: results[i] comes from workers[i],
            # so worker attribution is unambiguous at this point. For worker-local
            # this registers the decref GC finalizer; GLOBAL lifecycle is
            # queue-managed, so skip rebind/GC there.
            results = [self._rebind_tree(r, self.workers[i], worker_local=worker_local) for i, r in enumerate(results)]

            # Collect: merge primary rank results
            collected = collect_fn(self, results)

            if ctx is not None:
                output_metas = collect_leaves(collected, TensorRef)
                ctx.nodes.append(
                    RPCBackwardNode(
                        role_proxy=self,
                        call_id=call_id,
                        dispatch_mode=bwd_dispatch_mode,
                        input_metas=input_metas,
                        output_metas=output_metas,
                    )
                )

            return collected

        handle_fn.__name__ = method_name
        handle_fn.__doc__ = f"SPMD handle: {method_name} (dispatch={dispatch_fn.__name__})"
        return handle_fn

    # ── Execute strategies ──

    def _execute_all(self, method_name: str, shards: List, grad_mode: bool = False, call_id=None) -> List:
        """Send RPC to all Workers."""
        return [
            w.call.remote(self.role_name, method_name, s_args, s_kwargs, grad_mode, call_id)
            for w, (s_args, s_kwargs) in zip(self.workers, shards)
        ]

    def _execute_rank_zero(self, method_name: str, shards: List, grad_mode: bool = False, call_id=None) -> List:
        """Send RPC to rank 0 only."""
        return [
            self.workers[0].call.remote(self.role_name, method_name, shards[0][0], shards[0][1], grad_mode, call_id)
        ]

    # ── TensorHandle rebinding ──

    def _rebind_tree(self, obj, worker_handle, *, worker_local: bool = True):
        """Rebind every ref leaf onto ``worker_handle`` and wrap bare handles in TensorRef.

        For worker-local backends, ``rebind`` attaches the worker actor handle and
        registers the decref GC finalizer. For GLOBAL backends the refs resolve
        anywhere and lifecycle is queue-managed, so no rebind/GC is done (and the
        refs need not be TensorHandle). Only the per-leaf rebind policy lives here;
        the tree recursion (Batch/tuple/list/dict, cu_seqlens preserved) is delegated
        to the shared :func:`map_tree`.
        """

        def rebind_leaf(o):
            if isinstance(o, GPUTensorHandle):
                if worker_local:
                    o.rebind(worker_handle)
                return TensorRef.from_handles([o])
            if isinstance(o, TensorRef) and worker_local:
                for s in o.spans:
                    s.handle.rebind(worker_handle)
            return o

        return map_tree(obj, rebind_leaf)
