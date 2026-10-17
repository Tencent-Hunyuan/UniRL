"""Handle — controller-side SPMD handle for a group of logical workers."""

from __future__ import annotations

import inspect
import logging
import threading
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import ray

from unirl.distributed.group.dispatch import (
    DISPATCH_MODE_REGISTRY,
    DISTRIBUTED_CONFIG_ATTR,
    Dispatch,
    Execute,
    resolve_backward_dispatch_mode,
)
from unirl.distributed.group.ray_utils import get_actor_results, inspect_ready_actor_results
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


logger = logging.getLogger(__name__)


_role_name_counter: Dict[str, int] = {}


def _read_positive_size(cfg: Dict[str, Any], key: str) -> int:
    val = cfg.get(key)
    if val is None:
        return 1
    size = int(val)
    if size < 1:
        raise ValueError(f"{key} must be >= 1, got {val!r}")
    return size


def _build_rank_infos(
    init_kwargs: Optional[Dict[str, Any]],
    world_size: int,
    *,
    is_tp_engine: bool,
) -> List[RankInfo]:
    """Resolve ``(sp, tp, pp, ep)`` layout hints and build contiguous rank layout."""
    sp = tp = pp = ep = 1
    if init_kwargs:
        fsdp_cfg = init_kwargs.get("fsdp_cfg")
        sp = _read_positive_size(fsdp_cfg if fsdp_cfg is not None else init_kwargs, "sp_size")
        tp = _read_positive_size(init_kwargs, "tp_size")
        pp = _read_positive_size(init_kwargs, "pp_size")
        ep = _read_positive_size(init_kwargs, "ep_size")

        if is_tp_engine and init_kwargs.get("config") is not None:
            cfg = init_kwargs["config"]
            tp = max(tp, _read_positive_size(cfg, "tp_size"))
            pp = max(pp, _read_positive_size(cfg, "pp_size"))
            ep = max(ep, _read_positive_size(cfg, "ep_size"))

        for value in init_kwargs.values():
            if isinstance(value, HandleRef):
                sp = max(sp, value.sp_size)
                tp = max(tp, value.tp_size)
                pp = max(pp, value.pp_size)
                ep = max(ep, value.ep_size)

        if world_size % sp != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sp})")
        if sp > 1 and (tp > 1 or pp > 1):
            raise ValueError(f"sp_size ({sp}) cannot be combined with rollout tp/pp layout ({tp=}, {pp=})")
        inner = tp * pp
        if world_size % inner != 0:
            raise ValueError(f"world_size ({world_size}) must be divisible by tp_size*pp_size ({inner})")

    if sp > 1:
        dp_size = world_size // sp
        return [
            RankInfo(
                rank=i,
                world_size=world_size,
                dp_rank=i // sp,
                dp_size=dp_size,
                sp_rank=i % sp,
                sp_size=sp,
            )
            for i in range(world_size)
        ]

    inner = tp * pp
    dp_size = world_size // inner
    return [
        RankInfo(
            rank=i,
            world_size=world_size,
            dp_rank=i // inner,
            dp_size=dp_size,
            tp_rank=i % tp,
            tp_size=tp,
            pp_rank=(i // tp) % pp,
            pp_size=pp,
            ep_rank=0,
            ep_size=ep,
        )
        for i in range(world_size)
    ]


def _build_tp_visible_device_map(
    rank_infos: Sequence[RankInfo],
    *,
    node_ips: Sequence[str],
    cuda_visible_devices: Sequence[str],
) -> Dict[int, List[str]]:
    """Map each TP worker index to its node-local Ray CUDA token list."""
    size = len(rank_infos)
    if len(node_ips) != size or len(cuda_visible_devices) != size:
        raise ValueError(
            "TP worker metadata length mismatch: "
            f"rank_infos={size}, node_ips={len(node_ips)}, "
            f"cuda_visible_devices={len(cuda_visible_devices)}"
        )

    groups: Dict[Tuple[int, int], List[int]] = {}
    for index, rank_info in enumerate(rank_infos):
        if rank_info.tp_size > 1:
            groups.setdefault((rank_info.dp_rank, rank_info.pp_rank), []).append(index)

    result: Dict[int, List[str]] = {}
    for group_key, indices in groups.items():
        ordered = sorted(indices, key=lambda index: rank_infos[index].tp_rank)
        tp_size = rank_infos[ordered[0]].tp_size
        tp_ranks = [rank_infos[index].tp_rank for index in ordered]
        if len(ordered) != tp_size or tp_ranks != list(range(tp_size)):
            raise ValueError(f"incomplete TP group {group_key}: expected ranks 0..{tp_size - 1}, got {tp_ranks}")

        group_nodes = [node_ips[index].strip() for index in ordered]
        if any(not node for node in group_nodes) or len(set(group_nodes)) != 1:
            raise ValueError(
                f"each rollout TP group must be placed on a single node; group={group_key}, nodes={group_nodes}"
            )

        tokens: List[str] = []
        for index in ordered:
            raw = cuda_visible_devices[index].strip()
            if not raw:
                raise ValueError(f"SGLang TP worker {index} has an empty CUDA_VISIBLE_DEVICES token")
            split = [token.strip() for token in raw.split(",") if token.strip()]
            if len(split) != 1:
                raise ValueError(
                    "each rollout TP worker must expose exactly one CUDA_VISIBLE_DEVICES "
                    f"token; worker={index}, value={raw!r}"
                )
            tokens.append(split[0])
        if len(set(tokens)) != len(tokens):
            raise ValueError(
                "CUDA_VISIBLE_DEVICES tokens within an SGLang TP group must be unique; "
                f"group={group_key}, tokens={tokens}"
            )

        for index in ordered:
            result[index] = list(tokens)
    return result


@dataclass(frozen=True)
class HandleRef:
    """Serializable marker for a Handle."""

    role_name: str
    sp_size: int = 1
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1


class PendingHandleCall:
    """Future-like result of :meth:`Slot.launch`: launched, not yet collected."""

    def __init__(
        self,
        handle: "Handle",
        method_name: str,
        refs: List[Any],
        worker_local: bool,
        *,
        targets: Optional[List[Any]] = None,
        collect_fn: Callable,
        leases: Optional[List[Any]] = None,
    ) -> None:
        self._handle = handle
        self._method_name = method_name
        self._refs = refs
        self._worker_local = worker_local
        self._targets = targets
        self._collect_fn = collect_fn
        self._leases = leases
        self._consumed = False
        self._value: Any = None
        self._discard_futures: Optional[List[Any]] = None

    def ready(self) -> bool:
        """True when every rank succeeded; raise immediately on a ready rank-local error."""
        return inspect_ready_actor_results(
            self._refs,
            pool=self._handle.pool,
            role_name=self._handle.role_name,
            method_name=self._method_name,
        )

    def wait(self) -> None:
        """Block until completion and safely discard the collected return value."""
        if self._consumed:
            return
        self.result()
        self._value = None

    def discard_on_completion(self) -> None:
        """Retain leases and discard outputs when all worker refs finish."""
        if self._consumed:
            self._value = None
            return
        if self._discard_futures is not None:
            return
        try:
            futures = [ref.future() for ref in self._refs]
        except Exception:
            threading.Thread(
                target=self._discard_result,
                name="pending-handle-discard",
                daemon=True,
            ).start()
            return
        if not futures:
            self._discard_result()
            return

        remaining = len(futures)
        lock = threading.Lock()

        def on_done(_) -> None:
            nonlocal remaining
            with lock:
                remaining -= 1
                complete = remaining == 0
            if complete:
                self._discard_result()

        self._discard_futures = futures
        for future in futures:
            future.add_done_callback(on_done)

    def result(self) -> Any:
        """Block if needed, then rebind + collect: the method's collected return value."""
        if self._consumed:
            return self._value
        try:
            self._value = self._handle._resolve_call(
                self._collect_fn,
                self._refs,
                worker_local=self._worker_local,
                targets=self._targets,
                method_name=self._method_name,
            )
        finally:
            self._leases = None
        self._consumed = True
        return self._value

    def _discard_result(self) -> None:
        try:
            self.wait()
        except Exception:
            logger.debug("PendingHandleCall: discarded call failed during completion", exc_info=True)
        finally:
            self._discard_futures = None


class Slot:
    """Driver-side handle to one worker in a :class:`Handle`."""

    def __init__(self, handle: "Handle", index: int) -> None:
        self._handle = handle
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    def launch(self, method_name: str, *args, **kwargs) -> PendingHandleCall:
        """Launch an undecorated role method on this worker."""
        handle = self._handle
        handle.pool.assert_usable()
        if method_name in handle._distributed_methods:
            raise AttributeError(f"{method_name!r} is distributed; call it on the Handle")
        if not hasattr(handle._owning_cls, method_name):
            raise AttributeError(f"{method_name!r} is not a method of {handle._owning_cls.__name__}")
        if current_grad_context() is not None:
            raise RuntimeError(f"Slot call {method_name!r} is not valid inside a GradContext")

        transport_cls = handle.pool.transport_cls
        worker_local = issubclass(transport_cls, WorkerLocalTransport)
        shards = transport_cls.localize(
            [(args, kwargs)],
            handle.pool,
            [handle.device_ids[self._index]],
            [handle.worker_ids[self._index]],
        )
        worker = handle.workers[self._index]
        s_args, s_kwargs = shards[0]
        ref = worker.call.remote(handle.role_name, method_name, s_args, s_kwargs, False, None)
        return PendingHandleCall(
            handle,
            method_name,
            [ref],
            worker_local,
            targets=[worker],
            collect_fn=lambda _, results: results[0],
            leases=shards,
        )


class Handle:
    """Controller-side SPMD handle."""

    def __init__(
        self,
        role_cls: Type[Remote],
        pool: DevicePool,
        device_ids: List[int],
        init_kwargs: Optional[Dict[str, Any]] = None,
        slot_id: int = 0,
    ) -> None:
        pool.assert_usable()
        self._owning_cls: Type[Remote] = (
            role_cls.__self__ if inspect.ismethod(role_cls) and isinstance(role_cls.__self__, type) else role_cls
        )
        self.pool = pool
        base = self._owning_cls.__name__
        idx = _role_name_counter.get(base, 0)
        _role_name_counter[base] = idx + 1
        self.role_name = f"{base}_{idx}"

        self.device_ids = list(device_ids)
        self.world_size = len(self.device_ids)
        self.workers = pool.get_workers(self.device_ids, slot=slot_id)
        self.worker_ids = [f"dw{d}" if slot_id == 0 else f"dw{d}_s{slot_id}" for d in self.device_ids]

        self._group_port = ray.get(self.workers[0]._reserve_port.remote())
        group_master_addr = ray.get(self.workers[0].get_node_ip.remote())
        dist_env_base = {
            "MASTER_ADDR": group_master_addr,
            "MASTER_PORT": str(self._group_port),
            "WORLD_SIZE": str(self.world_size),
            "GROUP_NAME": self.role_name,
        }

        is_tp_engine = getattr(self._owning_cls, "_accepts_rollout_tp_kwargs", False) is True
        self.rank_infos = _build_rank_infos(init_kwargs, self.world_size, is_tp_engine=is_tp_engine)
        logger.info(
            "Handle layout: role=%s world=%d dp=%d sp=%d tp=%d pp=%d ep=%d",
            self.role_name,
            self.world_size,
            self.rank_infos[0].dp_size,
            self.rank_infos[0].sp_size,
            self.rank_infos[0].tp_size,
            self.rank_infos[0].pp_size,
            self.rank_infos[0].ep_size,
        )
        tp_visible_device_map: Dict[int, List[str]] = {}
        if is_tp_engine and any(rank_info.tp_size > 1 for rank_info in self.rank_infos):
            node_ips = get_actor_results(
                [worker.get_node_ip.remote() for worker in self.workers],
                pool=self.pool,
                role_name=self.role_name,
                method_name="get_node_ip",
            )
            cuda_visible_devices = get_actor_results(
                [worker.get_cuda_visible_devices.remote() for worker in self.workers],
                pool=self.pool,
                role_name=self.role_name,
                method_name="get_cuda_visible_devices",
            )
            tp_visible_device_map = _build_tp_visible_device_map(
                self.rank_infos,
                node_ips=node_ips,
                cuda_visible_devices=cuda_visible_devices,
            )
        base_init_kwargs = {} if init_kwargs is None else dict(init_kwargs)
        for key in ("sp_size", "tp_size", "pp_size", "ep_size"):
            base_init_kwargs.pop(key, None)

        def _rank_init_kwargs(i: int) -> Dict[str, Any]:
            kwargs = dict(base_init_kwargs)
            ri = self.rank_infos[i]
            if not is_tp_engine or (ri.tp_size <= 1 and ri.pp_size <= 1 and ri.ep_size <= 1):
                return kwargs
            kwargs.update(
                {
                    "tp_rank": ri.tp_rank,
                    "tp_size": ri.tp_size,
                    "pp_rank": ri.pp_rank,
                    "pp_size": ri.pp_size,
                    "ep_rank": ri.ep_rank,
                    "ep_size": ri.ep_size,
                }
            )
            if ri.tp_size > 1:
                kwargs["tp_visible_devices"] = tp_visible_device_map[i]
            return kwargs

        get_actor_results(
            [
                w.add_remote.remote(
                    self.role_name,
                    role_cls,
                    self.rank_infos[i],
                    init_kwargs=_rank_init_kwargs(i),
                    dist_env={"RANK": str(i), **dist_env_base},
                )
                for i, w in enumerate(self.workers)
            ],
            pool=self.pool,
            role_name=self.role_name,
            method_name="add_remote",
        )

        self._distributed_methods: set[str] = set()
        self._bind_methods()

        self._grad_call_counter = count()

    @property
    def dp_size(self) -> int:
        """Number of data-parallel groups."""
        return self.rank_infos[0].dp_size

    @property
    def sp_size(self) -> int:
        """Ulysses sequence-parallel degree of this handle's rank layout (1 = flat)."""
        return self.rank_infos[0].sp_size

    @property
    def tp_size(self) -> int:
        """Tensor-parallel degree of this handle's rank layout."""
        return self.rank_infos[0].tp_size

    @property
    def pp_size(self) -> int:
        """Pipeline-parallel degree of this handle's rank layout."""
        return self.rank_infos[0].pp_size

    def slot(self, index: int) -> Slot:
        if index < 0 or index >= self.world_size:
            raise IndexError(f"slot index {index} outside [0, {self.world_size})")
        return Slot(self, index)

    @property
    def engine_slots(self) -> List[Slot]:
        """Slots that host addressable rollout-engine heads, one per DP replica."""
        if self.sp_size > 1:
            raise RuntimeError(
                f"slot-local rollout dispatch does not support sequence-parallel engines; got sp_size={self.sp_size}"
            )
        slots = [
            self.slot(index)
            for index, info in enumerate(self.rank_infos)
            if info.tp_rank == 0 and info.pp_rank == 0 and info.sp_rank == 0
        ]
        if len(slots) != self.dp_size:
            raise RuntimeError(f"expected {self.dp_size} rollout engine slots, found {len(slots)}")
        return slots

    @property
    def ep_size(self) -> int:
        """Expert-parallel degree requested for rollout-side engines."""
        return self.rank_infos[0].ep_size

    @property
    def tp_zero_workers(self) -> List[Any]:
        """Worker actor handles that host a SGLang engine (tp_rank==0)."""
        return [w for w, ri in zip(self.workers, self.rank_infos) if ri.tp_rank == 0]

    def initialize(self, *args, **kwargs) -> None:
        """Call role.initialize(*args, **kwargs) on all workers."""
        self.pool.assert_usable()
        ray.get(self.workers[0]._release_port.remote(self._group_port))

        get_actor_results(
            [w.call.remote(self.role_name, "initialize", args, kwargs) for w in self.workers],
            pool=self.pool,
            role_name=self.role_name,
            method_name="initialize",
        )

        self.rank_infos = get_actor_results(
            [w.get_rank_info.remote(self.role_name) for w in self.workers],
            pool=self.pool,
            role_name=self.role_name,
            method_name="get_rank_info",
        )

    def _bind_methods(self) -> None:
        """Scan role_cls for @distributed methods and create handle functions."""
        role_cls = self._owning_cls
        for name in dir(role_cls):
            method = getattr(role_cls, name, None)
            if method is None:
                continue
            config = getattr(method, DISTRIBUTED_CONFIG_ATTR, None)
            if config is None:
                continue
            if name in {"slot", "engine_slots"}:
                raise TypeError(f"distributed method {name!r} collides with the Handle API")

            dispatch_mode = config["dispatch_mode"]
            fns = DISPATCH_MODE_REGISTRY[dispatch_mode]
            dispatch_fn = fns["dispatch_fn"]
            collect_fn = fns["collect_fn"]
            execute_All = config["execute_mode"] == Execute.ALL

            self._distributed_methods.add(name)
            setattr(self, name, self._make_handle_fn(name, dispatch_mode, dispatch_fn, collect_fn, execute_All))

    def _make_handle_fn(
        self,
        method_name: str,
        dispatch_mode: Dispatch,
        dispatch_fn: Callable,
        collect_fn: Callable,
        execute_all: bool,
    ) -> Callable:
        """Create handle method: dispatch → localize → execute → collect → rebind."""

        def handle_fn(*args, **kwargs):
            ray_get_timeout = kwargs.pop("_ray_get_timeout", None)
            ctx = current_grad_context()

            call_id = None
            input_metas = []
            bwd_dispatch_mode = None
            if ctx is not None:
                bwd_dispatch_mode = resolve_backward_dispatch_mode(method_name, dispatch_mode, self.rank_infos)
                call_id = f"{method_name}_{next(self._grad_call_counter)}"
                input_metas = collect_leaves(args, TensorRef) + collect_leaves(tuple(kwargs.values()), TensorRef)

            self.pool.assert_usable()
            batch_size = infer_batch_size(args, kwargs)
            if (
                dispatch_mode in (Dispatch.DP_SCATTER, Dispatch.DP_SCATTER_HEAD)
                and batch_size is not None
                and batch_size % self.dp_size != 0
            ):
                raise ValueError(f"batch_size={batch_size} not divisible by dp_size={self.dp_size}")

            shards = dispatch_fn(self, args, kwargs, batch_size)
            transport_cls = self.pool.transport_cls
            worker_local = issubclass(transport_cls, WorkerLocalTransport)
            shards = transport_cls.localize(shards, self.pool, self.device_ids, self.worker_ids)
            grad_mode = ctx is not None
            if execute_all:
                refs = [
                    w.call.remote(self.role_name, method_name, s_args, s_kwargs, grad_mode, call_id)
                    for w, (s_args, s_kwargs) in zip(self.workers, shards)
                ]
            else:
                refs = [
                    self.workers[0].call.remote(
                        self.role_name, method_name, shards[0][0], shards[0][1], grad_mode, call_id
                    )
                ]

            try:
                collected = self._resolve_call(
                    collect_fn,
                    refs,
                    worker_local=worker_local,
                    ray_get_timeout=ray_get_timeout,
                    method_name=method_name,
                )
            finally:
                shards.clear()

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

    def _resolve_call(
        self,
        collect_fn: Callable,
        refs: List,
        *,
        worker_local: bool,
        ray_get_timeout: Optional[float] = None,
        targets: Optional[List[Any]] = None,
        method_name: str,
    ):
        """Resolve a launched call into its collected method return value."""
        results = get_actor_results(
            refs,
            pool=self.pool,
            role_name=self.role_name,
            method_name=method_name,
            timeout=ray_get_timeout,
        )
        workers = self.workers if targets is None else targets

        def _rebind(obj, worker_handle):
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

        results = [_rebind(r, workers[i]) for i, r in enumerate(results)]
        return collect_fn(self, results)
