"""Handle — controller-side SPMD handle for a group of logical workers."""

from __future__ import annotations

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


def _owning_class(role_cls) -> Type[Remote]:
    """Return the class for Handle method-binding and role naming."""
    import inspect

    if inspect.ismethod(role_cls) and isinstance(role_cls.__self__, type):
        return role_cls.__self__
    return role_cls


def _make_role_name(role_cls) -> str:
    """Generate a unique role_name from the worker class name."""
    base = _owning_class(role_cls).__name__
    count = _role_name_counter.get(base, 0)
    _role_name_counter[base] = count + 1
    return f"{base}_{count}"


def reset_role_name_counter() -> None:
    """Reset the role name counter. For testing only."""
    _role_name_counter.clear()


def _cfg_get(cfg: Any, key: str, default: int) -> int:
    if isinstance(cfg, dict):
        val = cfg.get(key, default)
    else:
        val = getattr(cfg, key, default)
    return int(val or default)


def _is_sglang_rollout_role(role_cls: Type[Remote]) -> bool:
    """True if ``role_cls`` opts into per-rank rollout-TP kwargs."""
    cls = _owning_class(role_cls)
    return getattr(cls, "_accepts_rollout_tp_kwargs", False) is True


def _parallel_shape_from_init_kwargs(
    init_kwargs: Optional[Dict[str, Any]],
    world_size: int,
    role_cls: Type[Remote],
) -> Tuple[int, int, int, int]:
    """Resolve ``(sp, tp, pp, ep)`` layout hints for this handle."""
    if not init_kwargs:
        return 1, 1, 1, 1

    sp = 1
    fsdp_cfg = init_kwargs.get("fsdp_cfg")
    if fsdp_cfg is not None:
        sp = _cfg_get(fsdp_cfg, "sp_size", 1)
    elif "sp_size" in init_kwargs:
        sp = int(init_kwargs.get("sp_size") or 1)

    tp = int(init_kwargs.get("tp_size") or 1)
    pp = int(init_kwargs.get("pp_size") or 1)
    ep = int(init_kwargs.get("ep_size") or 1)

    if _is_sglang_rollout_role(role_cls):
        cfg = init_kwargs.get("config")
        if cfg is not None:
            tp = max(tp, _cfg_get(cfg, "tp_size", 1))
            pp = max(pp, _cfg_get(cfg, "pp_size", 1))
            ep = max(ep, _cfg_get(cfg, "ep_size", 1))

    for value in init_kwargs.values():
        if isinstance(value, HandleRef):
            sp = max(sp, int(getattr(value, "sp_size", 1) or 1))
            tp = max(tp, int(getattr(value, "tp_size", 1) or 1))
            pp = max(pp, int(getattr(value, "pp_size", 1) or 1))
            ep = max(ep, int(getattr(value, "ep_size", 1) or 1))

    sp = sp if (sp > 1 and world_size % sp == 0) else 1
    tp = max(1, tp)
    pp = max(1, pp)
    ep = max(1, ep)
    if sp > 1 and (tp > 1 or pp > 1):
        raise ValueError(f"sp_size ({sp}) cannot be combined with rollout tp/pp layout ({tp=}, {pp=})")
    inner = tp * pp
    if world_size % inner != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by tp_size*pp_size ({inner})")
    return sp, tp, pp, ep


def _build_rank_infos(
    world_size: int,
    sp_size: int = 1,
    tp_size: int = 1,
    pp_size: int = 1,
    ep_size: int = 1,
) -> List[RankInfo]:
    """Build contiguous rank layout."""
    if sp_size > 1:
        dp_size = world_size // sp_size
        return [
            RankInfo(
                rank=i,
                world_size=world_size,
                dp_rank=i // sp_size,
                dp_size=dp_size,
                sp_rank=i % sp_size,
                sp_size=sp_size,
            )
            for i in range(world_size)
        ]

    inner = tp_size * pp_size
    dp_size = world_size // inner
    return [
        RankInfo(
            rank=i,
            world_size=world_size,
            dp_rank=i // inner,
            dp_size=dp_size,
            tp_rank=i % tp_size,
            tp_size=tp_size,
            pp_rank=(i // tp_size) % pp_size,
            pp_size=pp_size,
            ep_rank=0,
            ep_size=ep_size,
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
        if int(rank_info.tp_size) > 1:
            groups.setdefault((int(rank_info.dp_rank), int(rank_info.pp_rank)), []).append(index)

    result: Dict[int, List[str]] = {}
    for group_key, indices in groups.items():
        ordered = sorted(indices, key=lambda index: int(rank_infos[index].tp_rank))
        tp_size = int(rank_infos[ordered[0]].tp_size)
        tp_ranks = [int(rank_infos[index].tp_rank) for index in ordered]
        if len(ordered) != tp_size or tp_ranks != list(range(tp_size)):
            raise ValueError(f"incomplete TP group {group_key}: expected ranks 0..{tp_size - 1}, got {tp_ranks}")

        group_nodes = [str(node_ips[index]).strip() for index in ordered]
        if any(not node for node in group_nodes) or len(set(group_nodes)) != 1:
            raise ValueError(
                f"each SGLang TP group must be placed on a single node; group={group_key}, nodes={group_nodes}"
            )

        tokens: List[str] = []
        for index in ordered:
            raw = str(cuda_visible_devices[index]).strip()
            if not raw:
                raise ValueError(f"SGLang TP worker {index} has an empty CUDA_VISIBLE_DEVICES token")
            split = [token.strip() for token in raw.split(",") if token.strip()]
            if len(split) != 1:
                raise ValueError(
                    "each SGLang TP Worker must expose exactly one CUDA_VISIBLE_DEVICES "
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
    """Future-like result of :meth:`Handle.launch_nowait`: launched, not yet collected."""

    def __init__(
        self,
        handle: "Handle",
        method_name: str,
        refs: List[Any],
        worker_local: bool,
        *,
        targets: Optional[List[Any]] = None,
        collect_fn: Optional[Callable] = None,
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
        """True once every worker's ref is resolved (non-blocking probe)."""
        done, _ = ray.wait(self._refs, num_returns=len(self._refs), timeout=0)
        return len(done) == len(self._refs)

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
        handle = self._handle
        collect_fn = self._collect_fn
        if collect_fn is None:
            _, _, collect_fn, _ = handle._method_configs[self._method_name]
        try:
            self._value = handle._resolve_call(
                collect_fn,
                self._refs,
                worker_local=self._worker_local,
                targets=self._targets,
            )
        finally:
            self._release_leases()
        self._consumed = True
        return self._value

    def _release_leases(self) -> None:
        """Release handles owning destination TensorStore entries after RPC consumption."""
        self._leases = None

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
        self._index = int(index)

    @property
    def index(self) -> int:
        return self._index

    def launch(self, method_name: str, *args, **kwargs) -> PendingHandleCall:
        """Launch an undecorated role method on this worker."""
        handle = self._handle
        if method_name in handle._method_configs:
            raise AttributeError(f"{method_name!r} is distributed; call it on the Handle")
        if not hasattr(_owning_class(handle.role_cls), method_name):
            raise AttributeError(f"{method_name!r} is not a method of {_owning_class(handle.role_cls).__name__}")
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

    def call(self, method_name: str, *args, **kwargs) -> Any:
        return self.launch(method_name, *args, **kwargs).result()


class Handle:
    """Controller-side SPMD handle."""

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

        if device_ids is not None:
            self.device_ids = list(device_ids)
        elif n_gpus is not None:
            self.device_ids = pool.allocate(n_gpus)
        else:
            raise ValueError("Must provide device_ids or n_gpus")

        self.world_size = len(self.device_ids)
        self.workers = pool.get_workers(self.device_ids, slot=slot_id)

        self.worker_ids = [f"dw{d}" if slot_id == 0 else f"dw{d}_s{slot_id}" for d in self.device_ids]

        self._group_port = ray.get(self.workers[0]._reserve_port.remote())
        self._group_master_addr = ray.get(self.workers[0].get_node_ip.remote())

        self._dist_env_base = {
            "MASTER_ADDR": self._group_master_addr,
            "MASTER_PORT": str(self._group_port),
            "WORLD_SIZE": str(self.world_size),
            "GROUP_NAME": self.role_name,
        }

        sp_size, tp_size, pp_size, ep_size = _parallel_shape_from_init_kwargs(init_kwargs, self.world_size, role_cls)
        self.rank_infos = _build_rank_infos(
            self.world_size,
            sp_size=sp_size,
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
        )
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
        is_tp_engine = _is_sglang_rollout_role(role_cls)
        tp_visible_device_map: Dict[int, List[str]] = {}
        if is_tp_engine and any(rank_info.tp_size > 1 for rank_info in self.rank_infos):
            node_ips = ray.get([worker.get_node_ip.remote() for worker in self.workers])
            cuda_visible_devices = ray.get([worker.get_cuda_visible_devices.remote() for worker in self.workers])
            tp_visible_device_map = _build_tp_visible_device_map(
                self.rank_infos,
                node_ips=node_ips,
                cuda_visible_devices=cuda_visible_devices,
            )
        base_init_kwargs = dict(init_kwargs or {})
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

        ray.get(
            [
                w.add_remote.remote(
                    self.role_name,
                    role_cls,
                    self.rank_infos[i],
                    init_kwargs=_rank_init_kwargs(i),
                    dist_env={"RANK": str(i), **self._dist_env_base},
                )
                for i, w in enumerate(self.workers)
            ]
        )

        self._method_configs: Dict[str, tuple] = {}
        self._bind_methods(role_cls)

        self._grad_call_counter = count()

    @property
    def dp_size(self) -> int:
        """Number of data-parallel groups."""
        return self.rank_infos[0].dp_size if self.rank_infos else self.world_size

    @property
    def sp_size(self) -> int:
        """Ulysses sequence-parallel degree of this handle's rank layout (1 = flat)."""
        return self.rank_infos[0].sp_size if self.rank_infos else 1

    @property
    def tp_size(self) -> int:
        """Tensor-parallel degree of this handle's rank layout."""
        return self.rank_infos[0].tp_size if self.rank_infos else 1

    @property
    def pp_size(self) -> int:
        """Pipeline-parallel degree of this handle's rank layout."""
        return self.rank_infos[0].pp_size if self.rank_infos else 1

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
        return self.rank_infos[0].ep_size if self.rank_infos else 1

    @property
    def tp_zero_workers(self) -> List[Any]:
        """Worker actor handles that host a SGLang engine (tp_rank==0)."""
        return [w for w, ri in zip(self.workers, self.rank_infos) if ri.tp_rank == 0]

    def initialize(self, *args, **kwargs) -> None:
        """Call role.initialize(*args, **kwargs) on all workers."""
        ray.get(self.workers[0]._release_port.remote(self._group_port))

        ray.get([w.call.remote(self.role_name, "initialize", args, kwargs) for w in self.workers])

        self.rank_infos = ray.get([w.get_rank_info.remote(self.role_name) for w in self.workers])

    def _bind_methods(self, role_cls) -> None:
        """Scan role_cls for @distributed methods and create handle functions."""
        role_cls = _owning_class(role_cls)
        for name in dir(role_cls):
            method = getattr(role_cls, name, None)
            if method is None:
                continue
            config = getattr(method, DISTRIBUTED_CONFIG_ATTR, None)
            if config is None:
                continue
            if name in {"slot", "engine_slots"}:
                raise TypeError(f"distributed method {name!r} collides with the Handle API")

            fns = DISPATCH_MODE_REGISTRY[config["dispatch_mode"]]
            dispatch_fn = fns["dispatch_fn"]
            collect_fn = fns["collect_fn"]

            if config["execute_mode"] == Execute.ALL:
                execute_fn = self._execute_all
            else:
                execute_fn = self._execute_rank_zero

            self._method_configs[name] = (config["dispatch_mode"], dispatch_fn, collect_fn, execute_fn)
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

            refs, worker_local, leases = self._launch_call(
                method_name,
                dispatch_mode,
                dispatch_fn,
                execute_fn,
                args,
                kwargs,
                grad_mode=ctx is not None,
                call_id=call_id,
            )
            try:
                collected = self._resolve_call(
                    collect_fn,
                    refs,
                    worker_local=worker_local,
                    ray_get_timeout=ray_get_timeout,
                )
            finally:
                leases.clear()

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

    def _launch_call(
        self,
        method_name: str,
        dispatch_mode: Dispatch,
        dispatch_fn: Callable,
        execute_fn: Callable,
        args: tuple,
        kwargs: dict,
        *,
        grad_mode: bool,
        call_id: Optional[str],
    ) -> Tuple[List, bool, List]:
        """Launch a distributed call and retain localized argument leases."""
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
        refs = execute_fn(method_name, shards, grad_mode=grad_mode, call_id=call_id)
        return refs, worker_local, shards

    def _resolve_call(
        self,
        collect_fn: Callable,
        refs: List,
        *,
        worker_local: bool,
        ray_get_timeout: Optional[float] = None,
        targets: Optional[List[Any]] = None,
    ):
        """Resolve a launched call into its collected method return value."""
        results = ray.get(refs, timeout=ray_get_timeout)
        workers = self.workers if targets is None else targets
        results = [self._rebind_tree(r, workers[i], worker_local=worker_local) for i, r in enumerate(results)]
        return collect_fn(self, results)

    def launch_nowait(self, method_name: str, *args, **kwargs) -> PendingHandleCall:
        """Launch a @distributed method without blocking: the launch phase of"""
        try:
            dispatch_mode, dispatch_fn, _, execute_fn = self._method_configs[method_name]
        except KeyError:
            raise AttributeError(
                f"{method_name!r} is not a @distributed method of {_owning_class(self.role_cls).__name__}"
            ) from None

        refs, worker_local, leases = self._launch_call(
            method_name,
            dispatch_mode,
            dispatch_fn,
            execute_fn,
            args,
            kwargs,
            grad_mode=False,
            call_id=None,
        )
        return PendingHandleCall(self, method_name, refs, worker_local, leases=leases)

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

    def _rebind_tree(self, obj, worker_handle, *, worker_local: bool = True):
        """Rebind every ref leaf onto ``worker_handle`` and wrap bare handles in TensorRef."""

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
