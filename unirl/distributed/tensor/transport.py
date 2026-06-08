"""TensorTransport: backend-agnostic tensor storage and retrieval.

``TensorMeta`` is the universal tensor proxy — a ``Batch`` subclass that
holds per-handle opaque refs and their sizes. Each ref corresponds to one
``put()`` call; ``sizes[i]`` records how many rows that ref covers.
``concat`` merges ref/size lists; ``batch_size`` is ``sum(sizes)``.
Per-row ``select`` / ``slice`` is not supported on dehydrated data —
hydrate first.

``TensorMeta`` also serves as a compute proxy: ``transform``, ``reshape``,
``permute``, ``local`` delegate to the active backend.

``TensorTransport`` is the ABC every backend implements: per-tensor ``put`` /
``get``, optional ``put_batch`` / ``get_batch`` overrides, and a generic
``transform`` for remote compute. It also owns the tree-walking
``dehydrate`` / ``hydrate`` methods and a ``session`` context manager for
cross-object batching.

``TensorTransportRuntime`` is the per-process singleton that call sites
reach for at runtime.
"""

from __future__ import annotations

import abc
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional, Set, Tuple

import ray
import torch

from unirl.distributed.tensor.batch import Batch, concat_field, shared_field

logger = logging.getLogger(__name__)


@dataclass
class TensorMeta(Batch):
    """Per-handle tensor proxy.

    Each element in ``refs`` is an opaque handle from a single ``put()``
    call; ``sizes[i]`` records how many rows that handle covers.
    ``batch_size`` is ``sum(sizes)``, not ``len(refs)``.
    """

    refs: List[Any] = concat_field(default_factory=list)
    sizes: List[int] = concat_field(default_factory=list)
    shape: Optional[Tuple[int, ...]] = shared_field(default=None)
    dtype: Optional[torch.dtype] = shared_field(default=None)
    device: Optional[str] = shared_field(default=None)
    grad: Optional["TensorMeta"] = shared_field(default=None)
    retain_grad_flag: bool = shared_field(default=False)

    @property
    def batch_size(self) -> int:
        return sum(self.sizes) if self.sizes else 0

    @classmethod
    def concat(cls, items: "list[TensorMeta]") -> "TensorMeta":
        refs: List[Any] = []
        sizes: List[int] = []
        for m in items:
            refs.extend(m.refs)
            sizes.extend(m.sizes)
        first = items[0]
        total = sum(sizes)
        return TensorMeta(
            refs=refs,
            sizes=sizes,
            shape=(total, *first.shape[1:]) if first.shape else None,
            dtype=first.dtype,
            device=first.device,
        )

    def select(self, indices):
        raise NotImplementedError("TensorMeta does not support select — hydrate first")

    def _slice_by_refs(self, start: int, end: int) -> "TensorMeta":
        """Partition refs for the row range ``[start:end)`` — inverse of concat.

        ``concat`` stacks per-source ref/size lists, so a range that lands on
        ref boundaries hands the corresponding refs back untouched: the exact
        structural inverse of the DP collect→re-dispatch round-trip that built
        this handle. Both callers pass a range in the same unit as ``sizes``
        (CONCAT: per-sample rows; PACKED: token offsets via ``cu_seqlens``), so
        a round-trip range always aligns. An arbitrary intra-ref range has no
        representation here — the data is remote and opaque — and still needs
        hydration first.
        """
        start, end = int(start), int(end)
        offsets = [0]
        for s in self.sizes:
            offsets.append(offsets[-1] + int(s))
        try:
            i0 = offsets.index(start)
            i1 = offsets.index(end)
        except ValueError:
            raise NotImplementedError(
                f"TensorMeta slice [{start}:{end}] does not align to ref boundaries "
                f"{offsets}; intra-handle slicing requires hydration first."
            )
        refs = list(self.refs[i0:i1])
        sizes = list(self.sizes[i0:i1])
        total = sum(int(s) for s in sizes)
        return TensorMeta(
            refs=refs,
            sizes=sizes,
            shape=(total, *self.shape[1:]) if self.shape else None,
            dtype=self.dtype,
            device=self.device,
        )

    def slice(self, start, end) -> "TensorMeta":
        # CONCAT path: Batch.slice → _slice_value → value.slice(start, end).
        return self._slice_by_refs(start, end)

    def __getitem__(self, key) -> "TensorMeta":
        # PACKED path: _slice_packed_data does ``value[cu[start]:cu[end]]``.
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise NotImplementedError("TensorMeta supports only contiguous (step=1) slicing")
            lo = 0 if key.start is None else int(key.start)
            hi = self.batch_size if key.stop is None else int(key.stop)
            return self._slice_by_refs(lo, hi)
        raise NotImplementedError(f"TensorMeta indexing supports slices only, got {type(key).__name__}")

    def transform(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "TensorMeta":
        backend = TensorTransportRuntime.current()
        if backend is None:
            raise RuntimeError("No TensorTransport backend installed")
        return backend.transform(self, fn)

    def reshape(self, *shape: int) -> "TensorMeta":
        return self.transform(lambda t: t.reshape(*shape))

    def permute(self, *dims: int) -> "TensorMeta":
        return self.transform(lambda t: t.permute(*dims))

    def local(self) -> torch.Tensor:
        backend = TensorTransportRuntime.current()
        if backend is None:
            raise RuntimeError("No TensorTransport backend installed")
        return backend.get(self.refs)

    def retain_grad(self) -> "TensorMeta":
        self.retain_grad_flag = True
        return self

    @classmethod
    def from_handles(cls, handles: list) -> "TensorMeta":
        return cls(
            refs=handles,
            sizes=[h.shape[0] for h in handles],
            shape=(sum(h.shape[0] for h in handles), *handles[0].shape[1:]) if handles else None,
            dtype=handles[0].dtype if handles else None,
            device=str(handles[0].device) if handles else None,
        )

    def __len__(self) -> int:
        return sum(self.sizes) if self.sizes else 0


# ---------------------------------------------------------------------------
# Type-based tree walker
# ---------------------------------------------------------------------------


def _collect_leaves(
    value: Any,
    prefix: str,
    leaf_type: type,
    collected: Dict[str, Any],
    setters: Dict[str, Callable[[Any], None]],
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> None:
    """Walk *value* recursively, collect leaves of *leaf_type*.

    For each leaf found, stores the value in *collected* keyed by its
    dotted path, and a setter closure in *setters* that can write a
    replacement back into the original structure.

    Dispatch:
      - ``Batch``  -> recurse into ``dataclasses.fields``
      - ``dict``   -> recurse into values
      - ``list``   -> recurse into elements
      - leaf_type  -> collect
      - else       -> skip
    """
    if isinstance(value, leaf_type):
        if filter_fn is None or filter_fn(prefix):
            collected[prefix] = value
        return

    if isinstance(value, Batch):
        for f in dc_fields(value):
            v = getattr(value, f.name)
            if v is None:
                continue
            key = f"{prefix}.{f.name}" if prefix else f.name
            if isinstance(v, leaf_type):
                if filter_fn is None or filter_fn(key):
                    collected[key] = v
                    _owner, _attr = value, f.name
                    setters[key] = lambda val, o=_owner, a=_attr: setattr(o, a, val)
            elif isinstance(v, Batch):
                _collect_leaves(v, key, leaf_type, collected, setters, filter_fn)
            elif isinstance(v, dict):
                _collect_dict(v, key, leaf_type, collected, setters, filter_fn)
            elif isinstance(v, list):
                _collect_list(v, key, leaf_type, collected, setters, filter_fn)
    elif isinstance(value, dict):
        _collect_dict(value, prefix, leaf_type, collected, setters, filter_fn)
    elif isinstance(value, list):
        _collect_list(value, prefix, leaf_type, collected, setters, filter_fn)


def _collect_dict(
    d: dict,
    prefix: str,
    leaf_type: type,
    collected: Dict[str, Any],
    setters: Dict[str, Callable[[Any], None]],
    filter_fn: Optional[Callable[[str], bool]],
) -> None:
    for dk, dv in d.items():
        subkey = f"{prefix}.{dk}" if prefix else str(dk)
        if isinstance(dv, leaf_type):
            if filter_fn is None or filter_fn(subkey):
                collected[subkey] = dv
                _d, _k = d, dk
                setters[subkey] = lambda val, dd=_d, kk=_k: dd.__setitem__(kk, val)
        elif isinstance(dv, Batch):
            _collect_leaves(dv, subkey, leaf_type, collected, setters, filter_fn)
        elif isinstance(dv, dict):
            _collect_dict(dv, subkey, leaf_type, collected, setters, filter_fn)
        elif isinstance(dv, list):
            _collect_list(dv, subkey, leaf_type, collected, setters, filter_fn)


def _collect_list(
    lst: list,
    prefix: str,
    leaf_type: type,
    collected: Dict[str, Any],
    setters: Dict[str, Callable[[Any], None]],
    filter_fn: Optional[Callable[[str], bool]],
) -> None:
    for i, elem in enumerate(lst):
        if isinstance(elem, Batch):
            subkey = f"{prefix}.{elem._eid}" if prefix else elem._eid
        else:
            subkey = f"{prefix}.{i}" if prefix else str(i)
        if isinstance(elem, leaf_type):
            if filter_fn is None or filter_fn(subkey):
                collected[subkey] = elem
                _l, _i = lst, i
                setters[subkey] = lambda val, ll=_l, ii=_i: ll.__setitem__(ii, val)
        elif isinstance(elem, Batch):
            _collect_leaves(elem, subkey, leaf_type, collected, setters, filter_fn)
        elif isinstance(elem, dict):
            _collect_dict(elem, subkey, leaf_type, collected, setters, filter_fn)
        elif isinstance(elem, list):
            _collect_list(elem, subkey, leaf_type, collected, setters, filter_fn)


# ---------------------------------------------------------------------------
# TensorTransport ABC
# ---------------------------------------------------------------------------


def _apply_tensor_op(t: torch.Tensor, op: str, *args) -> torch.Tensor:
    """Apply a named tensor op. Shared by the default ``tensor_op`` round-trip."""
    if op == "getitem":
        return t[args[0]]
    if op == "reshape":
        return t.reshape(args[0])
    if op == "permute":
        return t.permute(args[0])
    raise ValueError(f"Unknown tensor op: {op!r}")


class TensorTransport(abc.ABC):
    """Backend-agnostic tensor transport — the universal contract.

    Store/fetch refs (``put``/``get``/``is_ref``), the batched + tree-walking
    boundary helpers (``put_batch``/``get_batch``/``dehydrate``/``hydrate``/
    ``session``), and the compute proxy (``transform``). Worker-resident
    backends add storage-engine machinery via :class:`WorkerLocalTransport`.
    """

    @abc.abstractmethod
    def put(self, tensor: torch.Tensor) -> Any:
        """Store tensor, return a single opaque ref (handle)."""
        ...

    @abc.abstractmethod
    def get(self, refs: List[Any]) -> torch.Tensor:
        """Fetch tensors for each ref and cat along dim 0."""
        ...

    @abc.abstractmethod
    def is_ref(self, value: Any) -> bool:
        """True if *value* is a ``TensorMeta`` produced by this backend."""
        ...

    def put_batch(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, TensorMeta]:
        """Store multiple named tensors. Default: iterate per key."""
        result: Dict[str, TensorMeta] = {}
        for k, t in tensors.items():
            ref = self.put(t)
            bs = int(t.shape[0]) if isinstance(t, torch.Tensor) and t.dim() > 0 else 1
            result[k] = TensorMeta(
                refs=[ref],
                sizes=[bs],
                shape=tuple(t.shape) if isinstance(t, torch.Tensor) else None,
                dtype=t.dtype if isinstance(t, torch.Tensor) else None,
                device=str(t.device) if isinstance(t, torch.Tensor) else None,
            )
        return result

    def get_batch(self, metas: Dict[str, TensorMeta]) -> Dict[str, torch.Tensor]:
        """Fetch multiple named tensors. Default: iterate per key."""
        return {k: self.get(m.refs) for k, m in metas.items()}

    def transform(self, meta: TensorMeta, fn: Callable[[torch.Tensor], torch.Tensor]) -> TensorMeta:
        """Apply fn to the remote tensor, return new TensorMeta.

        Default: hydrate -> apply fn -> dehydrate (round-trip through local
        memory). Backends with remote compute (TensorStore) can override to
        execute on the worker without moving data.
        """
        tensor = self.get(meta.refs)
        result = fn(tensor)
        ref = self.put(result)
        bs = int(result.shape[0]) if result.dim() > 0 else 1
        return TensorMeta(
            refs=[ref],
            sizes=[bs],
            shape=tuple(result.shape),
            dtype=result.dtype,
            device=str(result.device),
        )

    def end_call(self) -> None:
        """Release any per-call resources (e.g. open IPC views). No-op default.

        Called by the Worker after each call() completes; backends with per-call
        state (gpu IPC views) override it.
        """

    @classmethod
    def localize(cls, shards: list, pool: Any, device_ids: list, worker_ids: list) -> list:
        """Make every ref in each shard resolvable on its target worker.

        Base (GLOBAL) backends: identity — a ref resolves from any process, so no
        controller-orchestrated transfer is needed. WorkerLocalTransport overrides
        with the NCCL/IPC routing skeleton. ``pool`` (topology) and the per-shard
        ``device_ids``/``worker_ids`` (dst identity) are unused here.
        """
        return shards

    # ---- dehydrate / hydrate ------------------------------------------------

    def dehydrate(self, value: Any) -> Any:
        """Replace tensors with ``TensorMeta`` refs.

        - ``torch.Tensor`` -> returns ``TensorMeta``
        - ``Batch`` / ``dict`` / ``list`` -> mutates in place, returns *value*
        - anything else -> returns *value* unchanged
        """
        if isinstance(value, torch.Tensor):
            ref = self.put(value)
            bs = int(value.shape[0]) if value.dim() > 0 else 1
            return TensorMeta(
                refs=[ref],
                sizes=[bs],
                shape=tuple(value.shape),
                dtype=value.dtype,
                device=str(value.device),
            )

        tensors: Dict[str, torch.Tensor] = {}
        setters: Dict[str, Callable[[Any], None]] = {}
        _collect_leaves(value, "", torch.Tensor, tensors, setters)
        if not tensors:
            return value

        metas = self.put_batch(tensors)
        for key, meta in metas.items():
            setters[key](meta)
        return value

    def hydrate(self, value: Any, fields: Optional[Set[str]] = None) -> Any:
        """Replace ``TensorMeta`` refs with tensors.

        - ``TensorMeta`` -> returns ``torch.Tensor``
        - ``Batch`` / ``dict`` / ``list`` -> mutates in place, returns *value*
        - anything else -> returns *value* unchanged

        If *fields* is given, only dotted-path keys matching a prefix in
        *fields* are hydrated; the rest stay as ``TensorMeta``.
        """
        if isinstance(value, TensorMeta):
            return self.get(value.refs)

        filter_fn: Optional[Callable[[str], bool]] = None
        if fields is not None:

            def filter_fn(key):
                return any(key == f or key.startswith(f + ".") for f in fields)

        meta_map: Dict[str, TensorMeta] = {}
        setters: Dict[str, Callable[[Any], None]] = {}
        _collect_leaves(value, "", TensorMeta, meta_map, setters, filter_fn)
        if not meta_map:
            return value

        tensors = self.get_batch(meta_map)
        for key, tensor in tensors.items():
            if key in setters:
                setters[key](tensor)
        return value

    # ---- session ------------------------------------------------------------

    @contextmanager
    def session(self) -> Iterator["TransportSession"]:
        """Batched dehydrate context.

        Collects tensors across multiple ``dehydrate()`` calls and flushes
        via ``put_batch`` per object on ``__exit__``.  Hydrate is immediate
        (no batching benefit from deferring).
        """
        sess = TransportSession(self)
        try:
            yield sess
        finally:
            sess._flush()


def map_tree(obj: Any, leaf_fn: Callable[[Any], Any]) -> Any:
    """Rebuild a value tree, applying ``leaf_fn`` to every node.

    The single tree-walker shared by the transport layer's rewrite passes
    (controller-side ``localize`` substitution and ``Handle._rebind_tree``,
    worker-side resolve/pack in ``Worker.call``). ``leaf_fn`` runs on every node
    first; if it returns a *different* object that replaces the node and recursion
    stops there. Otherwise containers are rebuilt structurally — ``Batch`` via
    :meth:`Batch._rebuild` (preserving framework-managed ``_packed_cu_seqlens``),
    ``tuple`` / ``list`` / ``dict`` element-wise. ``TensorMeta`` is an atomic leaf
    (never recursed into, despite being a ``Batch`` subclass); any other
    non-container value passes through. Functional (returns new trees), so it works
    on immutable tuples and lets each caller's ``leaf_fn`` decide what to swap.
    """
    new = leaf_fn(obj)
    if new is not obj:
        return new
    if isinstance(obj, TensorMeta):
        return obj
    if isinstance(obj, Batch):
        return obj._rebuild({f.name: map_tree(getattr(obj, f.name), leaf_fn) for f in dc_fields(obj)})
    if isinstance(obj, tuple):
        return tuple(map_tree(item, leaf_fn) for item in obj)
    if isinstance(obj, list):
        return [map_tree(item, leaf_fn) for item in obj]
    if isinstance(obj, dict):
        return {k: map_tree(v, leaf_fn) for k, v in obj.items()}
    return obj


class WorkerLocalTransport(TensorTransport):
    """The V2 Worker/Handle storage contract — worker-resident backends only.

    Adds the storage-engine machinery only a worker-resident store needs:
    ref-count lifecycle (``incref``/``decref``), controller-orchestrated
    cross-worker transfer (``setup_transfer``/``nccl_send``/``nccl_recv``), and
    on-worker remote compute (``tensor_op``/``cat``/``get_cpu``). The universal
    materialization surface (``get_batch``/``put_batch``/``end_call``) lives on
    the base. GLOBAL backends (e.g. the transfer queue) are plain
    :class:`TensorTransport` and implement none of this capability.

    ``isinstance(t, WorkerLocalTransport)`` is the locality discriminator the
    controller uses to decide whether cross-worker routing is required.
    """

    # Methods the controller may invoke on this transport via the Worker actor's
    # ``transport_op`` relay (TensorHandle GC/compute + Handle NCCL routing).
    # Adding a capability method below means adding its name here. Excludes
    # setup_transfer, which the Worker injects identity into via setup_global_pg.
    REMOTE_OPS: ClassVar[frozenset] = frozenset({"incref", "decref", "tensor_op", "get_cpu", "nccl_send", "nccl_recv"})

    # ---- lifecycle (ref-counting) ------------------------------------------

    def incref(self, key: Any) -> None:
        """Increment the ref count for a stored tensor. No-op by default."""

    def decref(self, key: Any) -> None:
        """Decrement the ref count; free at zero. No-op by default."""

    # ---- locality + cross-worker transfer (localize) -------------------

    def setup_transfer(self, global_rank: int, world_size: int) -> None:
        """Initialize the cross-worker transfer group."""

    def nccl_send(self, dst_rank: int, handles: List[Any]) -> None:
        raise NotImplementedError("transport does not support cross-worker send")

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        raise NotImplementedError("transport does not support cross-worker recv")

    @classmethod
    def _is_local(cls, ref: Any, dst_worker_id: str, dst_device_id: int, pool: Any) -> bool:
        """True if ``ref`` is already resolvable on the dst worker (no transfer needed).

        The one per-backend locality decision. Base: a ref is local only if produced by
        the dst worker (per-process store). gpu overrides to also accept same physical
        device, since its per-GPU TensorWorker is shared across that GPU's slots.
        """
        return ref.source_id == dst_worker_id

    @classmethod
    def localize(cls, shards: list, pool: Any, device_ids: List[int], worker_ids: List[str]) -> list:
        """Make every ref in each shard resolvable on its dst worker.

        Shared skeleton for all worker-local backends; the only thing that varies is
        ``_is_local`` (the locality predicate). A ref that is not already local and not
        an object_ref (CPU/plasma resolves anywhere) is moved cross-device via one
        batched NCCL hop between the two devices' slot0 workers, then substituted back
        (id()-keyed). Names no backend type — works through ``ref.routing_copy`` /
        ``ref.source_id`` / ``map_tree`` / the ``transport_op`` relay.
        """
        foreign: Dict[Tuple[int, int], List[Any]] = {}  # (src_device_id, dst_device_id) → [routing_copy, ...]

        def route(ref: Any, dst_worker_id: str, dst_device_id: int) -> Any:
            if getattr(ref, "object_ref", None) is not None:
                return ref  # CPU/plasma → resolvable anywhere
            if cls._is_local(ref, dst_worker_id, dst_device_id, pool):
                return ref
            src_device_id = pool.device_id_of(ref.source_id)
            routing = ref.routing_copy()
            foreign.setdefault((src_device_id, dst_device_id), []).append(routing)
            return routing

        def unwrap(obj: Any, dst_worker_id: str, dst_device_id: int) -> Any:
            if isinstance(obj, TensorMeta):
                return TensorMeta.from_handles([route(h, dst_worker_id, dst_device_id) for h in obj.refs])
            return obj

        routed: list = []
        for i, (s_args, s_kwargs) in enumerate(shards):

            def leaf(o, _w=worker_ids[i], _d=device_ids[i]):
                return unwrap(o, _w, _d)

            routed.append((map_tree(s_args, leaf), map_tree(s_kwargs, leaf)))

        if not foreign:
            return routed

        keys = list(foreign.keys())
        send_refs, recv_refs = [], []
        for src_device_id, dst_device_id in keys:
            handles = foreign[(src_device_id, dst_device_id)]
            send_refs.append(pool.slot0_worker(src_device_id).transport_op.remote("nccl_send", dst_device_id, handles))
            recv_refs.append(
                pool.slot0_worker(dst_device_id).transport_op.remote(
                    "nccl_recv", src_device_id, [h.shape for h in handles], [h.dtype for h in handles]
                )
            )
        ray.get(send_refs)
        recv_results = ray.get(recv_refs)

        subs: Dict[int, Any] = {}
        for (src_device_id, dst_device_id), new_handles in zip(keys, recv_results):
            dst_worker = pool.slot0_worker(dst_device_id)
            for old_h, new_h in zip(foreign[(src_device_id, dst_device_id)], new_handles):
                new_h.rebind(dst_worker)
                subs[id(old_h)] = new_h

        def substitute(obj: Any) -> Any:
            if isinstance(obj, TensorMeta):
                return TensorMeta.from_handles([subs.get(id(h), h) for h in obj.refs])
            return obj

        return [(map_tree(a, substitute), map_tree(k, substitute)) for a, k in routed]

    # ---- remote compute (controller-triggered) ----------------------------

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        """Apply a named op (getitem/reshape/permute) to a stored tensor.

        Default: round-trip get -> op -> put. Backends with on-worker compute
        override to avoid moving data.
        """
        result = _apply_tensor_op(self.get([handle]), op, *op_args).contiguous()
        return self.put(result)

    def get_cpu(self, handle: Any) -> torch.Tensor:
        """Return the stored tensor as a CPU tensor."""
        return self.get([handle]).cpu()


class TransportSession:
    """Accumulates dehydrate calls; flushes on close."""

    def __init__(self, backend: TensorTransport) -> None:
        self._backend = backend
        self._pending: List[Tuple[Dict[str, torch.Tensor], Dict[str, Callable[[Any], None]]]] = []

    def dehydrate(self, value: Any) -> Any:
        """Replace tensors with ``TensorMeta`` refs (deferred flush).

        Bare ``torch.Tensor`` is handled immediately (caller needs return
        value). ``Batch`` / ``dict`` / ``list`` are collected; the actual
        ``put_batch`` happens when the session closes.
        """
        if isinstance(value, torch.Tensor):
            return self._backend.dehydrate(value)

        tensors: Dict[str, torch.Tensor] = {}
        setters: Dict[str, Callable[[Any], None]] = {}
        _collect_leaves(value, "", torch.Tensor, tensors, setters)
        if tensors:
            self._pending.append((tensors, setters))
        return value

    def hydrate(self, value: Any, fields: Optional[Set[str]] = None) -> Any:
        """Immediate hydrate (delegates to backend)."""
        return self._backend.hydrate(value, fields)

    def _flush(self) -> None:
        for tensors, setters in self._pending:
            metas = self._backend.put_batch(tensors)
            for key, meta in metas.items():
                setters[key](meta)
        self._pending.clear()


class TensorTransportRuntime:
    """Per-process active backend singleton."""

    _current: Optional[TensorTransport] = None

    @classmethod
    def current(cls) -> Optional[TensorTransport]:
        return cls._current

    @classmethod
    def install(cls, backend: TensorTransport) -> TensorTransport:
        if cls._current is not None and cls._current is not backend:
            logger.warning("TensorTransportRuntime: replacing existing backend")
        cls._current = backend
        return backend

    @classmethod
    def clear_current(cls) -> None:
        cls._current = None


__all__ = [
    "TensorMeta",
    "TensorTransport",
    "TensorTransportRuntime",
    "TransportSession",
    "WorkerLocalTransport",
    "map_tree",
]
