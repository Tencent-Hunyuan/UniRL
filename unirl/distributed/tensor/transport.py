"""TensorTransport: backend-agnostic tensor storage and retrieval.

``TensorRef`` is the universal tensor proxy — a ``Batch`` subclass that
holds an ordered list of :class:`TensorSpan`, each a contiguous row-window
over one backend handle (from a single ``put()`` call). ``spans`` is the
concat axis; per-span row counts derive ``sizes`` and ``batch_size`` (no
stored size array). Per-row ``select`` / ``slice`` builds new spans — no data
motion, no hydration. A :class:`TensorSpan` is generic over its handle type,
bound by the :class:`TensorHandle` protocol.

``TensorRef`` also serves as a compute proxy: ``transform``, ``reshape``,
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
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import ray
import torch

from unirl.distributed.tensor.batch import Batch, concat_field, shared_field

logger = logging.getLogger(__name__)


@runtime_checkable
class TensorHandle(Protocol):
    """The minimal handle contract a :class:`TensorSpan` resolves through.

    The worker-local store handle (``GPUTensorHandle`` / ``ColocateTensorHandle``)
    and the global ``TQTensorHandle`` both satisfy it structurally. The
    worker-local-only surface (``store_key`` / ``source_id`` / ``object_ref`` /
    ``routing_copy``) is reached via ``span.handle`` in the backends that need it,
    never off the span — so it is intentionally absent from this protocol.
    """

    def local(self) -> torch.Tensor: ...


T = TypeVar("T", bound=TensorHandle)


class TensorSpan(Generic[T]):
    """A contiguous half-open ``[start:stop)`` row-window over one handle — the addressing primitive.

    ``TensorRef`` selection/permutation emits these instead of moving data:
    a span addresses ``handle[start:stop)`` along dim 0 while the bytes stay in
    the producing worker's store. Backends resolve a span by resolving the
    handle and slicing (zero-copy on the IPC/store path); ``localize`` ships
    only the ``[start:stop)`` rows cross-device, not the whole handle block.

    Lifecycle: a span holds a PYTHON REFERENCE to the handle object, so
    CPython's own reference counting aggregates all spans' lifetimes onto the
    handle — the handle's single GC finalizer fires only when the last span
    (and the handle itself) is gone. Spans carry no finalizer, no store_key
    of their own, and trigger no extra decref RPCs.

    Nested spans flatten at construction (``TensorSpan(TensorSpan(h,2,8),1,3)``
    is ``TensorSpan(h,3,5)``), so ``handle`` is always a backend handle.
    """

    __slots__ = ("handle", "start", "stop")
    handle: T

    def __init__(self, handle: T | "TensorSpan[T]", start: int, stop: int) -> None:
        start, stop = int(start), int(stop)
        if isinstance(handle, TensorSpan):
            start, stop = handle.start + start, handle.start + stop
            handle = handle.handle
        if not (0 <= start <= stop):
            raise ValueError(f"TensorSpan range [{start}, {stop}) is invalid")
        self.handle = handle
        self.start = start
        self.stop = stop

    @property
    def nrows(self) -> int:
        return self.stop - self.start

    def __len__(self) -> int:
        return self.stop - self.start

    # ── delegated metadata: shape (sliced) / dtype / device — read by nccl_recv
    #    and repr. Handle-specific identity (store_key/source_id/object_ref) is
    #    reached via ``span.handle`` in the worker-local backends, not off the span.

    @property
    def shape(self) -> tuple:
        handle_shape = tuple(self.handle.shape)
        return (self.stop - self.start, *handle_shape[1:])

    @property
    def dtype(self):
        return self.handle.dtype

    @property
    def device(self):
        return self.handle.device

    def local(self) -> torch.Tensor:
        return self.handle.local()[self.start : self.stop]

    def routing_copy(self) -> "TensorSpan":
        """Bare placeholder for localize's id()-keyed NCCL substitution.

        Carries the handle's routing copy plus this span's range so the send
        side can slice — dropping it must not touch the handle's ref count
        (the handle's routing_copy is equally bare).
        """
        return TensorSpan(self.handle.routing_copy(), self.start, self.stop)

    def __getstate__(self) -> dict:
        return {"handle": self.handle, "start": self.start, "stop": self.stop}

    def __setstate__(self, state: dict) -> None:
        self.handle = state["handle"]
        self.start = state["start"]
        self.stop = state["stop"]

    def __repr__(self) -> str:
        return f"TensorSpan({self.handle!r}[{self.start}:{self.stop}])"


def cat_rows(parts: List[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-ref tensors along dim 0 — the single assembly funnel.

    Trailing-dim CONTRACT: parts may be padded to different widths per
    producing shard (e.g. per-worker prompt blocks); 2D+ parts are
    right-padded with zeros to the max width before the cat — consumers of
    2D+ per-shard-padded fields must be mask-driven (the convention
    ``TextTokenCondition.concat`` already establishes).
    """
    if not parts:
        return torch.empty(0)
    if len(parts) == 1:
        return parts[0]
    if parts[0].dim() >= 2:
        widths = {int(t.shape[1]) for t in parts}
        if len(widths) > 1:
            target = max(widths)
            padded = []
            for t in parts:
                if int(t.shape[1]) < target:
                    pad = t.new_zeros((t.shape[0], target - t.shape[1]) + tuple(t.shape[2:]))
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            parts = padded
    return torch.cat(parts, dim=0)


@dataclass
class TensorRef(Batch):
    """The dehydrated-tensor proxy — an ordered list of row-window spans.

    Each element of ``spans`` is a :class:`TensorSpan` over one backend handle;
    ``spans`` is the concat axis. Per-span row counts derive ``sizes`` and
    ``batch_size`` — there is no stored size array. ``batch_size`` is the total
    row count, not ``len(spans)``.
    """

    spans: List[TensorSpan] = concat_field(default_factory=list)
    shape: Optional[Tuple[int, ...]] = shared_field(default=None)
    dtype: Optional[torch.dtype] = shared_field(default=None)
    device: Optional[str] = shared_field(default=None)
    grad: Optional["TensorRef"] = shared_field(default=None)
    retain_grad_flag: bool = shared_field(default=False)

    @property
    def sizes(self) -> List[int]:
        return [s.stop - s.start for s in self.spans]

    @property
    def batch_size(self) -> int:
        return sum(s.stop - s.start for s in self.spans)

    @classmethod
    def concat(cls, items: "list[TensorRef]") -> "TensorRef":
        spans: List[TensorSpan] = []
        for m in items:
            spans.extend(m.spans)
        first = items[0]
        total = sum(s.stop - s.start for s in spans)
        return TensorRef(
            spans=spans,
            shape=(total, *first.shape[1:]) if first.shape else None,
            dtype=first.dtype,
            device=first.device,
        )

    def select(self, indices) -> "TensorRef":
        """Re-index along the unit axis by building ref VIEWS (no data motion)."""
        idx = [int(i) for i in (indices.tolist() if hasattr(indices, "tolist") else indices)]
        return self.select_units(idx)

    def _offsets(self) -> List[int]:
        offsets = [0]
        for s in self.spans:
            offsets.append(offsets[-1] + (s.stop - s.start))
        return offsets

    def _pieces_for_range(self, g0: int, g1: int, offsets: List[int]) -> List[Tuple[int, int, int]]:
        """Global unit range [g0, g1) -> ordered (ref_idx, local_start, local_end) pieces."""
        if not (0 <= g0 <= g1 <= offsets[-1]):
            raise IndexError(f"unit range [{g0}, {g1}) out of bounds for size {offsets[-1]}")
        pieces: List[Tuple[int, int, int]] = []
        r = 0
        while g0 < g1:
            while offsets[r + 1] <= g0:
                r += 1
            take = min(g1, offsets[r + 1]) - g0
            pieces.append((r, g0 - offsets[r], g0 - offsets[r] + take))
            g0 += take
        return pieces

    def _from_pieces(self, pieces: List[Tuple[int, int, int]]) -> "TensorRef":
        """Build the selected ref: a whole-span piece passes the span through
        untouched (a boundary-aligned selection costs nothing); a partial piece
        becomes a new :class:`TensorSpan` (nested spans flatten in the ctor)."""
        spans: List[TensorSpan] = []
        for r, s, e in pieces:
            src = self.spans[r]
            if s == 0 and e == src.stop - src.start:
                spans.append(src)
            else:
                spans.append(TensorSpan(src, s, e))
        total = sum(sp.stop - sp.start for sp in spans)
        return TensorRef(
            spans=spans,
            shape=(total, *self.shape[1:]) if self.shape else None,
            dtype=self.dtype,
            device=self.device,
        )

    def select_units(self, idx: List[int]) -> "TensorRef":
        """Arbitrary re-index (gather/permute) as lazy ref views."""
        offsets = self._offsets()
        # Coalesce consecutive indices into ranges, then map ranges to pieces.
        pieces: List[Tuple[int, int, int]] = []
        i = 0
        while i < len(idx):
            j = i + 1
            while j < len(idx) and idx[j] == idx[j - 1] + 1:
                j += 1
            pieces.extend(self._pieces_for_range(int(idx[i]), int(idx[j - 1]) + 1, offsets))
            i = j
        return self._from_pieces(pieces)

    def select_segments(self, segments: List[Tuple[int, int]]) -> "TensorRef":
        """Re-index by global (start, end) unit ranges (PACKED token ranges)."""
        offsets = self._offsets()
        pieces: List[Tuple[int, int, int]] = []
        for g0, g1 in segments:
            pieces.extend(self._pieces_for_range(int(g0), int(g1), offsets))
        return self._from_pieces(pieces)

    def with_spans(self, spans: List[Any]) -> "TensorRef":
        """Clone with substituted (routed) spans, preserving shape/dtype/device.

        ``localize`` rebuilds refs after routing; this keeps the substitution
        structural (same span count, derived sizes, no re-derivation).
        """
        return TensorRef(
            spans=list(spans),
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def _slice_by_refs(self, start: int, end: int) -> "TensorRef":
        """Contiguous row range ``[start:end)`` — the structural inverse of concat.

        A range on ref boundaries hands the refs back untouched (the exact
        inverse of the DP collect→re-dispatch round-trip); an intra-ref range
        wraps the boundary refs in :class:`TensorSpan` — same code path, the
        whole-piece pass-through in ``_from_pieces`` is what preserves the
        zero-cost aligned case.
        """
        return self.select_segments([(int(start), int(end))])

    def slice(self, start, end) -> "TensorRef":
        # CONCAT path: Batch.slice → _slice_value → value.slice(start, end).
        return self._slice_by_refs(start, end)

    def __getitem__(self, key) -> "TensorRef":
        # PACKED path: _slice_packed_data does ``value[cu[start]:cu[end]]``.
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise NotImplementedError("TensorRef supports only contiguous (step=1) slicing")
            lo = 0 if key.start is None else int(key.start)
            hi = self.batch_size if key.stop is None else int(key.stop)
            return self._slice_by_refs(lo, hi)
        raise NotImplementedError(f"TensorRef indexing supports slices only, got {type(key).__name__}")

    def transform(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "TensorRef":
        backend = TensorTransportRuntime.current()
        if backend is None:
            raise RuntimeError("No TensorTransport backend installed")
        return backend.transform(self, fn)

    def reshape(self, *shape: int) -> "TensorRef":
        return self.transform(lambda t: t.reshape(*shape))

    def permute(self, *dims: int) -> "TensorRef":
        return self.transform(lambda t: t.permute(*dims))

    def local(self) -> torch.Tensor:
        return self.materialize()

    def materialize(self, backend: "Optional[TensorTransport]" = None) -> torch.Tensor:
        """Fetch this meta into a real tensor (driver- or worker-side).

        With a backend: one ``get`` over the spans (backends resolve a
        :class:`TensorSpan` by resolving its handle and slicing). Without:
        per-span ``local()`` round-trips assembled via :func:`cat_rows` (its
        ragged right-pad contract applies to 2D+ per-shard-padded fields).
        """
        if backend is None:
            backend = TensorTransportRuntime.current()
        if not self.spans:
            return torch.empty(0)
        if backend is not None:
            return backend.get(self.spans)
        return cat_rows([s.local() for s in self.spans])

    def retain_grad(self) -> "TensorRef":
        self.retain_grad_flag = True
        return self

    @classmethod
    def from_handles(cls, handles: list) -> "TensorRef":
        """Wrap freshly-put handles as full-range spans — the single wrap chokepoint."""
        spans = [TensorSpan(h, 0, int(h.shape[0])) for h in handles]
        return cls(
            spans=spans,
            shape=(sum(int(h.shape[0]) for h in handles), *handles[0].shape[1:]) if handles else None,
            dtype=handles[0].dtype if handles else None,
            device=str(handles[0].device) if handles else None,
        )

    def __len__(self) -> int:
        return self.batch_size


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
        """True if *value* is a ``TensorRef`` produced by this backend."""
        ...

    def put_batch(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, TensorRef]:
        """Store multiple named tensors. Default: iterate per key."""
        result: Dict[str, TensorRef] = {}
        for k, t in tensors.items():
            ref = self.put(t)
            bs = int(t.shape[0]) if isinstance(t, torch.Tensor) and t.dim() > 0 else 1
            result[k] = TensorRef(
                spans=[TensorSpan(ref, 0, bs)],
                shape=tuple(t.shape) if isinstance(t, torch.Tensor) else None,
                dtype=t.dtype if isinstance(t, torch.Tensor) else None,
                device=str(t.device) if isinstance(t, torch.Tensor) else None,
            )
        return result

    def get_batch(self, metas: Dict[str, TensorRef]) -> Dict[str, torch.Tensor]:
        """Fetch multiple named tensors. Default: iterate per key."""
        return {k: self.get(m.spans) for k, m in metas.items()}

    def transform(self, meta: TensorRef, fn: Callable[[torch.Tensor], torch.Tensor]) -> TensorRef:
        """Apply fn to the remote tensor, return new TensorRef.

        Default: hydrate -> apply fn -> dehydrate (round-trip through local
        memory). Backends with remote compute (TensorStore) can override to
        execute on the worker without moving data.
        """
        tensor = self.get(meta.spans)
        result = fn(tensor)
        ref = self.put(result)
        bs = int(result.shape[0]) if result.dim() > 0 else 1
        return TensorRef(
            spans=[TensorSpan(ref, 0, bs)],
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
        """Replace tensors with ``TensorRef`` refs.

        - ``torch.Tensor`` -> returns ``TensorRef``
        - ``Batch`` / ``dict`` / ``list`` -> mutates in place, returns *value*
        - anything else -> returns *value* unchanged
        """
        if isinstance(value, torch.Tensor):
            ref = self.put(value)
            bs = int(value.shape[0]) if value.dim() > 0 else 1
            return TensorRef(
                spans=[TensorSpan(ref, 0, bs)],
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
        """Replace ``TensorRef`` refs with tensors.

        - ``TensorRef`` -> returns ``torch.Tensor``
        - ``Batch`` / ``dict`` / ``list`` -> mutates in place, returns *value*
        - anything else -> returns *value* unchanged

        If *fields* is given, only dotted-path keys matching a prefix in
        *fields* are hydrated; the rest stay as ``TensorRef``.
        """
        if isinstance(value, TensorRef):
            return value.materialize(backend=self)

        filter_fn: Optional[Callable[[str], bool]] = None
        if fields is not None:

            def filter_fn(key):
                return any(key == f or key.startswith(f + ".") for f in fields)

        meta_map: Dict[str, TensorRef] = {}
        setters: Dict[str, Callable[[Any], None]] = {}
        _collect_leaves(value, "", TensorRef, meta_map, setters, filter_fn)
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
    ``tuple`` / ``list`` / ``dict`` element-wise. ``TensorRef`` is an atomic leaf
    (never recursed into, despite being a ``Batch`` subclass); any other
    non-container value passes through. Functional (returns new trees), so it works
    on immutable tuples and lets each caller's ``leaf_fn`` decide what to swap.
    """
    new = leaf_fn(obj)
    if new is not obj:
        return new
    if isinstance(obj, TensorRef):
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

        def route(span: Any, dst_worker_id: str, dst_device_id: int) -> Any:
            if getattr(span.handle, "object_ref", None) is not None:
                return span  # CPU/plasma → resolvable anywhere
            if cls._is_local(span.handle, dst_worker_id, dst_device_id, pool):
                return span
            src_device_id = pool.device_id_of(span.handle.source_id)
            routing = span.routing_copy()
            foreign.setdefault((src_device_id, dst_device_id), []).append(routing)
            return routing

        def unwrap(obj: Any, dst_worker_id: str, dst_device_id: int) -> Any:
            if isinstance(obj, TensorRef):
                # A foreign span ships ONLY its [start:stop) rows — the send side
                # slices by the routing copy's range (its .shape is sliced).
                return obj.with_spans([route(s, dst_worker_id, dst_device_id) for s in obj.spans])
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
            for old_span, new_h in zip(foreign[(src_device_id, dst_device_id)], new_handles):
                new_h.rebind(dst_worker)
                # The recv handle holds exactly the sliced rows → full-range span.
                subs[id(old_span)] = TensorSpan(new_h, 0, int(new_h.shape[0]))

        def substitute(obj: Any) -> Any:
            if isinstance(obj, TensorRef):
                return obj.with_spans([subs.get(id(s), s) for s in obj.spans])
            return obj

        return [(map_tree(a, substitute), map_tree(k, substitute)) for a, k in routed]

    # ---- remote compute (controller-triggered) ----------------------------

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        """Apply a named op (getitem/reshape/permute) to a stored tensor.

        Default: round-trip get -> op -> put. Backends with on-worker compute
        override to avoid moving data.
        """
        result = _apply_tensor_op(self.get([TensorSpan(handle, 0, int(handle.shape[0]))]), op, *op_args).contiguous()
        return self.put(result)

    def get_cpu(self, handle: Any) -> torch.Tensor:
        """Return the stored tensor as a CPU tensor."""
        return self.get([TensorSpan(handle, 0, int(handle.shape[0]))]).cpu()


class TransportSession:
    """Accumulates dehydrate calls; flushes on close."""

    def __init__(self, backend: TensorTransport) -> None:
        self._backend = backend
        self._pending: List[Tuple[Dict[str, torch.Tensor], Dict[str, Callable[[Any], None]]]] = []

    def dehydrate(self, value: Any) -> Any:
        """Replace tensors with ``TensorRef`` refs (deferred flush).

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
    "TensorHandle",
    "TensorSpan",
    "TensorRef",
    "TensorTransport",
    "TensorTransportRuntime",
    "TransportSession",
    "WorkerLocalTransport",
    "cat_rows",
    "map_tree",
]
