"""TensorTransport — the backend-agnostic storage/retrieval contract.

The ABC every backend implements: per-tensor ``put`` / ``get`` (+ optional
batched overrides), the tree-walking ``dehydrate`` / ``hydrate`` boundary, a
``session`` context for cross-object batching, and a generic ``transform`` for
remote compute. Worker-resident backends add their machinery via
``unirl.distributed.tensor.worker_local.WorkerLocalTransport``;
``TensorTransportRuntime`` is the per-process singleton call sites reach at
runtime. The ``TensorRef`` / ``TensorSpan`` data model lives in
``unirl.distributed.tensor.ref``.
"""

from __future__ import annotations

import abc
import logging
from contextlib import contextmanager
from dataclasses import fields as dc_fields
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch

from unirl.distributed.tensor.batch import Batch
from unirl.distributed.tensor.ref import TensorRef, TensorSpan, cat_rows

logger = logging.getLogger(__name__)


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
    def _resolve_handles(self, handles: List[Any]) -> List[torch.Tensor]:
        """Resolve each handle to its full dim-0 base tensor, aligned to *handles*.

        The single backend-specific resolution primitive. Backends batch and dedup
        internally (gpu: one ``batch_borrow`` + IPC-view cache; tq: one ``_fetch``
        per put-group; colocate: per-tensor ``store.get``). ``get`` / ``get_batch``
        below slice each span's ``[start:stop)`` rows off these bases — the slice is
        universal, so it lives on the base, not in the backends.
        """
        ...

    def get(self, spans: List[Any]) -> torch.Tensor:
        """Resolve each span (its handle, sliced to ``[start:stop)``) and cat along dim 0."""
        if not spans:
            raise ValueError("get: empty spans list")
        bases = self._resolve_handles([s.handle for s in spans])
        return cat_rows([b[s.start : s.stop] for s, b in zip(spans, bases)])

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
        """Fetch multiple named tensors, batching handle resolution across ALL keys.

        Flatten every key's spans into one ``_resolve_handles`` call (one borrow /
        fetch for the whole object), then slice + cat per key. An empty per-key span
        list yields ``cat_rows([]) -> empty(0)``.
        """
        flat: List[Any] = []
        owners: List[str] = []
        for k, m in metas.items():
            for s in m.spans:
                flat.append(s)
                owners.append(k)
        bases = self._resolve_handles([s.handle for s in flat])
        parts: Dict[str, List[torch.Tensor]] = {k: [] for k in metas}
        for k, s, b in zip(owners, flat, bases):
            parts[k].append(b[s.start : s.stop])
        return {k: cat_rows(parts[k]) for k in metas}

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


__all__ = ["TensorTransport", "TensorTransportRuntime", "TransportSession"]
