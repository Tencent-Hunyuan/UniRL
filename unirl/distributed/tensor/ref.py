"""TensorRef / TensorSpan — the dehydrated-tensor data model and addressing.

``TensorRef`` is the universal tensor proxy: a ``Batch`` subclass holding an
ordered list of :class:`TensorSpan`, each a contiguous row-window over one
backend handle. Per-row ``select`` / ``slice`` / ``select_ranges`` build new
spans (no data motion, no hydration); ``materialize`` hydrates
through the active backend. ``cat_rows`` is the per-span assembly funnel and
``map_tree`` the structural walker shared by the transport rewrite passes.

Pure data model — no Ray, no backend. ``TensorRef.transform`` / ``materialize``
reach the active backend via a function-local import of ``TensorTransportRuntime``
to avoid a module cycle with ``transport``.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dc_fields
from typing import TYPE_CHECKING, Any, Callable, Generic, List, Optional, Protocol, Tuple, TypeVar, runtime_checkable

import torch

from unirl.distributed.tensor.batch import Batch, concat_field, shared_field

if TYPE_CHECKING:
    from unirl.distributed.tensor.transport import TensorTransport


@runtime_checkable
class TensorHandle(Protocol):
    """The minimal handle contract a :class:`TensorSpan` resolves through.

    The worker-local store handle (``GPUTensorHandle`` / ``ColocateTensorHandle``)
    and the global ``TQTensorHandle`` both satisfy it structurally. The
    worker-local-only surface (``store_key`` / ``source_id`` / ``object_ref``) is
    reached via ``span.handle`` in the backends that need it, never off the span —
    so it is intentionally absent from this protocol.
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
        for item in items:
            spans.extend(item.spans)
        return items[0]._from_spans(spans)

    def select(self, indices) -> "TensorRef":
        """Re-index along the row axis by building ref VIEWS (no data motion)."""
        idx = [int(i) for i in (indices.tolist() if hasattr(indices, "tolist") else indices)]
        return self._gather(idx)

    def _offsets(self) -> List[int]:
        offsets = [0]
        for s in self.spans:
            offsets.append(offsets[-1] + (s.stop - s.start))
        return offsets

    def select_ranges(self, ranges: List[Tuple[int, int]]) -> "TensorRef":
        """Re-index by global ``[start, stop)`` row ranges — the single selection engine.

        Splits each range at span boundaries into views: a boundary-aligned piece
        reuses its span untouched (zero-cost); a partial piece wraps in a new
        :class:`TensorSpan` (nested spans flatten in the ctor). No sort/dedup, so
        out-of-order, overlapping, and empty range lists behave as given.
        """
        span_offsets = self._offsets()  # cumulative row boundaries; span_offsets[-1] == total rows
        selected: List[TensorSpan] = []
        for range_start, range_stop in ranges:
            range_start, range_stop = int(range_start), int(range_stop)
            if not (0 <= range_start <= range_stop <= span_offsets[-1]):
                raise IndexError(f"row range [{range_start}, {range_stop}) out of bounds for size {span_offsets[-1]}")
            cursor = range_start
            span_idx = 0
            while cursor < range_stop:
                while span_offsets[span_idx + 1] <= cursor:
                    span_idx += 1
                take = min(range_stop, span_offsets[span_idx + 1]) - cursor
                src_span = self.spans[span_idx]
                local_start = cursor - span_offsets[span_idx]
                local_stop = local_start + take
                selected.append(
                    src_span
                    if (local_start == 0 and local_stop == src_span.nrows)
                    else TensorSpan(src_span, local_start, local_stop)
                )
                cursor += take
        return self._from_spans(selected)

    def _from_spans(self, spans: List[TensorSpan]) -> "TensorRef":
        """Build a ref over the given spans, deriving shape from the row total.

        Deliberately does NOT carry ``grad`` / ``retain_grad_flag`` — matches the
        existing constructor default for a freshly selected/sliced view.
        """
        total_rows = sum(span.nrows for span in spans)
        return TensorRef(
            spans=spans,
            shape=(total_rows, *self.shape[1:]) if self.shape else None,
            dtype=self.dtype,
            device=self.device,
        )

    def _gather(self, idx: List[int]) -> "TensorRef":
        """Arbitrary re-index (gather/permute): coalesce consecutive runs into ranges."""
        ranges: List[Tuple[int, int]] = []
        run_start = 0
        while run_start < len(idx):
            run_end = run_start + 1
            while run_end < len(idx) and idx[run_end] == idx[run_end - 1] + 1:
                run_end += 1
            ranges.append((int(idx[run_start]), int(idx[run_end - 1]) + 1))
            run_start = run_end
        return self.select_ranges(ranges)

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

    def slice(self, start, end) -> "TensorRef":
        # CONCAT path: Batch.slice → _slice_value → value.slice(start, end).
        return self.select_ranges([(int(start), int(end))])

    def __getitem__(self, key) -> "TensorRef":
        # PACKED path: _slice_packed_data does ``value[cu[start]:cu[end]]``.
        if isinstance(key, slice):
            if key.step not in (None, 1):
                raise NotImplementedError("TensorRef supports only contiguous (step=1) slicing")
            start = 0 if key.start is None else int(key.start)
            stop = self.batch_size if key.stop is None else int(key.stop)
            return self.select_ranges([(start, stop)])
        raise NotImplementedError(f"TensorRef indexing supports slices only, got {type(key).__name__}")

    def transform(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "TensorRef":
        from unirl.distributed.tensor.transport import TensorTransportRuntime

        backend = TensorTransportRuntime.current()
        if backend is None:
            raise RuntimeError("No TensorTransport backend installed")
        return backend.transform(self, fn)

    def reshape(self, *shape: int) -> "TensorRef":
        return self.transform(lambda t: t.reshape(*shape))

    def permute(self, *dims: int) -> "TensorRef":
        return self.transform(lambda t: t.permute(*dims))

    def materialize(self, backend: "Optional[TensorTransport]" = None) -> torch.Tensor:
        """Fetch this meta into a real tensor (driver- or worker-side).

        With a backend: one ``get`` over the spans (backends resolve a
        :class:`TensorSpan` by resolving its handle and slicing). Without:
        per-span ``local()`` round-trips assembled via :func:`cat_rows` (its
        ragged right-pad contract applies to 2D+ per-shard-padded fields).
        """
        if backend is None:
            from unirl.distributed.tensor.transport import TensorTransportRuntime

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
# Driver-side hydration
# ---------------------------------------------------------------------------


def hydrate(value: Any) -> Any:
    """Driver-side hydrate of a ``TensorRef`` proxy back to a real ``torch.Tensor``.

    ``Worker.call`` packs every ``torch.Tensor`` leaf of a worker's return value
    into the store (via ``transport.put_batch``), so fields like ``track.rewards``
    arrive at the driver as ``TensorRef`` proxies even though downstream driver-side
    code (advantage computation) does arithmetic on them as if they were tensors.
    This fetches the underlying tensor(s) via each span's bound worker and cats
    them. Returns the value unchanged when it is already a real tensor (or any
    non-ref), and ``None`` for an empty-span ref.

    Distinct from :meth:`TensorTransport.hydrate`, which resolves refs through an
    explicit backend; this is the backend-less, per-ref path used off the driver.
    """
    if not isinstance(value, TensorRef):
        return value
    if not value.spans:
        return None
    # materialize(None) = per-span local() fetch + cat_rows (each span slices its
    # handle; ragged 2D parts follow the documented right-pad contract).
    return value.materialize(backend=None)


# ---------------------------------------------------------------------------
# Type-based tree walker
# ---------------------------------------------------------------------------


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


__all__ = ["TensorHandle", "TensorSpan", "TensorRef", "cat_rows", "map_tree"]
