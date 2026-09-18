"""Field-metadata-driven batch container with automatic concat/select/slice."""

from __future__ import annotations

import copy
import inspect as _inspect
import uuid as _uuid
from dataclasses import field as _dc_field
from dataclasses import fields as dc_fields
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import torch

T = TypeVar("T", bound="Batch")

_FIELD_KIND_KEY = "kind"


class FieldKind(Enum):
    CONCAT = auto()
    SHARED = auto()
    MAX = auto()
    MIN = auto()
    SUM = auto()
    MEAN = auto()
    PACKED = auto()


_REDUCTION_KINDS = frozenset({FieldKind.MAX, FieldKind.MIN, FieldKind.SUM, FieldKind.MEAN})

_DC_FIELD_PARAMS = frozenset(_inspect.signature(_dc_field).parameters)


def field(**kwargs: Any) -> Any:
    """Generic field constructor with open metadata."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    dc_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _DC_FIELD_PARAMS:
            dc_kwargs[k] = v
        else:
            metadata[k] = v
    return _dc_field(metadata=metadata, **dc_kwargs)


def concat_field(**kwargs: Any) -> Any:
    """Declare a per-sample field (concatenated along the batch dimension)."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = FieldKind.CONCAT
    return _dc_field(metadata=metadata, **kwargs)


def packed_field(**kwargs: Any) -> Any:
    """Declare a per-sample variable-length field packed along dim 0."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = FieldKind.PACKED
    return _dc_field(metadata=metadata, **kwargs)


def shared_field(**kwargs: Any) -> Any:
    """Declare a shared field (identical for every sample in the batch)."""
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = FieldKind.SHARED
    return _dc_field(metadata=metadata, **kwargs)


def _reduction_field(kind: FieldKind, **kwargs: Any) -> Any:
    metadata = dict(kwargs.pop("metadata", None) or {})
    metadata[_FIELD_KIND_KEY] = kind
    return _dc_field(metadata=metadata, **kwargs)


def max_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise max on concat."""
    return _reduction_field(FieldKind.MAX, **kwargs)


def min_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise min on concat."""
    return _reduction_field(FieldKind.MIN, **kwargs)


def sum_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise sum on concat."""
    return _reduction_field(FieldKind.SUM, **kwargs)


def mean_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise mean on concat."""
    return _reduction_field(FieldKind.MEAN, **kwargs)


def _field_kind(f: Any) -> FieldKind:
    return f.metadata.get(_FIELD_KIND_KEY, FieldKind.SHARED)


def _infer_batch_size(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor) and value.dim() > 0:
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, Batch):
        return value.batch_size
    if isinstance(value, dict):
        for v in value.values():
            bs = _infer_batch_size(v)
            if bs is not None:
                return bs
    return None


def _reduce_value(values: List[Any], kind: FieldKind) -> Any:
    """Reduce a per-instance value across N instances by ``kind``."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None

    if all(isinstance(v, torch.Tensor) for v in non_none):
        stacked = torch.stack(non_none, dim=0)
        if kind is FieldKind.MAX:
            return stacked.amax(dim=0)
        if kind is FieldKind.MIN:
            return stacked.amin(dim=0)
        if kind is FieldKind.SUM:
            return stacked.sum(dim=0)
        if kind is FieldKind.MEAN:
            return stacked.mean(dim=0)

    if kind is FieldKind.MAX:
        return max(non_none)
    if kind is FieldKind.MIN:
        return min(non_none)
    if kind is FieldKind.SUM:
        return sum(non_none)
    if kind is FieldKind.MEAN:
        total = sum(non_none)
        return total / len(non_none)

    raise ValueError(f"Unsupported reduction kind: {kind}")


def _concat_value(values: List[Any], batch_sizes: List[int]) -> Any:
    """Concatenate per-sample values along the batch axis."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None

    if all(isinstance(v, torch.Tensor) for v in non_none):
        is_batched = [v.dim() > 0 and int(v.shape[0]) == bs for v, bs in zip(values, batch_sizes) if v is not None]
        if all(is_batched):
            return torch.cat(non_none, dim=0)
        if not any(is_batched):
            return non_none[0]
        raise ValueError("Mixed batched / non-batched tensors in concat field")

    if all(isinstance(v, list) for v in non_none):
        is_batched = [len(v) == bs for v, bs in zip(values, batch_sizes) if v is not None]
        if all(is_batched):
            merged: List[Any] = []
            for v in non_none:
                merged.extend(v)
            return merged

    if all(isinstance(v, tuple) for v in non_none):
        is_batched = [len(v) == bs for v, bs in zip(values, batch_sizes) if v is not None]
        if all(is_batched):
            merged_t: List[Any] = []
            for v in non_none:
                merged_t.extend(v)
            return tuple(merged_t)

    if all(isinstance(v, dict) for v in non_none):
        keys = sorted({k for v in non_none for k in v})
        return {
            k: _concat_value(
                [v.get(k) if isinstance(v, dict) else None for v in values],
                batch_sizes=batch_sizes,
            )
            for k in keys
        }

    if all(isinstance(v, Batch) for v in non_none):
        return type(non_none[0]).concat(non_none)

    first = non_none[0]
    if torch.is_tensor(first):
        if all(
            torch.is_tensor(v) and v.shape == first.shape and torch.equal(v.to(first.device), first)
            for v in non_none[1:]
        ):
            return first
    elif all(v == first for v in non_none[1:]):
        return copy.deepcopy(first)

    raise ValueError(f"Cannot concat values: types={[type(v).__name__ for v in values]}, batch_sizes={batch_sizes}")


def _to_index_list(indices: Union[torch.Tensor, Sequence[int]]) -> List[int]:
    if isinstance(indices, torch.Tensor):
        return indices.tolist()
    return list(indices)


def _to_index_tensor(indices: Union[torch.Tensor, Sequence[int]]) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        return indices
    return torch.tensor(indices, dtype=torch.long)


def _select_value(
    value: Any,
    indices: Union[torch.Tensor, Sequence[int]],
    batch_size: int,
) -> Any:
    """Re-index a per-sample value by an index tensor or list of ints."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            idx = _to_index_tensor(indices)
            return value.index_select(0, idx.to(value.device))
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return [value[i] for i in _to_index_list(indices)]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[i] for i in _to_index_list(indices))
    if isinstance(value, dict):
        return {k: _select_value(v, indices, batch_size) for k, v in value.items()}
    if isinstance(value, Batch):
        return value.select(indices)
    return value


def _slice_value(value: Any, start: int, end: int, batch_size: int) -> Any:
    """Slice a per-sample value along the batch dimension."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            return value[start:end].clone()
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return list(value[start:end])
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(value[start:end])
    if isinstance(value, dict):
        return {k: _slice_value(v, start, end, batch_size) for k, v in value.items()}
    if isinstance(value, Batch):
        return value.slice(start, end)
    return value


def _repeat_interleave_value(value: Any, n: int, batch_size: int) -> Any:
    """Replicate a per-sample value ``n`` times along the batch axis (group-by-parent)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() > 0 and int(value.shape[0]) == batch_size:
            return value.repeat_interleave(n, dim=0)
        return value
    if isinstance(value, list) and len(value) == batch_size:
        return [v for v in value for _ in range(n)]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(v for v in value for _ in range(n))
    if isinstance(value, dict):
        return {k: _repeat_interleave_value(v, n, batch_size) for k, v in value.items()}
    if hasattr(value, "select_ranges") and getattr(value, "batch_size", None) == batch_size:
        ranges = [(index, index + 1) for index in range(batch_size) for _ in range(n)]
        return value.select_ranges(ranges)
    if isinstance(value, Batch):
        return value.repeat_interleave(n)
    return value


def _move_value(value: Any, device: Union[str, torch.device]) -> Any:
    """Move tensors in a value tree to *device*."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move_value(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_move_value(v, device) for v in value]
        return type(value)(moved)
    if isinstance(value, Batch):
        return value.to_device(device)
    return value


def _clone_value(value: Any) -> Any:
    """Deep-clone a value (tensors are ``.clone()``'d, dicts/lists recursed)."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        cloned = [_clone_value(v) for v in value]
        return type(value)(cloned)
    if isinstance(value, Batch):
        return value.clone()
    return copy.deepcopy(value)


def _concat_cu_seqlens(cus: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Merge per-shard cu_seqlens with offset shift."""
    non_none = [c for c in cus if c is not None]
    if not non_none:
        return None
    parts: List[torch.Tensor] = [non_none[0]]
    running = int(non_none[0][-1].item())
    for cu in non_none[1:]:
        parts.append(cu[1:] + running)
        running += int(cu[-1].item())
    return torch.cat(parts, dim=0)


def _slice_cu_seqlens(cu: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Slice cu_seqlens for samples ``[start, end)`` and re-zero."""
    sliced = cu[start : end + 1]
    if sliced.numel() == 0:
        return torch.zeros(1, dtype=cu.dtype, device=cu.device)
    return (sliced - sliced[0]).clone()


def _select_cu_seqlens(cu: torch.Tensor, indices: List[int]) -> torch.Tensor:
    """Rebuild cu_seqlens from selected sample sizes."""
    sizes = [int(cu[i + 1].item() - cu[i].item()) for i in indices]
    cu_list = [0]
    for s in sizes:
        cu_list.append(cu_list[-1] + s)
    return torch.tensor(cu_list, dtype=cu.dtype, device=cu.device)


def _concat_packed_data(values: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
    """Concat packed-data tensors along dim 0 (no shape-vs-batch_size check)."""
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None
    if all(isinstance(v, Batch) for v in non_none):
        return type(non_none[0]).concat(non_none)
    return torch.cat(non_none, dim=0)


def _slice_packed_data(
    value: Optional[torch.Tensor],
    start: int,
    end: int,
    cu: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Slice ``data[cu[start]:cu[end]]``. Requires cu_seqlens to be set."""
    if value is None:
        return None
    if cu is None:
        raise ValueError(
            "packed_field slice requires _packed_cu_seqlens to be populated; "
            "construct via the regular dataclass __init__ with per-sample lists."
        )
    a, b = int(cu[start].item()), int(cu[end].item())
    return value[a:b].clone()


def _select_packed_data(
    value: Optional[torch.Tensor],
    indices: List[int],
    cu: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Gather chunks ``data[cu[i]:cu[i+1]]`` per index, concat along dim 0."""
    if value is None:
        return None
    if cu is None:
        raise ValueError(
            "packed_field select requires _packed_cu_seqlens to be populated; "
            "construct via the regular dataclass __init__ with per-sample lists."
        )
    if not indices:
        if hasattr(value, "select_ranges"):
            return value.select_ranges([])
        return value[:0].clone()
    if hasattr(value, "select_ranges"):
        return value.select_ranges([(int(cu[i].item()), int(cu[i + 1].item())) for i in indices])
    chunks = [value[int(cu[i].item()) : int(cu[i + 1].item())] for i in indices]
    return torch.cat(chunks, dim=0)


def _repeat_interleave_packed_data(
    value: Optional[torch.Tensor],
    n: int,
    cu: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Repeat each per-sample chunk ``n`` times along dim 0 (group-by-parent order)."""
    if value is None:
        return None
    if cu is None:
        raise ValueError(
            "packed_field repeat_interleave requires _packed_cu_seqlens to be populated; "
            "construct via the regular dataclass __init__ with per-sample lists."
        )
    if n <= 0:
        if hasattr(value, "select_ranges"):
            return value.select_ranges([])
        return value[:0].clone()
    if n == 1:
        return value.clone()
    if hasattr(value, "select_ranges"):
        ranges = [(int(cu[i].item()), int(cu[i + 1].item())) for i in range(int(cu.numel()) - 1) for _ in range(n)]
        return value.select_ranges(ranges)
    chunks: List[torch.Tensor] = []
    for i in range(int(cu.numel()) - 1):
        chunk = value[int(cu[i].item()) : int(cu[i + 1].item())]
        for _ in range(n):
            chunks.append(chunk)
    return torch.cat(chunks, dim=0) if chunks else value[:0].clone()


def _repeat_interleave_cu_seqlens(cu: torch.Tensor, n: int) -> torch.Tensor:
    """Rebuild cu_seqlens after repeating each chunk ``n`` times (group-by-parent order)."""
    sizes = (cu[1:] - cu[:-1]).tolist()
    new_sizes = [s for s in sizes for _ in range(n)]
    cu_list = [0]
    for s in new_sizes:
        cu_list.append(cu_list[-1] + s)
    return torch.tensor(cu_list, dtype=cu.dtype, device=cu.device)


class Batch:
    """Mixin / base for ``@dataclass`` containers with concat/shared fields."""

    _packed_cu_seqlens: Optional[torch.Tensor] = None

    @property
    def _eid(self) -> str:
        """Lazily-assigned UUID for list-element wire-key disambiguation."""
        eid = getattr(self, "__eid__", None)
        if eid is None:
            eid = _uuid.uuid4().hex[:12]
            object.__setattr__(self, "__eid__", eid)
        return eid

    @property
    def cu_seqlens(self) -> Optional[torch.Tensor]:
        """Framework-managed cumulative offsets ``[N+1]`` for packed fields."""
        return self._packed_cu_seqlens

    @property
    def lengths(self) -> Optional[torch.Tensor]:
        """Per-sample lengths derived from :attr:`cu_seqlens`."""
        cu = self._packed_cu_seqlens
        if cu is None:
            return None
        return cu[1:] - cu[:-1]

    @property
    def batch_size(self) -> int:
        for f in dc_fields(self):  # type: ignore[arg-type]
            if _field_kind(f) is not FieldKind.CONCAT:
                continue
            bs = _infer_batch_size(getattr(self, f.name))
            if bs is not None:
                return bs
        cu = self._packed_cu_seqlens
        if cu is not None and cu.numel() > 0:
            return int(cu.shape[0]) - 1
        return 0

    @classmethod
    def pack(cls: Type[T], **kwargs: Any) -> T:
        """Construct ``cls``, packing per-sample tensor lists for ``packed_field``s."""
        sizes: Optional[List[int]] = None
        packed_kwargs: Dict[str, Any] = dict(kwargs)
        for f in dc_fields(cls):  # type: ignore[arg-type]
            if _field_kind(f) is not FieldKind.PACKED:
                continue
            if f.name not in packed_kwargs:
                continue
            value = packed_kwargs[f.name]
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{cls.__name__}.pack: packed_field {f.name!r} expects a "
                    f"Sequence[Tensor] of per-sample tensors, not an "
                    f"already-packed Tensor. Use the regular {cls.__name__}(...) "
                    f"constructor for already-packed inputs."
                )
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"{cls.__name__}.pack: packed_field {f.name!r} expects a "
                    f"Sequence[Tensor]; got {type(value).__name__}"
                )
            if not all(isinstance(v, torch.Tensor) for v in value):
                raise TypeError(
                    f"{cls.__name__}.pack: packed_field {f.name!r} elements "
                    f"must all be torch.Tensor; got types "
                    f"{[type(v).__name__ for v in value]}"
                )
            local_sizes = [int(v.shape[0]) if v.dim() > 0 else 1 for v in value]
            if sizes is None:
                sizes = local_sizes
            elif local_sizes != sizes:
                raise ValueError(
                    f"{cls.__name__}.pack: packed_field {f.name!r} per-sample "
                    f"sizes {local_sizes} don't match earlier packed-field "
                    f"sizes {sizes}"
                )
            packed_kwargs[f.name] = torch.cat(value, dim=0) if value else value

        instance = cls(**packed_kwargs)
        if sizes is not None:
            cu_list = [0]
            for s in sizes:
                cu_list.append(cu_list[-1] + s)
            cu = torch.tensor(cu_list, dtype=torch.long)
            object.__setattr__(instance, "_packed_cu_seqlens", cu)
        return instance

    @classmethod
    def concat(cls: Type[T], items: Sequence[T]) -> T:
        """Concatenate multiple instances along the batch dimension."""
        if not items:
            raise ValueError(f"Cannot concat empty sequence of {cls.__name__}")
        if len(items) == 1:
            return items[0]

        has_packed = any(
            _field_kind(f) is FieldKind.PACKED
            for f in dc_fields(items[0])  # type: ignore[arg-type]
        )
        merged_cu = _concat_cu_seqlens([item._packed_cu_seqlens for item in items]) if has_packed else None

        batch_sizes = [item.batch_size for item in items]
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(items[0]):  # type: ignore[arg-type]
            values = [getattr(item, f.name) for item in items]
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _concat_value(values, batch_sizes)
            elif kind is FieldKind.PACKED:
                kwargs[f.name] = _concat_packed_data(values)
            elif kind in _REDUCTION_KINDS:
                kwargs[f.name] = _reduce_value(values, kind)
            else:
                kwargs[f.name] = values[0]

        instance = cls(**kwargs)
        if has_packed:
            object.__setattr__(instance, "_packed_cu_seqlens", merged_cu)
        return instance

    def concat_with(self: T, *others: T) -> T:
        """Concatenate ``self`` with one or more other instances."""
        return type(self).concat([self, *others])

    def chunk(self: T, n: int) -> List[T]:
        """Split into ``n`` equal contiguous shards along the batch dimension."""
        if n <= 0:
            raise ValueError(f"chunk: n must be positive, got {n}")
        bs = self.batch_size
        if bs % n != 0:
            raise ValueError(f"chunk: batch_size={bs} not divisible by n={n}")
        size = bs // n
        return [self.slice(i * size, (i + 1) * size) for i in range(n)]

    def select(self: T, indices: torch.Tensor) -> T:
        """Re-index along the batch dimension (gather / shuffle / subsample)."""
        bs = self.batch_size
        cu = self._packed_cu_seqlens
        idx_list = _to_index_list(indices)
        has_packed = any(
            _field_kind(f) is FieldKind.PACKED
            for f in dc_fields(self)  # type: ignore[arg-type]
        )
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _select_value(val, indices, bs)
            elif kind is FieldKind.PACKED:
                kwargs[f.name] = _select_packed_data(val, idx_list, cu)
            else:
                kwargs[f.name] = val
        instance = type(self)(**kwargs)
        if has_packed:
            new_cu = _select_cu_seqlens(cu, idx_list) if cu is not None else None
            object.__setattr__(instance, "_packed_cu_seqlens", new_cu)
        return instance

    def slice(self: T, start: int, end: int) -> T:
        """Slice ``[start, end)`` along the batch dimension."""
        bs = self.batch_size
        cu = self._packed_cu_seqlens
        has_packed = any(
            _field_kind(f) is FieldKind.PACKED
            for f in dc_fields(self)  # type: ignore[arg-type]
        )
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _slice_value(val, start, end, bs)
            elif kind is FieldKind.PACKED:
                kwargs[f.name] = _slice_packed_data(val, start, end, cu)
            else:
                kwargs[f.name] = val
        instance = type(self)(**kwargs)
        if has_packed:
            new_cu = _slice_cu_seqlens(cu, start, end) if cu is not None else None
            object.__setattr__(instance, "_packed_cu_seqlens", new_cu)
        return instance

    def repeat_interleave(self: T, n: int) -> T:
        """Replicate each sample ``n`` times along the batch dimension (group-by-parent)."""
        if n < 0:
            raise ValueError(f"repeat_interleave: n must be non-negative, got {n}")
        if n == 1:
            return self.clone()
        bs = self.batch_size
        cu = self._packed_cu_seqlens
        has_packed = any(
            _field_kind(f) is FieldKind.PACKED
            for f in dc_fields(self)  # type: ignore[arg-type]
        )
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _repeat_interleave_value(val, n, bs)
            elif kind is FieldKind.PACKED:
                kwargs[f.name] = _repeat_interleave_packed_data(val, n, cu)
            else:
                kwargs[f.name] = val
        instance = type(self)(**kwargs)
        if has_packed:
            new_cu = _repeat_interleave_cu_seqlens(cu, n) if cu is not None else None
            object.__setattr__(instance, "_packed_cu_seqlens", new_cu)
        return instance

    def to_device(self: T, device: Union[str, torch.device]) -> T:
        """Move all tensor-like values to *device*."""
        has_packed = any(
            _field_kind(f) is FieldKind.PACKED
            for f in dc_fields(self)  # type: ignore[arg-type]
        )
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            kwargs[f.name] = _move_value(getattr(self, f.name), device)
        instance = type(self)(**kwargs)
        if has_packed:
            cu = self._packed_cu_seqlens
            object.__setattr__(
                instance,
                "_packed_cu_seqlens",
                cu.to(device) if cu is not None else None,
            )
        return instance

    def map(self: T, fn: Callable[[Any], Any]) -> T:
        """Rebuild a same-type instance by applying ``fn`` to each field value."""
        instance = type(self)(**{f.name: fn(getattr(self, f.name)) for f in dc_fields(self)})
        if self._packed_cu_seqlens is not None:
            object.__setattr__(instance, "_packed_cu_seqlens", self._packed_cu_seqlens)
        return instance

    def _rebuild(self: T, field_values: Dict[str, Any]) -> T:
        """Reconstruct from precomputed field values, preserving hidden state."""
        instance = type(self)(**field_values)
        if self._packed_cu_seqlens is not None:
            object.__setattr__(instance, "_packed_cu_seqlens", self._packed_cu_seqlens)
        return instance

    def clone(self: T) -> T:
        """Deep-clone the container (tensors are ``.clone()``'d)."""
        has_packed = any(
            _field_kind(f) is FieldKind.PACKED
            for f in dc_fields(self)  # type: ignore[arg-type]
        )
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            kwargs[f.name] = _clone_value(getattr(self, f.name))
        instance = type(self)(**kwargs)
        if has_packed:
            cu = self._packed_cu_seqlens
            object.__setattr__(
                instance,
                "_packed_cu_seqlens",
                cu.clone() if cu is not None else None,
            )
        return instance


__all__ = [
    "Batch",
    "FieldKind",
    "concat_field",
    "field",
    "max_field",
    "mean_field",
    "min_field",
    "packed_field",
    "shared_field",
    "sum_field",
]
