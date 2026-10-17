"""Field-metadata-driven batch container with automatic concat/select/slice."""

from __future__ import annotations

import copy
import inspect as _inspect
import itertools
import reprlib
import uuid as _uuid
from dataclasses import field as _dc_field
from dataclasses import fields as dc_fields
from dataclasses import is_dataclass
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
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


class FieldKind(Enum):
    CONCAT = auto()
    SHARED = auto()
    MAX = auto()
    PACKED = auto()


_DC_FIELD_PARAMS = frozenset(_inspect.signature(_dc_field).parameters)


def field(**kwargs: Any) -> Any:
    """Generic field constructor with open metadata."""
    raw_metadata = kwargs.pop("metadata", None)
    metadata = {} if raw_metadata is None else dict(raw_metadata)
    dc_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in _DC_FIELD_PARAMS:
            dc_kwargs[k] = v
        else:
            metadata[k] = v
    return _dc_field(metadata=metadata, **dc_kwargs)


def concat_field(**kwargs: Any) -> Any:
    """Declare a per-sample field (concatenated along the batch dimension)."""
    return field(kind=FieldKind.CONCAT, **kwargs)


def packed_field(**kwargs: Any) -> Any:
    """Declare a per-sample variable-length field packed along dim 0."""
    return field(kind=FieldKind.PACKED, **kwargs)


def shared_field(**kwargs: Any) -> Any:
    """Declare a shared field (identical for every sample in the batch)."""
    return field(kind=FieldKind.SHARED, **kwargs)


def max_field(**kwargs: Any) -> Any:
    """Declare a scalar/tensor field reduced by elementwise max on concat."""
    return field(kind=FieldKind.MAX, **kwargs)


def _field_kind(f: Any) -> FieldKind:
    return f.metadata.get("kind", FieldKind.SHARED)


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


def _first_difference(a: Any, b: Any, path: str) -> Optional[str]:
    """Describe the first structural difference, or ``None``; tensor proxies compare by shape and dtype only."""
    if a is b:
        return None
    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{path}: {tuple(a.shape)} {a.dtype} != {tuple(b.shape)} {b.dtype}"
        b = b.to(a.device)
        if torch.equal(a, b):
            return None
        index = int((a != b).reshape(-1).nonzero()[0])
        return f"{path}.flat[{index}]: {a.reshape(-1)[index].item()} != {b.reshape(-1)[index].item()}"
    if isinstance(a, Batch) and a._is_tensor_proxy:
        if a.shape != b.shape or a.dtype != b.dtype:
            return f"{path}: {a.shape} {a.dtype} != {b.shape} {b.dtype}"
        return None
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return f"{path}: keys {sorted(a, key=repr)} != {sorted(b, key=repr)}"
        children = ((f"{path}[{k!r}]", a[k], b[k]) for k in a)
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        children = ((f"{path}[{i}]", x, y) for i, (x, y) in enumerate(zip(a, b)))
    elif is_dataclass(a):
        children = ((f"{path}.{f.name}", getattr(a, f.name), getattr(b, f.name)) for f in dc_fields(a))
    else:
        return None if a == b else f"{path}: {reprlib.repr(a)} != {reprlib.repr(b)}"
    for child_path, x, y in children:
        difference = _first_difference(x, y, child_path)
        if difference is not None:
            return difference
    return None


def _shared_value(values: List[Any], name: str) -> Any:
    """Return the value every item agrees on, or raise naming the first difference."""
    first = values[0]
    for i, value in enumerate(values[1:], start=1):
        difference = _first_difference(first, value, name)
        if difference is not None:
            raise ValueError(f"shared value differs between concatenated items 0 and {i}: {difference}")
    return first


def _any_present(values: List[Any], batch_sizes: List[int], name: str) -> bool:
    """Whether the field is set; only zero-row items may leave it ``None`` while others set it."""
    if all(v is None for v in values):
        return False
    missing = [i for i, (v, bs) in enumerate(zip(values, batch_sizes)) if v is None and bs > 0]
    if missing:
        raise ValueError(f"{name}: per-sample value is missing in items {missing} but set in others")
    return True


def _concat_value(values: List[Any], batch_sizes: List[int], name: str) -> Any:
    """Concatenate per-sample values along the batch axis."""
    if not _any_present(values, batch_sizes, name):
        return None
    items = [(v, bs) for v, bs in zip(values, batch_sizes) if v is not None]
    non_none = [v for v, _ in items]

    if all(isinstance(v, torch.Tensor) for v in non_none):
        is_batched = [v.dim() > 0 and v.shape[0] == bs for v, bs in items]
        if all(is_batched):
            return torch.cat(non_none, dim=0)
        if any(is_batched):
            raise ValueError(f"{name}: mixed batched / non-batched tensors")
        return non_none[0]

    if all(isinstance(v, Batch) and v._is_tensor_proxy for v in non_none):
        rows = [v.batch_size for v in non_none]
        sizes = [bs for _, bs in items]
        if rows != sizes:
            # TensorRef.concat joins spans regardless of row count, which would misalign the merged rows.
            raise ValueError(f"{name}: tensor reference rows {rows} do not match item batch sizes {sizes}")
        return type(non_none[0]).concat(non_none)

    seq_type = type(non_none[0])
    if seq_type in (list, tuple) and all(type(v) is seq_type and len(v) == bs for v, bs in items):
        return seq_type(x for v in non_none for x in v)

    if all(isinstance(v, dict) for v in non_none):
        keys = sorted({k for v in non_none for k in v})
        return {
            k: _concat_value(
                [v.get(k) if isinstance(v, dict) else None for v in values],
                batch_sizes,
                f"{name}[{k!r}]",
            )
            for k in keys
        }

    if all(isinstance(v, Batch) for v in non_none):
        return type(non_none[0]).concat(non_none)

    return copy.deepcopy(_shared_value(non_none, name))


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
            idx = indices if isinstance(indices, torch.Tensor) else torch.tensor(indices, dtype=torch.long)
            return value.index_select(0, idx.to(value.device))
        return value
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        idx_list = indices.tolist() if isinstance(indices, torch.Tensor) else indices
        return type(value)(value[i] for i in idx_list)
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
    if isinstance(value, (list, tuple)) and len(value) == batch_size:
        return type(value)(value[start:end])
    if isinstance(value, dict):
        return {k: _slice_value(v, start, end, batch_size) for k, v in value.items()}
    if isinstance(value, Batch):
        return value.slice(start, end)
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
        return type(value)(_move_value(v, device) for v in value)
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
        return type(value)(_clone_value(v) for v in value)
    if isinstance(value, Batch):
        return value.clone()
    return copy.deepcopy(value)


class Batch:
    """Mixin / base for ``@dataclass`` containers with concat/shared fields."""

    _packed_cu_seqlens: Optional[torch.Tensor] = None
    _is_tensor_proxy: ClassVar[bool] = False

    @property
    def _eid(self) -> str:
        """Lazily-assigned UUID for list-element wire-key disambiguation."""
        eid = getattr(self, "__eid__", None)
        if eid is None:
            eid = _uuid.uuid4().hex[:12]
            self.__eid__ = eid
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
            instance._packed_cu_seqlens = torch.tensor([0, *itertools.accumulate(sizes)], dtype=torch.long)
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
        batch_sizes = [item.batch_size for item in items]
        merged_cu: Optional[torch.Tensor] = None
        if has_packed:
            cus = [item._packed_cu_seqlens for item in items]
            if _any_present(cus, batch_sizes, f"{cls.__name__}.cu_seqlens"):
                non_none_cu = [c for c in cus if c is not None]
                parts: List[torch.Tensor] = [non_none_cu[0]]
                running = int(non_none_cu[0][-1].item())
                for cu in non_none_cu[1:]:
                    parts.append(cu[1:] + running)
                    running += int(cu[-1].item())
                merged_cu = torch.cat(parts, dim=0)

        kwargs: Dict[str, Any] = {}
        for f in dc_fields(items[0]):  # type: ignore[arg-type]
            values = [getattr(item, f.name) for item in items]
            kind = _field_kind(f)
            name = f"{cls.__name__}.{f.name}"
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _concat_value(values, batch_sizes, name)
            elif kind is FieldKind.PACKED:
                if not _any_present(values, batch_sizes, name):
                    kwargs[f.name] = None
                else:
                    non_none = [v for v in values if v is not None]
                    kwargs[f.name] = (
                        type(non_none[0]).concat(non_none)
                        if all(isinstance(v, Batch) for v in non_none)
                        else torch.cat(non_none, dim=0)
                    )
            elif kind is FieldKind.MAX:
                non_none = [v for v in values if v is not None]
                if not non_none:
                    kwargs[f.name] = None
                elif all(isinstance(v, torch.Tensor) for v in non_none):
                    kwargs[f.name] = torch.stack(non_none, dim=0).amax(dim=0)
                else:
                    kwargs[f.name] = max(non_none)
            else:
                kwargs[f.name] = _shared_value(values, name)

        instance = cls(**kwargs)
        if has_packed:
            instance._packed_cu_seqlens = merged_cu
        return instance

    def chunk(self: T, n: int) -> List[T]:
        """Split into ``n`` equal contiguous shards along the batch dimension."""
        if n <= 0:
            raise ValueError(f"chunk: n must be positive, got {n}")
        bs = self.batch_size
        if bs % n != 0:
            raise ValueError(f"chunk: batch_size={bs} not divisible by n={n}")
        size = bs // n
        return [self.slice(i * size, (i + 1) * size) for i in range(n)]

    def select(self: T, indices: Union[torch.Tensor, Sequence[int]]) -> T:
        """Re-index along the batch dimension (gather / shuffle / subsample)."""
        bs = self.batch_size
        cu = self._packed_cu_seqlens
        idx_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _select_value(val, indices, bs)
            elif kind is FieldKind.PACKED:
                if val is None:
                    kwargs[f.name] = None
                elif cu is None:
                    raise ValueError(
                        "packed_field select requires _packed_cu_seqlens to be populated; "
                        "construct via the regular dataclass __init__ with per-sample lists."
                    )
                else:
                    is_proxy = isinstance(val, Batch) and val._is_tensor_proxy
                    if not idx_list:
                        kwargs[f.name] = val.select_ranges([]) if is_proxy else val[:0].clone()
                    elif is_proxy:
                        kwargs[f.name] = val.select_ranges(
                            [(int(cu[i].item()), int(cu[i + 1].item())) for i in idx_list]
                        )
                    else:
                        chunks = [val[int(cu[i].item()) : int(cu[i + 1].item())] for i in idx_list]
                        kwargs[f.name] = torch.cat(chunks, dim=0)
            else:
                kwargs[f.name] = val
        instance = type(self)(**kwargs)
        if cu is not None:
            sizes = (int(cu[i + 1].item() - cu[i].item()) for i in idx_list)
            instance._packed_cu_seqlens = torch.tensor(
                [0, *itertools.accumulate(sizes)], dtype=cu.dtype, device=cu.device
            )
        return instance

    def slice(self: T, start: int, end: int) -> T:
        """Slice ``[start, end)`` along the batch dimension."""
        bs = self.batch_size
        cu = self._packed_cu_seqlens
        kwargs: Dict[str, Any] = {}
        for f in dc_fields(self):  # type: ignore[arg-type]
            val = getattr(self, f.name)
            kind = _field_kind(f)
            if kind is FieldKind.CONCAT:
                kwargs[f.name] = _slice_value(val, start, end, bs)
            elif kind is FieldKind.PACKED:
                if val is None:
                    kwargs[f.name] = None
                elif cu is None:
                    raise ValueError(
                        "packed_field slice requires _packed_cu_seqlens to be populated; "
                        "construct via the regular dataclass __init__ with per-sample lists."
                    )
                else:
                    a, b = int(cu[start].item()), int(cu[end].item())
                    kwargs[f.name] = val[a:b].clone()
            else:
                kwargs[f.name] = val
        instance = type(self)(**kwargs)
        if cu is not None:
            sliced = cu[start : end + 1]
            instance._packed_cu_seqlens = (
                torch.zeros(1, dtype=cu.dtype, device=cu.device)
                if sliced.numel() == 0
                else (sliced - sliced[0]).clone()
            )
        return instance

    def to_device(self: T, device: Union[str, torch.device]) -> T:
        """Move all tensor-like values to *device*."""
        kwargs: Dict[str, Any] = {f.name: _move_value(getattr(self, f.name), device) for f in dc_fields(self)}  # type: ignore[arg-type]
        instance = type(self)(**kwargs)
        if self._packed_cu_seqlens is not None:
            instance._packed_cu_seqlens = self._packed_cu_seqlens.to(device)
        return instance

    def rebuild(self: T, field_values: Dict[str, Any]) -> T:
        """Reconstruct from precomputed field values, preserving hidden state."""
        instance = type(self)(**field_values)
        if self._packed_cu_seqlens is not None:
            instance._packed_cu_seqlens = self._packed_cu_seqlens
        return instance

    def clone(self: T) -> T:
        """Deep-clone the container (tensors are ``.clone()``'d)."""
        kwargs: Dict[str, Any] = {f.name: _clone_value(getattr(self, f.name)) for f in dc_fields(self)}  # type: ignore[arg-type]
        instance = type(self)(**kwargs)
        if self._packed_cu_seqlens is not None:
            instance._packed_cu_seqlens = self._packed_cu_seqlens.clone()
        return instance


__all__ = [
    "Batch",
    "FieldKind",
    "concat_field",
    "field",
    "max_field",
    "packed_field",
    "shared_field",
]
