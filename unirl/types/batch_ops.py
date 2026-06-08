"""Unified batch-shaped payload helpers for rollout and training types.

Provides a single set of functions that handle copy / clone / move / slice /
pad / concat / reindex on mixed-type batch data (tensors, lists, tuples,
nested dicts, or duck-typed objects with a ``.slice()`` method).

The ``recursive`` and ``deep_clone`` parameters let callers opt in to
dict-recursion and deep-copy semantics where needed (nested dicts of
tensors).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

if TYPE_CHECKING:
    from torch import device as TorchDevice


# ---------------------------------------------------------------------------
# copy / clone / move  (leaf utilities)
# ---------------------------------------------------------------------------


def copy_mapping(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Shallow-copy a mapping, turning list/tuple values into owned lists."""
    result: Dict[str, Any] = {}
    if not isinstance(mapping, Mapping):
        return result
    for key, value in mapping.items():
        if isinstance(value, (list, tuple)):
            result[str(key)] = list(value)
        else:
            result[str(key)] = value
    return result


def batch_clone(value: Any) -> Any:
    """Deep-clone a payload value (tensor ``.clone()``, containers recursed)."""
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {str(k): batch_clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [batch_clone(v) for v in value]
    if isinstance(value, tuple):
        return tuple(batch_clone(v) for v in value)
    if isinstance(value, set):
        return {batch_clone(v) for v in value}
    return value


def batch_move(value: Any, device: Union[str, "TorchDevice"]) -> Any:
    """Recursively move all tensors inside *value* to *device*."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {str(k): batch_move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [batch_move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(batch_move(v, device) for v in value)
    if isinstance(value, set):
        return {batch_move(v, device) for v in value}
    return value


# ---------------------------------------------------------------------------
# batch_slice
# ---------------------------------------------------------------------------


def batch_slice(
    value: Any,
    *,
    batch_size: int,
    start: int,
    end: int,
    recursive: bool = False,
    deep_clone: bool = False,
) -> Any:
    """Slice *value* along the batch dimension ``[start:end]``.

    Parameters
    ----------
    recursive : bool
        When ``True``, recurse into ``dict`` values before applying the
        batch-size heuristic (needed for nested extras).
    deep_clone : bool
        When ``True``, non-batched / fallback values are deep-cloned
        instead of passed through by reference.
    """
    if value is None:
        return None
    if recursive and isinstance(value, dict):
        return {
            str(k): batch_slice(
                v,
                batch_size=batch_size,
                start=start,
                end=end,
                recursive=True,
                deep_clone=deep_clone,
            )
            for k, v in value.items()
        }
    if isinstance(value, list) and len(value) == batch_size:
        sliced = value[start:end]
        return [batch_clone(v) for v in sliced] if deep_clone else sliced
    if isinstance(value, tuple) and len(value) == batch_size:
        sliced = value[start:end]
        return tuple(batch_clone(v) for v in sliced) if deep_clone else sliced
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        return value[start:end].clone()
    if hasattr(value, "slice") and callable(getattr(value, "slice")):
        return value.slice(start, end)
    return batch_clone(value) if deep_clone else value


# ---------------------------------------------------------------------------
# batch_pad
# ---------------------------------------------------------------------------


def batch_pad(
    value: Any,
    *,
    batch_size: int,
    target_size: int,
) -> Any:
    """Pad *value* from *batch_size* to *target_size* by repeating the last element."""
    if value is None or target_size <= batch_size:
        return value

    pad_count = target_size - batch_size
    if isinstance(value, list) and len(value) == batch_size:
        return value + [value[-1]] * pad_count
    if isinstance(value, tuple) and len(value) == batch_size:
        return value + (value[-1],) * pad_count
    if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == batch_size:
        pad = value[-1:].repeat(pad_count, *([1] * (value.dim() - 1)))
        return torch.cat([value, pad], dim=0)
    return value


# ---------------------------------------------------------------------------
# batch_concat
# ---------------------------------------------------------------------------


def batch_concat(
    values: List[Any],
    *,
    batch_sizes: List[int],
    deep_clone: bool = False,
    strict: bool = True,
) -> Any:
    """Concatenate *values* along the batch dimension.

    Parameters
    ----------
    deep_clone : bool
        When ``True``, list/tuple elements and fallback values are
        deep-cloned during concatenation.
    strict : bool
        When ``True`` (default), raise ``ValueError`` if non-batched
        values mismatch.  When ``False``, fall back to wrapping in a list.
    """
    non_none = [v for v in values if v is not None]
    if not non_none:
        return None

    if all(isinstance(v, Mapping) for v in non_none):
        keys = sorted({str(k) for v in non_none for k in v.keys()})
        return {
            key: batch_concat(
                [dict(v).get(key) if isinstance(v, Mapping) else None for v in values],
                batch_sizes=batch_sizes,
                deep_clone=deep_clone,
                strict=strict,
            )
            for key in keys
        }

    if all(isinstance(v, list) and len(v) == bs for v, bs in zip(values, batch_sizes) if v is not None):
        merged: List[Any] = []
        for v in values:
            if v is not None:
                if deep_clone:
                    merged.extend(batch_clone(item) for item in v)
                else:
                    merged.extend(list(v))
        return merged

    if all(isinstance(v, tuple) and len(v) == bs for v, bs in zip(values, batch_sizes) if v is not None):
        merged_t: List[Any] = []
        for v in values:
            if v is not None:
                if deep_clone:
                    merged_t.extend(batch_clone(item) for item in v)
                else:
                    merged_t.extend(list(v))
        return tuple(merged_t)

    if all(
        isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == bs
        for v, bs in zip(values, batch_sizes)
        if v is not None
    ):
        return torch.cat([v for v in values if v is not None], dim=0)

    first = non_none[0]
    if torch.is_tensor(first):
        if all(torch.is_tensor(v) and torch.equal(v, first) for v in non_none[1:]):
            return batch_clone(first) if deep_clone else first
    elif all(v == first for v in non_none[1:]):
        return batch_clone(first) if deep_clone else first

    if strict:
        raise ValueError(
            "Cannot concatenate values with mismatched non-batched content: "
            f"types={[type(v).__name__ if v is not None else None for v in values]}"
        )
    return [batch_clone(v) for v in values]


# ---------------------------------------------------------------------------
# batch_reindex
# ---------------------------------------------------------------------------


def batch_reindex(
    value: Any,
    *,
    indices: torch.Tensor,
    batch_size: int,
    recursive: bool = False,
    deep_clone: bool = False,
) -> Any:
    """Reindex *value* along the batch dimension using a permutation tensor.

    Parameters
    ----------
    recursive / deep_clone :
        Same semantics as :func:`batch_slice`.
    """
    if value is None:
        return None
    if recursive and isinstance(value, dict):
        return {
            str(k): batch_reindex(
                v,
                indices=indices,
                batch_size=batch_size,
                recursive=True,
                deep_clone=deep_clone,
            )
            for k, v in value.items()
        }
    index_list = indices.tolist()
    if isinstance(value, list) and len(value) == batch_size:
        return [batch_clone(value[i]) for i in index_list] if deep_clone else [value[i] for i in index_list]
    if isinstance(value, tuple) and len(value) == batch_size:
        return tuple(batch_clone(value[i]) for i in index_list) if deep_clone else tuple(value[i] for i in index_list)
    if isinstance(value, torch.Tensor) and value.dim() > 0 and int(value.shape[0]) == batch_size:
        return value[indices.to(value.device)]
    return batch_clone(value) if deep_clone else value


# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    "batch_clone",
    "batch_concat",
    "batch_move",
    "batch_pad",
    "batch_reindex",
    "batch_slice",
    "copy_mapping",
]
