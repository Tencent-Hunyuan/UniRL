"""Pytree-aware batch-axis ops over a same-structured tree."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle
from unirl.distributed.tensor.batch import Batch
from unirl.distributed.tensor.ref import TensorRef
from unirl.distributed.utils import Broadcast


def _value_batch_size(value) -> Optional[int]:
    """First batch-axis size found in ``value``, or ``None``."""
    if isinstance(value, Broadcast):
        return None
    if isinstance(value, (torch.Tensor, np.ndarray, GPUTensorHandle, TensorRef)):
        shape = value.shape
        return int(shape[0]) if shape else None
    if isinstance(value, list):
        return len(value)
    if isinstance(value, Batch):
        return value.batch_size or None
    if isinstance(value, tuple):
        for v in value:
            bs = _value_batch_size(v)
            if bs is not None:
                return bs
    if isinstance(value, dict):
        for v in value.values():
            bs = _value_batch_size(v)
            if bs is not None:
                return bs
    return None


def infer_batch_size(args: tuple, kwargs: dict) -> Optional[int]:
    """Canonical batch size for DP chunking, inferred from a call payload."""
    for v in args:
        bs = _value_batch_size(v)
        if bs is not None:
            return bs
    for v in kwargs.values():
        bs = _value_batch_size(v)
        if bs is not None:
            return bs
    return None


def pytree_chunk(value, dp_size: int, batch_size: int) -> list:
    """Recursively split a value into ``dp_size`` shards along axis 0."""
    if isinstance(value, Broadcast):
        return [value.value] * dp_size

    elif isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return [value] * dp_size
        if value.shape[0] != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        chunk_size = batch_size // dp_size
        return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(dp_size)]

    elif isinstance(value, np.ndarray):
        if value.shape[0] != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        chunk_size = batch_size // dp_size
        return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(dp_size)]

    elif isinstance(value, list):
        if len(value) != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        chunk_size = batch_size // dp_size
        return [value[i * chunk_size : (i + 1) * chunk_size] for i in range(dp_size)]

    elif isinstance(value, dict):
        split_dict = {k: pytree_chunk(v, dp_size, batch_size) for k, v in value.items()}
        return [{k: split_dict[k][i] for k in value} for i in range(dp_size)]

    elif isinstance(value, tuple):
        split_elems = [pytree_chunk(v, dp_size, batch_size) for v in value]
        return [tuple(split_elems[j][i] for j in range(len(value))) for i in range(dp_size)]

    elif isinstance(value, Batch):
        if value.batch_size != batch_size:
            return [value] * dp_size
        return value.chunk(dp_size)

    else:
        return [value] * dp_size


def pytree_cat(results: list) -> Any:
    """Recursively merge same-structure results along axis 0."""
    if not results:
        return None

    first = results[0]

    if first is None:
        return None
    elif isinstance(first, torch.Tensor):
        return torch.cat(results, dim=0)
    elif isinstance(first, TensorRef):
        all_spans = []
        for m in results:
            all_spans.extend(m.spans)
        total = sum(s.stop - s.start for s in all_spans)
        return TensorRef(
            spans=all_spans,
            shape=(total, *first.shape[1:]) if first.shape else None,
            dtype=first.dtype,
            device=first.device,
        )
    elif isinstance(first, np.ndarray):
        return np.concatenate(results, axis=0)
    elif isinstance(first, list):
        return sum(results, [])
    elif isinstance(first, tuple):
        return tuple(pytree_cat([r[i] for r in results]) for i in range(len(first)))
    elif isinstance(first, dict):
        return {k: pytree_cat([r[k] for r in results]) for k in first}
    elif isinstance(first, Batch):
        return type(first).concat(results)
    else:
        return first


__all__ = ["infer_batch_size", "pytree_cat", "pytree_chunk"]
