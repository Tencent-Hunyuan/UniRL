"""Pytree-aware batch-axis ops over a same-structured tree.

``pytree_chunk`` shards one same-structured tree into ``N`` per-rank trees;
``pytree_cat`` merges ``N`` same-structured trees back into one (the inverse
pair). Both recurse over the same node types (``Tensor`` / ``ndarray`` /
``list`` / ``tuple`` / ``dict`` / ``Batch`` / ``TensorMeta``) and operate
along axis 0.

``infer_batch_size`` is the companion that derives the ``batch_size`` argument
``pytree_chunk`` splits along: it walks an ``args`` / ``kwargs`` payload and
returns the first batch-axis size found (``Broadcast``-wrapped values opt out).

These are the wire-layer walkers used by DP dispatch and gradient propagation.
The per-field walkers inside ``Batch`` (``_concat_value``, ``_slice_value``, …)
are a separate, field-kind-aware layer that ``pytree_cat`` delegates to when
it encounters a ``Batch`` node.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from unirl.distributed.tensor.backend.colocate_store.handle import TensorHandle
from unirl.distributed.tensor.batch import Batch
from unirl.distributed.tensor.transport import TensorMeta
from unirl.distributed.utils import Broadcast

# ── Batch-size inference ──


def _value_batch_size(value) -> Optional[int]:
    """First batch-axis size found in ``value``, or ``None``.

    Node classification mirrors :func:`pytree_chunk`'s split contract, so the
    inferred size lines up with what actually gets chunked:

      - ``Broadcast`` → skipped (never contributes a size)
      - ``Tensor`` / ``ndarray`` / ``TensorHandle`` / ``TensorMeta`` →
        ``shape[0]`` (0-dim scalars contribute nothing)
      - ``list`` → ``len`` (per-sample batch)
      - ``Batch`` → its own ``batch_size`` (concat-field aligned)
      - ``tuple`` / ``dict`` → structural; recurse and take the first hit
      - anything else (int / float / str / None) → skipped (broadcast)

    Following ``Batch._infer_batch_size``, the *first* size found wins (not the
    largest): per-rollout metadata whose leading dim differs from the real
    batch is replicated by :func:`pytree_chunk` anyway, but a field whose
    leading dim coincides with the batch must be wrapped in :class:`Broadcast`
    to opt out of splitting.
    """
    if isinstance(value, Broadcast):
        return None
    if isinstance(value, (torch.Tensor, np.ndarray, TensorHandle, TensorMeta)):
        # ``TensorMeta.shape`` is Optional and 0-dim tensors have an empty
        # shape — both mean "no batch axis".
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
    """Canonical batch size for DP chunking, inferred from a call payload.

    Walks ``args`` then ``kwargs`` and returns the first batch-axis size found
    (see :func:`_value_batch_size` for the per-node rules, which match
    :func:`pytree_chunk`'s split contract). Returns ``None`` for a pure
    broadcast call — no batched field — in which case the dispatch layer
    replicates the whole payload to every worker instead of splitting it.
    """
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
    """Recursively split a value into ``dp_size`` shards along axis 0.

    Inverse of :func:`pytree_cat` on the equal-chunk case.

    Rules:
      - ``Broadcast(x)`` → ``[x] * dp_size``  (explicit opt-out of splitting)
      - ``Tensor`` → chunk along dim 0 (must be divisible)
      - ``ndarray`` → split along axis 0 (must be divisible)
      - ``TensorMeta`` → chunk refs across ``dp_size`` (requires divisibility)
      - ``list`` → slice into equal parts (must be divisible)
      - ``tuple`` → recurse element-wise, reassemble per-shard tuples
      - ``dict`` → recurse into values, reassemble per-shard dicts
      - ``Batch`` → split each field, reassemble per-shard ``Batch`` objects
      - other (int / float / str / None) → ``[value] * dp_size``  (broadcast)

    To prevent a value inside a tuple/dict/Batch from being split, wrap it
    in ``Broadcast(x)``.
    """
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

    elif isinstance(value, TensorMeta):
        total = value.shape[0]
        if total != batch_size:
            return [value] * dp_size
        if batch_size % dp_size != 0:
            raise ValueError(f"batch_size={batch_size} not divisible by dp_size={dp_size}")
        n_refs = len(value.refs)
        if n_refs % dp_size != 0:
            raise ValueError(f"TensorMeta has {n_refs} refs, not divisible by dp_size={dp_size}")
        refs_per_shard = n_refs // dp_size
        parts = []
        for i in range(dp_size):
            start = i * refs_per_shard
            end = start + refs_per_shard
            shard_refs = value.refs[start:end]
            shard_sizes = value.sizes[start:end]
            shard_total = sum(shard_sizes)
            parts.append(
                TensorMeta(
                    refs=shard_refs,
                    sizes=shard_sizes,
                    shape=(shard_total, *value.shape[1:]) if value.shape else None,
                    dtype=value.dtype,
                    device=value.device,
                )
            )
        return parts

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
        # Kind-aware split, mirroring pytree_cat's Batch.concat delegation on the
        # collect side: Batch.chunk slices each field by its kind (CONCAT/PACKED
        # split, SHARED/reduction passed through) and recomputes cu_seqlens for
        # packed fields. A Batch whose own batch dim differs from the dispatch dim
        # doesn't participate in the split -> replicate it.
        if value.batch_size != batch_size:
            return [value] * dp_size
        return value.chunk(dp_size)

    else:
        return [value] * dp_size


def pytree_cat(results: list) -> Any:
    """Recursively merge same-structure results along axis 0.

    Inverse of :func:`pytree_chunk` on the merge side.

    Rules:
      - ``Tensor`` → ``torch.cat`` along dim 0
      - ``TensorMeta`` → merge all refs into one ``TensorMeta``
      - ``ndarray`` → ``np.concatenate`` along axis 0
      - ``list`` → flatten (concatenate lists)
      - ``tuple`` → recurse element-wise (record-style), return tuple
      - ``dict`` → recurse per-key, return dict
      - ``Batch`` → delegate to ``type(first).concat(results)``
      - ``None`` → ``None``
      - scalar (int / float / str / ...) → take first (all should match)
    """
    if not results:
        return None

    first = results[0]

    if first is None:
        return None
    elif isinstance(first, torch.Tensor):
        return torch.cat(results, dim=0)
    elif isinstance(first, TensorMeta):
        all_refs = []
        all_sizes = []
        for m in results:
            all_refs.extend(m.refs)
            all_sizes.extend(m.sizes)
        total = sum(all_sizes)
        return TensorMeta(
            refs=all_refs,
            sizes=all_sizes,
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
