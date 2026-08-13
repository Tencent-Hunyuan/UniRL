"""TQTransport: Transfer Queue backend for TensorTransport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tensordict import NonTensorData, TensorDict

from unirl.distributed.tensor.backend.transfer_queue.runtime import (
    _DEFAULT_PARTITION_ID,
    TransferQueueRuntime,
    _run_async_in_temp_loop,
)
from unirl.distributed.tensor.ref import TensorRef, TensorSpan
from unirl.distributed.tensor.transport import TensorTransport, TensorTransportRuntime


@dataclass
class TQTensorHandle:
    """One TransferQueue tensor: its upstream row-metadata, the field name it occupied on put, and its shape."""

    meta: Any  # Upstream SampleMeta or BatchMeta.
    field: str  # the field name the tensor occupies in its put's TensorDict
    orig_shape: Optional[Tuple[int, ...]] = None  # shape to restore on fetch (None for non-tensors)

    def _gkey(self) -> tuple:
        """Originating-put identity: same key ⇒ column-unionable in one get."""
        return (tuple(sorted(self.meta.global_indexes)), tuple(self.meta.partition_ids))

    def local(self) -> torch.Tensor:
        backend = TensorTransportRuntime.current()
        if backend is None:
            raise RuntimeError("TQTensorHandle.local(): no TensorTransport installed to resolve the ref.")
        return backend._resolve_handles([self])[0]


def _store_shape(t: torch.Tensor) -> torch.Tensor:
    """Pad to >=2 dims so every per-row slice is >=1-dim."""
    if t.dim() == 0:
        return t.reshape(1, 1)
    if t.dim() == 1:
        return t.unsqueeze(1)  # (N,) -> (N, 1)
    return t


def _restore(val: Any, orig_shape: Optional[Tuple[int, ...]]) -> Any:
    """Reshape a fetched tensor back to the shape it was put with (no-op unless the backend altered it)."""
    if orig_shape is not None and isinstance(val, torch.Tensor) and val.shape != orig_shape:
        return val.reshape(orig_shape)
    return val


def _extract(td: Any, field: str) -> Any:
    val = td[field]
    if isinstance(val, list) and val and all(isinstance(t, torch.Tensor) for t in val):
        return torch.stack(val, dim=0)
    return val


class TQTransport(TensorTransport):
    """Transfer Queue backend — batches named tensors into single round-trips."""

    def __init__(
        self,
        runtime: TransferQueueRuntime,
        partition_id: str = _DEFAULT_PARTITION_ID,
    ) -> None:
        self._runtime = runtime
        self._partition_id = partition_id

    @property
    def _client(self) -> Any:
        return self._runtime.client

    async def _fetch(self, handles: List[TQTensorHandle]) -> List[torch.Tensor]:
        """Resolve handles to tensors, parallel to the input order."""
        groups: Dict[tuple, List[TQTensorHandle]] = {}
        order: List[tuple] = []
        for h in handles:
            gk = h._gkey()
            if gk not in groups:
                groups[gk] = []
                order.append(gk)
            groups[gk].append(h)

        td_of: Dict[tuple, Any] = {}
        for gk in order:
            members = groups[gk]
            union = members[0].meta
            for m in members[1:]:
                union = union.union(m.meta)
            td_of[gk] = await self._client.async_get_data(union)

        return [_restore(_extract(td_of[h._gkey()], h.field), h.orig_shape) for h in handles]

    def put(self, tensor: torch.Tensor) -> Any:
        async def _put() -> Any:
            orig_shape = tuple(tensor.shape)
            t = _store_shape(tensor)
            bs = int(t.shape[0]) if t.dim() > 0 else 1
            td = TensorDict({"_": t}, batch_size=bs).cpu()
            put_meta = await self._client.async_put(data=td, partition_id=self._partition_id)
            return TQTensorHandle(meta=put_meta.select_fields(["_"]), field="_", orig_shape=orig_shape)

        return _run_async_in_temp_loop(_put)

    def _resolve_handles(self, handles: List[TQTensorHandle]) -> List[torch.Tensor]:
        return _run_async_in_temp_loop(self._fetch, handles)

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorRef)

    def put_batch(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, TensorRef]:
        if not tensors:
            return {}

        async def _put_batch() -> Dict[str, TensorRef]:
            def _bs(t: Any) -> int:
                if isinstance(t, torch.Tensor):
                    return int(t.shape[0]) if t.dim() > 0 else 1
                return 1  # list / NonTensorData → one row

            groups: Dict[int, list] = {}
            for k, t in tensors.items():
                if not isinstance(t, (torch.Tensor, list)):
                    raise TypeError(f"TQTransport.put_batch: unsupported type {type(t).__name__} for key {k!r}")
                groups.setdefault(_bs(t), []).append(k)

            result: Dict[str, TensorRef] = {}
            for bs, keys in groups.items():
                d: dict = {}
                for k in keys:
                    t = tensors[k]
                    d[k] = _store_shape(t) if isinstance(t, torch.Tensor) else NonTensorData(t)
                td = TensorDict(d, batch_size=bs).cpu()
                put_meta = await self._client.async_put(data=td, partition_id=self._partition_id)
                for k in keys:
                    t = tensors[k]
                    is_tensor = isinstance(t, torch.Tensor)
                    result[k] = TensorRef(
                        spans=[
                            TensorSpan(
                                TQTensorHandle(
                                    meta=put_meta.select_fields([k]),
                                    field=k,
                                    orig_shape=tuple(t.shape) if is_tensor else None,
                                ),
                                0,
                                bs,
                            )
                        ],
                        shape=tuple(t.shape) if is_tensor else None,
                        dtype=t.dtype if is_tensor else None,
                        device=str(t.device) if is_tensor else None,
                    )
            return result

        return _run_async_in_temp_loop(_put_batch)


__all__ = ["TQTransport", "TQTensorHandle"]
