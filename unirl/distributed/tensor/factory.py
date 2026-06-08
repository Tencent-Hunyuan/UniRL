"""build_transport — construct the per-Worker TensorTransport for a backend kind.

Called inside the Worker actor process (so the installed
``TensorTransportRuntime`` singleton lives where ``TensorMeta.local()`` runs).
Backend dependencies differ: colocate builds an in-process ``TensorStore``; gpu
needs a ``TensorWorker`` handle (``tw``) injected by ``DevicePool``; transfer
queue bootstraps its per-process TQ client here from the driver's ``tq_handoff``.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.distributed.tensor.transport import TensorTransport


def build_transport(
    kind: Optional[str],
    *,
    worker_id: str,
    device: str,
    device_id: int,
    tw: Any = None,
    tq_handoff: Optional[dict] = None,
    global_rank: Optional[int] = None,
    world_size: int = 1,
) -> TensorTransport:
    """Build the transport for ``kind`` (default ``colocate_store``)."""
    kind = kind or "colocate_store"

    if kind in ("colocate_store", "colocate"):
        from unirl.distributed.tensor.backend.colocate_store.store import TensorStore
        from unirl.distributed.tensor.backend.colocate_store.transport import ColocateStoreTransport

        store = TensorStore(worker_id, device, global_rank, world_size)
        return ColocateStoreTransport(store)

    if kind in ("gpu_store", "gpu"):
        from unirl.distributed.tensor.backend.gpu_store.transport import GPUStoreTransport

        if tw is None:
            raise RuntimeError(
                "gpu_store transport requires a TensorWorker handle (tw); "
                "DevicePool must create and inject one per GPU before build_transport."
            )
        return GPUStoreTransport(worker_id=worker_id, device_id=device_id, device=device, tw=tw)

    if kind in ("transfer_queue", "tq"):
        from unirl.distributed.tensor.backend.transfer_queue.runtime import (
            _DEFAULT_PARTITION_ID,
            TransferQueueRuntime,
        )
        from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport

        if tq_handoff is None:
            raise RuntimeError(
                "transfer_queue transport requires tq_handoff (the driver's TransferQueueRuntime.init() actor handoff)."
            )
        runtime = TransferQueueRuntime().install()
        runtime.create_client("Worker", tq_handoff)
        return TQTransport(runtime, partition_id=_DEFAULT_PARTITION_ID)

    raise ValueError(f"unknown transport kind {kind!r}")


__all__ = ["build_transport"]
