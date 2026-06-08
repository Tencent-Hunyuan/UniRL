"""Driver- and actor-side TransferQueue lifecycle.

``TransferQueueRuntime`` owns the per-process state — the TQ client plus the
driver-only ``Backend`` and ``TransferQueueController`` anchors — and exposes
it as instance methods. Exactly one runtime is "current" per process; the
``TQTransport`` wraps the client for the ``TensorTransport`` interface.

The driver instantiates one runtime, calls ``install()`` to bind it as
current, then ``init(cfg)`` to spawn the controller and bootstrap the
backend. ``init`` returns the *(controller_handoff, actor_handoff)* tuple
that flows over Ray RPC; both sides ultimately feed their handoff to
``create_client``. Disabled = ``cfg.transfer_queue`` is absent — ``init``
returns ``None`` and the runtime stays empty.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Callable

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.tensor.backend.transfer_queue.base import Backend

if TYPE_CHECKING:
    from ray.actor import ActorHandle
    from transfer_queue import AsyncTransferQueueClient, TransferQueueClient


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("TQ_LOGGING_LEVEL", "WARN"))

_DEFAULT_PARTITION_ID = "train_partition"


def _get_local_ip() -> str:
    import socket

    host = socket.gethostname()
    return socket.gethostbyname(host)


def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    """Run a coroutine on a fresh background event loop.

    Needed because the calling context (server mode) may already own an event
    loop, and we can't reuse it for synchronous bridging.
    """
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta dataproto converter",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


class TransferQueueRuntime:
    """Per-process owner of the TransferQueue client + driver-side anchors.

    Exactly one runtime is "current" per process. ``current()`` returns it,
    ``install()`` binds ``self`` as current, ``clear_current()`` unbinds.
    On actors, ``backend`` and ``controller`` stay ``None``; on the driver,
    ``init()`` populates them.
    """

    _current: "TransferQueueRuntime | None" = None

    def __init__(self) -> None:
        self.client: "AsyncTransferQueueClient | TransferQueueClient | None" = None
        self.backend: "Backend | None" = None
        self.controller: "ActorHandle | None" = None

    # -- process-singleton plumbing ---------------------------------------

    @classmethod
    def current(cls) -> "TransferQueueRuntime | None":
        """Return the runtime bound to this process, or ``None`` if unbound."""
        return cls._current

    def install(self) -> "TransferQueueRuntime":
        """Bind ``self`` as the current process runtime. Returns ``self``."""
        if type(self)._current is not None and type(self)._current is not self:
            logger.warning("TransferQueueRuntime: replacing existing current runtime")
        type(self)._current = self
        return self

    @classmethod
    def clear_current(cls) -> None:
        """Unbind the process runtime (test/teardown helper)."""
        cls._current = None

    # -- driver-side ------------------------------------------------------

    def init(self, cfg: DictConfig) -> "tuple[dict, dict] | None":
        """Spawn controller + backend-side actors; return ``(controller, actor)`` handoffs.

        Returns ``None`` when ``cfg.transfer_queue`` is absent (TQ disabled).
        """
        tq_cfg = cfg.get("transfer_queue")
        if tq_cfg is None:
            return None

        from transfer_queue import TransferQueueController, process_zmq_server_info

        # The `transfer_queue:` block is a standard Hydra _target_ config (the
        # backend and its nested zero_copy both carry `_target_`), so instantiate
        # it directly.
        self.backend = instantiate(tq_cfg)
        self.controller = TransferQueueController.remote()
        controller_info = process_zmq_server_info(self.controller)

        actor_handoff = self.backend.bootstrap(controller_info=controller_info)
        controller_handoff = self.backend.specialize_for_controller(actor_handoff)
        return controller_handoff, actor_handoff

    def init_remote_actor_clients(self, actors: list, handoff: dict) -> None:
        """Fan ``handoff`` out to each actor's ``init_transferqueue_client`` RPC."""
        import ray

        refs = [actor.init_transferqueue_client.remote(handoff=handoff) for actor in actors]
        ray.get(refs)

    def reset_actors_zero_copy_buffer_free(self, actors: list) -> None:
        # Zero-copy buffer free-list reset is Mooncake-specific; skip it for
        # other backends (e.g. the simple in-Ray storage backend has no such
        # buffers, so the upstream reset call is meaningless there).
        if self.backend is None or self.backend.manager_type != "MooncakeStorageManager":
            return
        import ray

        refs = [actor.reset_zero_copy_buffer_free.remote() for actor in actors]
        ray.get(refs)

    # -- per-process (driver and actor) -----------------------------------

    def create_client(
        self,
        client_id: str,
        handoff: dict,
        *,
        sync: bool = False,
    ) -> "AsyncTransferQueueClient | TransferQueueClient":
        """Construct (and cache on ``self``) the per-process TransferQueue client."""
        from transfer_queue import AsyncTransferQueueClient, TransferQueueClient

        if handoff.get("manager_type") == "MooncakeStorageManager":
            # Mooncake binds to LOCAL_IP; otherwise it picks a random interface.
            local_ip = os.getenv("LOCAL_IP", _get_local_ip())
            os.environ["MC_TCP_BIND_ADDRESS"] = local_ip
            handoff["local_hostname"] = local_ip

            # Per-process GPU↔HCA affinity. With this env flag set and a
            # comma-list device_name, Mooncake's setup() picks the PIX-distance
            # HCA from the active CUDA context. Without it, every client binds
            # to the first listed bond regardless of GPU placement, causing
            # `-800` on wrong-NUMA ranks once CUDA initializes.
            # See LIN-186/docs/mooncake_-800_diagnosis.md (probe G6).
            os.environ["MC_ENABLE_DEST_DEVICE_AFFINITY"] = "1"
            if not handoff.get("device_name"):
                from unirl.distributed.tensor.backend.transfer_queue.topology import list_rdma_bonds

                bonds = list_rdma_bonds()
                if not bonds:
                    raise RuntimeError(
                        "transfer_queue: no InfiniBand device found under "
                        "/sys/class/infiniband; Mooncake requires an RDMA-capable NIC"
                    )
                handoff["device_name"] = ",".join(bonds)
                logger.info("TQ device_name auto-discovered: %s", handoff["device_name"])
            else:
                logger.info("TQ device_name explicit override: %s", handoff["device_name"])
        logger.info(f"create_transferqueue_client, handoff: {handoff}")

        if self.client is not None:
            logger.warning("transferqueue_client already exists!")
            return self.client

        client_cls = TransferQueueClient if sync else AsyncTransferQueueClient
        self.client = client_cls(client_id, handoff.get("controller_info"))
        self.client.initialize_storage_manager(manager_type=handoff["manager_type"], config=handoff)
        return self.client

    def is_enabled(self) -> bool:
        """True iff this runtime has a client wired up."""
        return self.client is not None

    def reset_zero_copy_buffer_free(self) -> None:
        if self.client is None:
            return
        from transfer_queue.storage.clients.mooncake_client import RegisterBufferType

        _run_async_in_temp_loop(self.client.async_reset_zero_copy_all_keys_free, RegisterBufferType.GET_BYTES)
        _run_async_in_temp_loop(self.client.async_reset_zero_copy_all_keys_free, RegisterBufferType.GET_TENSOR)
        _run_async_in_temp_loop(self.client.async_reset_zero_copy_all_keys_free, RegisterBufferType.PUT_TENSOR)
        _run_async_in_temp_loop(self.client.async_reset_zero_copy_all_keys_free, RegisterBufferType.PUT_BYTES)

    def clear_partition(self, partition_id: str = _DEFAULT_PARTITION_ID) -> None:
        if self.client is None:
            return
        self.client.clear_partition(partition_id)


__all__ = [
    "_DEFAULT_PARTITION_ID",
    "_run_async_in_temp_loop",
    "TransferQueueRuntime",
]
