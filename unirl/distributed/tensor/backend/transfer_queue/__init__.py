"""TransferQueue subsystem: typed config + driver bootstrap + actor bridge.

Each backend variant registers a dataclass under Hydra group ``transfer_queue``
whose ``_target_`` points at a ``Backend`` subclass; ``TransferQueueRuntime.init``
instantiates it via ``hydra.utils.instantiate(cfg.transfer_queue)``. Disabled =
``cfg.transfer_queue`` is absent (no defaults entry).
"""

from unirl.distributed.tensor.backend.transfer_queue.base import Backend
from unirl.distributed.tensor.backend.transfer_queue.mooncake import (
    MooncakeBackend,
    MooncakeBackendConfig,
    MooncakeZeroCopyConfig,
)
from unirl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime
from unirl.distributed.tensor.backend.transfer_queue.simple import (
    SimpleBackend,
    SimpleBackendConfig,
)
from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport

__all__ = [
    "Backend",
    "MooncakeBackend",
    "MooncakeBackendConfig",
    "MooncakeZeroCopyConfig",
    "SimpleBackend",
    "SimpleBackendConfig",
    "TQTransport",
    "TransferQueueRuntime",
]
