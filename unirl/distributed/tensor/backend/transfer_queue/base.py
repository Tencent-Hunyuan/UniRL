"""Shared transfer-queue primitives: backend abstract base."""

from __future__ import annotations

import abc
from typing import Any, ClassVar


class Backend(abc.ABC):
    """Driver-side bootstrap + per-actor wire-dict producer.

    Each backend variant ``bootstrap()``s any backend-specific Ray actors
    (storage units for the simple backend; nothing for Mooncake) and returns
    the *actor handoff* dict that ``TransferQueueRuntime.create_client``
    consumes. The handoff carries ``manager_type`` (the upstream
    ``initialize_storage_manager`` discriminator), ``controller_info``, and
    backend-specific keys.

    ``specialize_for_controller`` produces the controller-only variant of the
    handoff (Mooncake swaps in smaller zero-copy buffers); default is identity.

    Backend instances hold spawned actor handles as instance attributes so
    they stay alive as long as the backend object is referenced.
    """

    manager_type: ClassVar[str]

    @abc.abstractmethod
    def bootstrap(self, *, controller_info: Any) -> dict:
        raise NotImplementedError

    def specialize_for_controller(self, actor_handoff: dict) -> dict:
        return dict(actor_handoff)


__all__ = ["Backend"]
