"""Shared transfer-queue primitives: backend abstract base."""

from __future__ import annotations

import abc
from typing import Any, ClassVar


class Backend(abc.ABC):
    """Driver-side bootstrap + per-actor wire-dict producer."""

    manager_type: ClassVar[str]

    @abc.abstractmethod
    def bootstrap(self, *, controller_info: Any) -> dict:
        raise NotImplementedError

    def specialize_for_controller(self, actor_handoff: dict) -> dict:
        return dict(actor_handoff)


__all__ = ["Backend"]
