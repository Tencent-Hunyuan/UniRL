"""Distributed coordination helpers (lazy re-exports).

Contract:

- ``distributed/`` owns distributed coordination semantics and sync protocols
- it may depend on the active transport/runtime boundary when needed
- it must not own Ray actors, group construction, placement, or business workflow

Re-exports are lazy via ``__getattr__`` so that importing a leaf module
(e.g. ``unirl.distributed.tensor.transport``) does NOT eagerly
pull in ``weight_sync`` → ``rollout/engine`` → ``types/rollout_req``,
which would close the loop back on a mid-init ``types/conditions/base``.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {}

__all__ = list(_LAZY_ATTRS.keys())


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
