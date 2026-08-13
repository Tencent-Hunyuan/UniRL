"""Placement context manager for device allocation."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from unirl.distributed.group.device_pool import DevicePool


_current: Optional["Placement"] = None


def current_placement() -> Optional["Placement"]:
    """Return the innermost active placement scope, or None if outside any."""
    return _current


@dataclass
class Placement:
    pool: "DevicePool"
    devices: Tuple[int, ...]
    shared_workers: bool
    parent: Optional["Placement"]
    _base_slot: int
    _next_isolated_slot: int = field(init=False)

    def __post_init__(self) -> None:
        self._next_isolated_slot = self._base_slot

    def assign(self) -> Tuple[List[int], int]:
        """Return ``(device_ids, slot_id)`` for the next ``create_remote`` call."""
        if self.shared_workers:
            return list(self.devices), self._base_slot
        slot = self._next_isolated_slot
        self._next_isolated_slot += 1
        return list(self.devices), slot


def _parent_next_free_slot(parent: Placement) -> int:
    if parent.shared_workers:
        return parent._base_slot + 1
    return parent._next_isolated_slot


@contextmanager
def placement(
    pool: "DevicePool",
    *,
    fraction: float = 1.0,
    shared_workers: bool = True,
) -> Iterator[Placement]:
    """Scope ``DevicePool.create_remote`` calls to a device slab."""
    global _current
    parent = _current

    if parent is None:
        reference_size = pool.num_devices
        candidates: Tuple[int, ...] = tuple(d for d in range(pool.num_devices) if d not in pool._claimed)
    else:
        reference_size = len(parent.devices)
        candidates = parent.devices

    n_take_f = fraction * reference_size
    n_take = int(round(n_take_f))
    if abs(n_take_f - n_take) > 1e-9:
        raise ValueError(
            f"placement(fraction={fraction}) of {reference_size} reference devices "
            f"requires {n_take_f} devices (not an integer)"
        )
    if n_take <= 0:
        raise ValueError(f"placement(fraction={fraction}) yields 0 devices")
    if n_take > len(candidates):
        raise ValueError(
            f"placement(fraction={fraction}) wants {n_take} devices but only {len(candidates)} are available"
        )

    devices = tuple(candidates[:n_take])

    if parent is None:
        base_slot = 0
    elif shared_workers:
        base_slot = parent._base_slot
    else:
        base_slot = _parent_next_free_slot(parent)

    scope = Placement(
        pool=pool,
        devices=devices,
        shared_workers=shared_workers,
        parent=parent,
        _base_slot=base_slot,
    )

    if parent is None:
        pool._claimed.update(devices)

    _current = scope
    try:
        yield scope
    finally:
        _current = parent
        if parent is not None:
            parent._next_isolated_slot = max(parent._next_isolated_slot, scope._next_isolated_slot)


def remote(role_cls, **kwargs):
    """Instantiate ``role_cls`` inside the active ``placement(...)`` block."""
    scope = current_placement()
    if scope is None:
        raise RuntimeError("remote() must be called inside a placement(...) block")
    init_kwargs = {k: _to_marker(v) for k, v in kwargs.items()}
    return scope.pool.create_remote(role_cls, init_kwargs=init_kwargs)


def _to_marker(value):
    """Replace Handle values with serializable HandleRef markers."""
    from unirl.distributed.group.handle import Handle, HandleRef

    if isinstance(value, Handle):
        return HandleRef(
            role_name=value.role_name,
            sp_size=value.sp_size,
            tp_size=value.tp_size,
            pp_size=value.pp_size,
            ep_size=value.ep_size,
        )
    return value


__all__ = ["Placement", "current_placement", "placement", "remote"]
