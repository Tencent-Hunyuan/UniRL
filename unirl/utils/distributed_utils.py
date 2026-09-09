"""Shared distributed-process and process-group utilities."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.distributed_c10d import Backend, Store

logger = logging.getLogger(__name__)

GLOO_GROUP = None


def find_dtensor_mesh(model: torch.nn.Module) -> DeviceMesh | None:
    """Return one DTensor parameter's device mesh."""
    from torch.distributed.tensor import DTensor

    for param in model.parameters():
        if isinstance(param, DTensor):
            return param.device_mesh
    return None


def ensure_dist_initialized(local_rank: int | None = None) -> None:
    """Idempotently bring up the default process group."""
    if not dist.is_available():
        raise RuntimeError("torch.distributed is unavailable")
    if torch.cuda.is_available() and local_rank is not None:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group()
        logger.info(
            "ensure_dist_initialized: default process group up (rank=%s world=%s)",
            dist.get_rank(),
            dist.get_world_size(),
        )


def init_gloo_group():
    """Initialize and memoize a shared gloo process group."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        GLOO_GROUP = dist.new_group(backend="gloo")
    return GLOO_GROUP


def get_gloo_group():
    """Return the shared gloo process group."""
    global GLOO_GROUP
    if GLOO_GROUP is None:
        raise RuntimeError("Gloo group has not been initialized. Call init_gloo_group() first.")
    return GLOO_GROUP


def init_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: timedelta | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    """Copy of PyTorch init_process_group that can create extra main groups."""
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (init_method is None), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)
        store = PrefixStore(group_name, store)

    import inspect

    _npg_sig = inspect.signature(_new_process_group_helper)
    _npg_params = set(_npg_sig.parameters.keys())

    pg_extra_kwargs: dict = {}
    if "backend_options" in _npg_params:
        pg_extra_kwargs["backend_options"] = pg_options
    elif "pg_options" in _npg_params:
        pg_extra_kwargs["pg_options"] = pg_options

    default_pg = dist.group.WORLD if dist.is_initialized() else None
    saved_bound_device_id = None
    if default_pg is not None and getattr(default_pg, "bound_device_id", None):
        saved_bound_device_id = default_pg.bound_device_id
        default_pg.bound_device_id = None

    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **pg_extra_kwargs,
        timeout=timeout,
    )

    if saved_bound_device_id is not None:
        default_pg.bound_device_id = saved_bound_device_id

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    return pg
