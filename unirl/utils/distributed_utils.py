"""Distributed helper utilities shared by rollout-side weight sync."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)

GLOO_GROUP = None


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

    # Detect the correct keyword for process group options. PyTorch renamed
    # this parameter across versions:
    #   < 2.6  : pg_options
    #   2.6-2.x: backend_options
    #   some builds: neither (positional only or removed)
    # Introspect the actual signature to avoid version-string comparison bugs.
    #
    # Context: SGLang separate mode uses the vllm-sglang-omni-v3 image which
    # ships a newer PyTorch (≥2.6) that renamed pg_options → backend_options.
    # The unirl:latest image still uses the old name. Naive string
    # comparison (str(torch.__version__) >= "2.6") breaks on versions like
    # "2.10" (string '1' < '6'). Using inspect.signature avoids all such
    # issues — we just ask the function what it actually accepts.
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


def distributed_masked_whiten(
    values: torch.Tensor,
    mask: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
    shift_mean: bool = True,
    epsilon: float = 1e-8,
):
    """Whiten tensors using global statistics across the process group."""
    local_sum = (values * mask).sum()
    local_sum_sq = ((values**2) * mask).sum()
    local_mask_sum = mask.sum()

    stats_tensor = torch.tensor(
        [local_sum, local_sum_sq, local_mask_sum],
        device=values.device,
        dtype=torch.float32,
    )
    dist.all_reduce(stats_tensor, group=process_group)

    global_sum, global_sum_sq, global_mask_sum = stats_tensor
    if global_mask_sum.item() == 0:
        raise ValueError("The global mask sum across all participating GPUs is zero.")

    global_mean = global_sum / global_mask_sum
    global_mean_sq = global_sum_sq / global_mask_sum
    global_var = global_mean_sq - global_mean**2

    if global_mask_sum.item() >= 2:
        bessel_correction = global_mask_sum / (global_mask_sum - 1)
        global_var = global_var * bessel_correction

    whitened_values = (values - global_mean) * torch.rsqrt(global_var + epsilon)
    if not shift_mean:
        whitened_values += global_mean
    return whitened_values
