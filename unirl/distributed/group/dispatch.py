"""Dispatch modes, dispatch/collect functions, and @distributed decorator."""

from __future__ import annotations

from enum import Enum, auto
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeAlias

from unirl.distributed.tensor.pytree import pytree_cat, pytree_chunk
from unirl.distributed.utils import Broadcast

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle


Shard: TypeAlias = Tuple[Tuple[Any, ...], Dict[str, Any]]

DispatchFn: TypeAlias = Callable[
    ["Handle", Tuple[Any, ...], Dict[str, Any], Optional[int]],
    List[Shard],
]

CollectFn: TypeAlias = Callable[["Handle", List[Any]], Any]


def _unwrap_broadcast(args: tuple, kwargs: dict):
    """Strip top-level Broadcast wrappers from args and kwargs."""
    clean_args = tuple(v.value if isinstance(v, Broadcast) else v for v in args)
    clean_kwargs = {k: (v.value if isinstance(v, Broadcast) else v) for k, v in kwargs.items()}
    return clean_args, clean_kwargs


class Dispatch(Enum):
    """How to distribute input to workers."""

    BROADCAST = auto()  # Same data to every worker
    SCATTER = auto()  # Split N ways across world (one shard per worker)
    DP_SCATTER = auto()  # One shard per DP group; all ranks receive it.
    DP_SCATTER_HEAD = auto()  # One shard per DP group; only its head receives it.


class Execute(Enum):
    """Which workers execute."""

    ALL = auto()  # All workers execute
    RANK_ZERO = auto()  # Only rank 0 executes


def _dispatch_broadcast(
    wg: "Handle",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_size: Optional[int],
) -> List[Shard]:
    """Broadcast same args/kwargs to all workers."""
    args, kwargs = _unwrap_broadcast(args, kwargs)
    return [(args, kwargs)] * wg.world_size


def _dispatch_scatter(
    wg: "Handle",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_size: Optional[int],
) -> List[Shard]:
    """Split args/kwargs by world_size (treat every worker as its own DP rank)."""
    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs)] * wg.world_size

    split_args = tuple(pytree_chunk(v, wg.world_size, batch_size) for v in args)
    split_kwargs = {k: pytree_chunk(v, wg.world_size, batch_size) for k, v in kwargs.items()}

    return [
        (tuple(split_args[j][i] for j in range(len(args))), {k: split_kwargs[k][i] for k in kwargs})
        for i in range(wg.world_size)
    ]


def _dispatch_dp_scatter(
    wg: "Handle",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_size: Optional[int],
) -> List[Shard]:
    """Split args/kwargs by dp_size, assign by dp_rank."""
    dp_size = wg.dp_size

    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs)] * wg.world_size

    split_args = tuple(pytree_chunk(v, dp_size, batch_size) for v in args)
    split_kwargs = {k: pytree_chunk(v, dp_size, batch_size) for k, v in kwargs.items()}

    dp_shards = []
    for dp_rank in range(dp_size):
        shard_args = tuple(split_args[j][dp_rank] for j in range(len(args)))
        shard_kwargs = {k: split_kwargs[k][dp_rank] for k in kwargs}
        dp_shards.append((shard_args, shard_kwargs))

    return [dp_shards[wg.rank_infos[i].dp_rank] for i in range(wg.world_size)]


def _dispatch_dp_scatter_head(
    wg: "Handle",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_size: Optional[int],
) -> List[Shard]:
    """Like DP_SCATTER, but non-head ranks receive empty args/kwargs."""
    dp_size = wg.dp_size

    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs) if _is_dp_head(wg.rank_infos[i]) else ((), {}) for i in range(wg.world_size)]

    split_args = tuple(pytree_chunk(v, dp_size, batch_size) for v in args)
    split_kwargs = {k: pytree_chunk(v, dp_size, batch_size) for k, v in kwargs.items()}

    dp_shards = []
    for dp_rank in range(dp_size):
        shard_args = tuple(split_args[j][dp_rank] for j in range(len(args)))
        shard_kwargs = {k: split_kwargs[k][dp_rank] for k in kwargs}
        dp_shards.append((shard_args, shard_kwargs))

    return [
        dp_shards[wg.rank_infos[i].dp_rank] if _is_dp_head(wg.rank_infos[i]) else ((), {}) for i in range(wg.world_size)
    ]


def _is_dp_head(ri) -> bool:
    return ri.tp_rank == 0 and ri.pp_rank == 0 and ri.sp_rank == 0


def _collect_passthrough(wg, results: List) -> List:
    """Return all results as list (raw)."""
    return results


def _collect_dp_merge(wg, results: List) -> Any:
    """Collect only DP-head results per DP group, then merge."""
    dp_results = []
    for i in range(len(results)):
        ri = wg.rank_infos[i]
        if ri.tp_rank == 0 and ri.is_pipeline_last_stage and ri.sp_rank == 0:
            dp_results.append(results[i])

    if not dp_results:
        return None
    if len(dp_results) == 1:
        return dp_results[0]

    return pytree_cat(dp_results)


DISPATCH_MODE_REGISTRY: Dict[Dispatch, Dict[str, Callable]] = {
    Dispatch.BROADCAST: {"dispatch_fn": _dispatch_broadcast, "collect_fn": _collect_passthrough},
    Dispatch.SCATTER: {"dispatch_fn": _dispatch_scatter, "collect_fn": _collect_passthrough},
    Dispatch.DP_SCATTER: {"dispatch_fn": _dispatch_dp_scatter, "collect_fn": _collect_dp_merge},
    Dispatch.DP_SCATTER_HEAD: {"dispatch_fn": _dispatch_dp_scatter_head, "collect_fn": _collect_dp_merge},
}


def resolve_backward_dispatch_mode(
    method_name: str,
    fwd_dispatch_mode: Dispatch,
    rank_infos: list,
) -> Dispatch:
    """Return the dispatch mode for the backward RPC, or raise if unsupported."""
    if fwd_dispatch_mode in (Dispatch.BROADCAST, Dispatch.SCATTER):
        raise ValueError(
            f"Method '{method_name}' uses dispatch_mode={fwd_dispatch_mode.name}, "
            f"which does not support auto-backward (no shared batch dimension). "
            f"Do not call this method inside enable_grad()."
        )

    pp_sizes = {ri.pp_size for ri in rank_infos}
    if any(pp > 1 for pp in pp_sizes):
        raise ValueError(
            f"Method '{method_name}' has pp_size>1. "
            f"Auto-backward cannot propagate gradients across pipeline stages. "
            f"Do not call this method inside enable_grad()."
        )

    return Dispatch.DP_SCATTER


DISTRIBUTED_CONFIG_ATTR = "_distributed_config"


def distributed(
    _func: Callable = None,
    *,
    dispatch_mode: Dispatch = Dispatch.DP_SCATTER,
    execute_mode: Execute = Execute.ALL,
) -> Callable:
    """Declare SPMD dispatch/execute mode on a Role method."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(
            wrapper,
            DISTRIBUTED_CONFIG_ATTR,
            {
                "dispatch_mode": dispatch_mode,
                "execute_mode": execute_mode,
            },
        )
        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator
