"""Dispatch modes, dispatch/collect functions, and @distributed decorator.

All dispatch/collect logic lives here — single source of truth.
Handle imports from this module and uses DISPATCH_MODE_REGISTRY.

Design:
  - Dispatch enum: declares how input flows to workers
  - Execute enum: declares which workers run
  - Each Dispatch mode is paired with dispatch_fn + collect_fn in DISPATCH_MODE_REGISTRY
  - dispatch/collect functions take (wg, args, kwargs, batch_size) to access rank_info, dp_size, etc.
  - @distributed decorator marks Remote methods with their dispatch/execute modes

DP-aware dispatch (DP_SCATTER, DP_SCATTER_HEAD):
  - Input is split by dp_size (not world_size) using recursive pytree_chunk
  - Workers in the same DP group (varying TP/PP/SP rank) receive the SAME shard
  - Collect filters: only tp_rank==0, pp_last_stage, sp_rank==0 results are kept
  - Kept results are merged via pytree_cat to reconstruct the full batch
"""

from __future__ import annotations

from enum import Enum, auto
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeAlias

from unirl.distributed.tensor.pytree import pytree_cat, pytree_chunk
from unirl.distributed.utils import Broadcast

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle


# Per-worker dispatch payload: (positional args, keyword args).
Shard: TypeAlias = Tuple[Tuple[Any, ...], Dict[str, Any]]

# Signature contract for entries in DISPATCH_MODE_REGISTRY["dispatch_fn"].
DispatchFn: TypeAlias = Callable[
    ["Handle", Tuple[Any, ...], Dict[str, Any], Optional[int]],
    List[Shard],
]

# Signature contract for entries in DISPATCH_MODE_REGISTRY["collect_fn"].
CollectFn: TypeAlias = Callable[["Handle", List[Any]], Any]


def _unwrap_broadcast(args: tuple, kwargs: dict):
    """Strip top-level Broadcast wrappers from args and kwargs.

    Broadcast is a controller-side dispatch annotation: it must be consumed
    here and never reach workers. Only top-level args/kwargs values can be
    Broadcast — nesting is not supported.
    """
    clean_args = tuple(v.value if isinstance(v, Broadcast) else v for v in args)
    clean_kwargs = {k: (v.value if isinstance(v, Broadcast) else v) for k, v in kwargs.items()}
    return clean_args, clean_kwargs


# ── Enums ──


class Dispatch(Enum):
    """How to distribute input to workers."""

    BROADCAST = auto()  # Same data to every worker
    SCATTER = auto()  # Split N ways across world (one shard per worker)
    DP_SCATTER = auto()  # Chunk by dp_size; all ranks in DP group get the same shard; collect merge
    DP_SCATTER_HEAD = auto()  # Chunk by dp_size; only DP head gets shard, others empty; collect merge


class Execute(Enum):
    """Which workers execute."""

    ALL = auto()  # All workers execute
    RANK_ZERO = auto()  # Only rank 0 executes


# ── Dispatch functions (wg, args, kwargs, batch_size) → List[(args_i, kwargs_i)] ──


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
    """Split args/kwargs by world_size (treat every worker as its own DP rank).

    Equivalent to DP_SCATTER with dp_size == world_size.
    """
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
    """Split args/kwargs by dp_size, assign by dp_rank.

    Workers in the same DP group (different TP/PP/SP ranks) receive
    the identical data shard. Each worker is responsible for internal
    slicing (TP slices hidden dim, PP runs its own layers, etc.).

    If batch_size is None (all broadcast), replicate to all workers.
    """
    dp_size = wg.dp_size

    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs)] * wg.world_size

    # Split into dp_size shards
    split_args = tuple(pytree_chunk(v, dp_size, batch_size) for v in args)
    split_kwargs = {k: pytree_chunk(v, dp_size, batch_size) for k, v in kwargs.items()}

    # Build per-dp-rank shards
    dp_shards = []
    for dp_rank in range(dp_size):
        shard_args = tuple(split_args[j][dp_rank] for j in range(len(args)))
        shard_kwargs = {k: split_kwargs[k][dp_rank] for k in kwargs}
        dp_shards.append((shard_args, shard_kwargs))

    # Map each worker to its DP shard
    return [dp_shards[wg.rank_infos[i].dp_rank] for i in range(wg.world_size)]


def _dispatch_dp_scatter_head(
    wg: "Handle",
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    batch_size: Optional[int],
) -> List[Shard]:
    """Like DP_SCATTER, but non-head ranks receive empty args/kwargs.

    DP head rank per group: tp_rank==0, pp_rank==0, sp_rank==0.
    This saves RPC bandwidth when workers broadcast data internally.
    """
    dp_size = wg.dp_size

    if batch_size is None:
        args, kwargs = _unwrap_broadcast(args, kwargs)
        return [(args, kwargs) if _is_dp_head(wg.rank_infos[i]) else ((), {}) for i in range(wg.world_size)]

    # Split into dp_size shards
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


# ── Collect functions (wg, results) → collected ──


def _collect_passthrough(wg, results: List) -> List:
    """Return all results as list (raw)."""
    return results


def _collect_dp_merge(wg, results: List) -> Any:
    """Collect only DP-head results per DP group, then merge.

    DP head: tp_rank==0, is_pipeline_last_stage, sp_rank==0.
    Returns the pytree_cat'd result across DP ranks.

    Handles Execute.RANK_ZERO case where len(results) < world_size.
    """
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


# ── Registry: Dispatch mode → paired (dispatch_fn, collect_fn) ──

DISPATCH_MODE_REGISTRY: Dict[Dispatch, Dict[str, Callable]] = {
    Dispatch.BROADCAST: {"dispatch_fn": _dispatch_broadcast, "collect_fn": _collect_passthrough},
    Dispatch.SCATTER: {"dispatch_fn": _dispatch_scatter, "collect_fn": _collect_passthrough},
    Dispatch.DP_SCATTER: {"dispatch_fn": _dispatch_dp_scatter, "collect_fn": _collect_dp_merge},
    Dispatch.DP_SCATTER_HEAD: {"dispatch_fn": _dispatch_dp_scatter_head, "collect_fn": _collect_dp_merge},
}


# ── Backward dispatch mode resolution ────────────────────────────────────────


def resolve_backward_dispatch_mode(
    method_name: str,
    fwd_dispatch_mode: Dispatch,
    rank_infos: list,
) -> Dispatch:
    """Return the dispatch mode for the backward RPC, or raise if unsupported.

    Rules:
      DP_SCATTER      + pp_size==1 → DP_SCATTER (grad shards align with output shards)
      DP_SCATTER_HEAD + pp_size==1 → DP_SCATTER (all ranks must participate in backward)
      DP_SCATTER / DP_SCATTER_HEAD + pp_size>1 → Error (autograd graph broken across PP)
      BROADCAST → Error
      SCATTER   → Error

    !! IMPORTANT — adding a new Dispatch variant !!
    Update this function to decide whether DP_SCATTER backward is correct,
    or a hard error is needed.  Also check Remote._auto_backward's dispatch_mode.
    """
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

    # DP_SCATTER_HEAD → DP_SCATTER (all ranks must participate in backward)
    # DP_SCATTER      → DP_SCATTER (unchanged)
    return Dispatch.DP_SCATTER


# ── @distributed decorator ──

DISTRIBUTED_CONFIG_ATTR = "_distributed_config"


def distributed(
    _func: Callable = None,
    *,
    dispatch_mode: Dispatch = Dispatch.DP_SCATTER,
    execute_mode: Execute = Execute.ALL,
) -> Callable:
    """Declare SPMD dispatch/execute mode on a Role method.

    Handle scans for this attribute and auto-generates proxy methods.
    Default dispatch mode is DP_SCATTER.

    Usage:
        class DiffusionRemote(Remote):
            @distributed
            def rollout(self, samples, prompts):
                ...

            @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
            def get_metrics(self):
                ...
    """

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
        # Called as @distributed without parentheses
        return decorator(_func)
    return decorator
