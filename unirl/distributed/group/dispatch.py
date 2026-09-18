"""Dispatch modes, dispatch/collect functions, and @distributed decorator."""

from __future__ import annotations

from collections import Counter
from enum import Enum, auto
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, TypeAlias

from unirl.distributed.tensor.pytree import pytree_cat, pytree_chunk
from unirl.distributed.tensor.ref import TensorRef, ref_is_required, ref_store_keys
from unirl.distributed.utils import Broadcast, collect_leaves

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


# ── Partial localization (``reads=`` / ``skips=``) ──


def _subtree_store_keys(selector: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Set[Any]:
    keys: Set[Any] = set()
    for ref in collect_leaves(selector(*args, **kwargs), TensorRef):
        keys |= ref_store_keys(ref)
    return keys


def has_partial_localization(config: Optional[Dict[str, Any]]) -> bool:
    """Whether a ``@distributed`` config narrows localization at all."""
    return bool(config) and (config.get("reads") is not None or config.get("skips") is not None)


def required_store_keys(
    config: Optional[Dict[str, Any]], args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Optional[Set[Any]]:
    """Return the storage keys one shard must resolve, or ``None`` to resolve all."""
    if not has_partial_localization(config):
        return None
    if not args and not kwargs:
        return set()
    reads_fn, skips_fn = config.get("reads"), config.get("skips")
    if reads_fn is not None:
        return _subtree_store_keys(reads_fn, args, kwargs)

    # Blacklist. Identity plus occurrence counts decides which refs are inside the
    # skipped subtrees. A selector that hands back a VIEW instead of the tree's own
    # ref matches nothing and degrades to full localization. If one ref object is
    # aliased both inside and outside the skipped subtree, the unmatched occurrence
    # keeps its key required.
    all_refs = collect_leaves(args, TensorRef) + collect_leaves(kwargs, TensorRef)
    skipped_refs = collect_leaves(skips_fn(*args, **kwargs), TensorRef)
    all_counts = Counter(id(ref) for ref in all_refs)
    skipped_counts = Counter(id(ref) for ref in skipped_refs)
    everything: Set[Any] = set()
    skipped: Set[Any] = set()
    claimed_elsewhere: Set[Any] = set()
    for ref in all_refs:
        keys = ref_store_keys(ref)
        everything |= keys
        ref_id = id(ref)
        if skipped_counts[ref_id]:
            skipped |= keys
        if all_counts[ref_id] > skipped_counts[ref_id]:
            claimed_elsewhere |= keys
    return everything - (skipped - claimed_elsewhere)


def remap_required_store_keys(
    required: Set[Any],
    before_args: Tuple[Any, ...],
    before_kwargs: Dict[str, Any],
    after_args: Tuple[Any, ...],
    after_kwargs: Dict[str, Any],
) -> Set[Any]:
    """Translate a pre-localization mask onto structurally identical localized refs."""
    before_refs = collect_leaves(before_args, TensorRef) + collect_leaves(before_kwargs, TensorRef)
    after_refs = collect_leaves(after_args, TensorRef) + collect_leaves(after_kwargs, TensorRef)
    if len(before_refs) != len(after_refs):
        raise RuntimeError("TensorTransport.localize changed the TensorRef tree structure")

    remapped: Set[Any] = set()
    for before, after in zip(before_refs, after_refs):
        if ref_is_required(before, required):
            remapped |= ref_store_keys(after)
    return remapped


# ── @distributed decorator ──

DISTRIBUTED_CONFIG_ATTR = "_distributed_config"


def distributed(
    _func: Callable = None,
    *,
    dispatch_mode: Dispatch = Dispatch.DP_SCATTER,
    execute_mode: Execute = Execute.ALL,
    reads: Optional[Callable] = None,
    skips: Optional[Callable] = None,
) -> Callable:
    """Declare SPMD dispatch/execute mode and optional tensor-read selectors on a Role method."""
    if reads is not None and skips is not None:
        raise ValueError("@distributed takes reads= or skips=, not both: the mask would be ambiguous.")

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
                "reads": reads,
                "skips": skips,
            },
        )
        return wrapper

    if _func is not None:
        return decorator(_func)
    return decorator
