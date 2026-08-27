"""FSDP2 model wrapping."""

from __future__ import annotations

import logging
from functools import partial
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from unirl.config.require import require
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


def _clone_checkpoint_kwarg(value: Any) -> Any:
    """Snapshot mutable KV-cache mappings without duplicating tensor storage."""
    if not (hasattr(value, "key_cache") and hasattr(value, "value_cache")):
        return value
    cloned = type(value)(value.num_layers)
    # BAGEL cache updates replace per-layer entries; they do not mutate the
    # existing K/V tensors. Copying the mappings is therefore sufficient to
    # freeze replay state and avoids O(num_layers**2) tensor duplication.
    cloned.key_cache = dict(value.key_cache)
    cloned.value_cache = dict(value.value_cache)
    return cloned


def _checkpoint_with_kwarg_snapshots(function: Any, *args: Any, **kwargs: Any) -> Any:
    """Checkpoint mutable kwargs from a frozen call-time mapping snapshot."""
    from torch.utils import checkpoint as torch_checkpoint

    checkpoint_kwargs = {key: _clone_checkpoint_kwarg(value) for key, value in kwargs.items()}

    def run(*inner_args: Any, **inner_kwargs: Any) -> Any:
        call_kwargs = {key: _clone_checkpoint_kwarg(value) for key, value in inner_kwargs.items()}
        return function(*inner_args, **call_kwargs)

    return torch_checkpoint.checkpoint(run, *args, use_reentrant=False, **checkpoint_kwargs)


def fsdp_wrap(
    model: nn.Module,
    stage: Optional[object] = None,
    *,
    block_class_names: Optional[Tuple[str, ...]] = None,
    param_dtype: str = "bf16",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    cast_forward_inputs: bool = True,
    fsdp_mode: str = "full",
    reshard_after_forward: bool = True,
    forward_prefetch: bool = False,
    activation_checkpointing: bool = False,
    ac_wrap_order: str = "outside",
    use_torch_compile: bool = False,
    master_dtype: Optional[str] = None,
    master_params: Tuple[torch.Tensor, ...] = (),
    root_wrap: bool = True,
) -> None:
    """Apply FSDP2 wrapping to the model.  No handle returned — DTensors"""
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        fully_shard,
    )
    from torch.distributed.tensor import DTensor

    target_dtype = parse_torch_dtype(param_dtype, field_name="training.fsdp.param_dtype")
    trainable_dtype = (
        parse_torch_dtype(master_dtype, field_name="training.fsdp.master_dtype") if master_dtype is not None else None
    )
    require(
        ac_wrap_order in {"inside", "outside"},
        f"fsdp_wrap: ac_wrap_order must be 'inside' or 'outside', got {ac_wrap_order!r}",
    )

    fsdp_kwargs: Dict[str, object] = {
        "reshard_after_forward": bool(reshard_after_forward),
    }
    if mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=target_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=bool(cast_forward_inputs),
        )
    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    mesh = _create_device_mesh(fsdp_mode)
    if mesh is not None:
        fsdp_kwargs["mesh"] = mesh

    if block_class_names is None:
        block_class_names = _discover_block_classes(model, stage)
    block_instances = _enumerate_block_instances(model, block_class_names)

    casts = 0
    master_param_ids = {id(p) for p in master_params}
    # Keep trainable and EMA shadow masters at master_dtype.
    for p in model.parameters():
        if isinstance(p, DTensor) or not p.dtype.is_floating_point:
            continue  # already-wrapped params and ints never cast
        if trainable_dtype is not None and (p.requires_grad or id(p) in master_param_ids):
            dst = trainable_dtype
        elif not mixed_precision:
            dst = target_dtype
        else:
            continue
        if p.dtype != dst:
            p.data = p.data.to(dst)
            casts += 1

    if activation_checkpointing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            CheckpointWrapper,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        block_ids = {id(layer) for layer in block_instances}
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                checkpoint_fn=_checkpoint_with_kwarg_snapshots,
            ),
            check_fn=lambda module: id(module) in block_ids,
        )
        # Where fully_shard lands relative to the AC wrapper decides whether the
        # recompute re-enters FSDP's gather/cast hooks:
        #
        # * "outside" (default; fully_shard on the CheckpointWrapper, torchtitan
        #   order): hooks fire once per use, outside the checkpoint region. This
        #   matches the composition every pre-knob AC recipe ran — the old
        #   forward-monkeypatch checkpoint also recomputed without re-entering
        #   hooks — so untouched recipes keep their validated behavior.
        # * "inside" (fully_shard on the INNER block): the recompute goes through
        #   the module's __call__, re-running the pre-forward gather and the
        #   mp_policy cast. Opt in per recipe where this order was actually
        #   smoke-validated (the stacked BAGEL it2i consumer pins it).
        if ac_wrap_order == "outside":
            wrapped = [m for m in model.modules() if isinstance(m, CheckpointWrapper)]
            require(
                len(wrapped) == len(block_instances),
                f"fsdp_wrap: expected {len(block_instances)} checkpoint wrappers, found {len(wrapped)}",
            )
            block_instances = wrapped

    for layer in block_instances:
        fully_shard(layer, **fsdp_kwargs)

    if root_wrap and not isinstance(model, FSDPModule):
        # Root-wrap leftover parameters but keep them materialized after forward.
        root_kwargs = dict(fsdp_kwargs)
        root_kwargs.pop("reshard_after_forward", None)
        fully_shard(model, **root_kwargs)
    else:
        # Reject trainable parameters outside FSDP groups to prevent rank drift.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            stray = [n for n, p in model.named_parameters() if p.requires_grad and not isinstance(p, DTensor)]
            require(
                not stray,
                f"fsdp_wrap(root_wrap=false): {len(stray)} trainable param(s) sit outside "
                f"every fully_shard group (e.g. {stray[:3]}); their grads would never be "
                "DP-synced and replicas drift. Enable training.fsdp.root_wrap or freeze them.",
            )

    if forward_prefetch:
        if not isinstance(model, FSDPModule):
            raise ValueError(
                "fsdp_wrap: forward_prefetch=True needs the model root-wrapped so FSDP2 "
                "initializes the shared all-gather comm context, but root_wrap did not run "
                "(training.fsdp.root_wrap=False). Set root_wrap=True, or forward_prefetch=False."
            )
        fsdp_groups = [m for m in model.modules() if isinstance(m, FSDPModule)]
        for cur, nxt in zip(fsdp_groups, fsdp_groups[1:]):
            cur.set_modules_to_forward_prefetch([nxt])

    if use_torch_compile:
        for layer in block_instances:
            layer.forward = torch.compile(layer.forward)

    if _current_rank() == 0:
        logger.info(
            "fsdp_wrap: wrapped %d block(s) of class %r "
            "(%s, cpu_offload=%s, mixed_precision=%s, cast_forward_inputs=%s, reshard=%s, prefetch=%s, "
            "ac=%s, compile=%s, dtype_casts=%d, master_dtype=%s, root_wrap=%s)",
            len(block_instances),
            tuple(block_class_names),
            fsdp_mode,
            cpu_offload,
            mixed_precision,
            cast_forward_inputs,
            reshard_after_forward,
            forward_prefetch,
            activation_checkpointing,
            use_torch_compile,
            casts,
            master_dtype,
            root_wrap,
        )


def _discover_block_classes(model: nn.Module, stage: object) -> Tuple[str, ...]:
    for cls in type(model).__mro__:
        attr = getattr(cls, "_no_split_modules", None)
        if attr:
            return tuple(str(n) for n in attr)
    leaf_source = stage
    while hasattr(leaf_source, "source"):
        leaf_source = leaf_source.source
    attr = getattr(type(leaf_source), "_no_split_modules", None)
    if attr:
        return tuple(str(n) for n in attr)
    if _current_rank() == 0:
        logger.warning(
            "fsdp_wrap: no block classes discovered for %r (stage %r). Falling back to root-only wrap.",
            type(model).__name__,
            type(leaf_source).__name__,
        )
    return ()


def _enumerate_block_instances(
    model: nn.Module,
    class_names: Tuple[str, ...],
) -> Tuple[nn.Module, ...]:
    if not class_names:
        return ()
    names = set(class_names)
    return tuple(m for _, m in model.named_modules() if type(m).__name__ in names)


# Parameter shard degree: full = world default, hybrid = 8 ranks, no_shard = 1 rank (DDP).
_SHARD_DEGREE: Dict[str, Optional[int]] = {"full": None, "hybrid": 8, "no_shard": 1}


def _create_device_mesh(fsdp_mode: str) -> Optional[object]:
    mode = str(fsdp_mode).strip().lower()
    require(
        mode in _SHARD_DEGREE,
        f"training.fsdp.fsdp_mode={fsdp_mode!r} is not one of {sorted(_SHARD_DEGREE)}; "
        "an unrecognized mode would silently fall back to full sharding.",
    )
    shard_size = _SHARD_DEGREE[mode]
    if shard_size is None:
        return None

    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return None

    world_size = dist.get_world_size()
    # A world that the shard degree cannot split (including single-rank
    # ``no_shard``) already matches the default 1D mesh.
    if world_size <= shard_size or world_size % shard_size != 0:
        return None

    from torch.distributed.device_mesh import init_device_mesh

    replicate_size = world_size // shard_size
    mesh = init_device_mesh(
        "cuda",
        (replicate_size, shard_size),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    logger.info("fsdp_wrap: %s mesh dp_replicate=%d x dp_shard=%d", mode, replicate_size, shard_size)
    return mesh


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0
