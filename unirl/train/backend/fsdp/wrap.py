"""FSDP2 model wrapping.

:func:`fsdp_wrap` applies per-block ``fully_shard`` to the trainable module.
No handle is returned — the DTensors ARE the handle.  Ported from
``FSDPPolicy._wrap_model``.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from unirl.config.require import require
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


def fsdp_wrap(
    model: nn.Module,
    stage: Optional[object] = None,
    *,
    block_class_names: Optional[Tuple[str, ...]] = None,
    param_dtype: str = "bf16",
    cpu_offload: bool = False,
    mixed_precision: bool = True,
    fsdp_mode: str = "full",
    reshard_after_forward: bool = True,
    forward_prefetch: bool = False,
    activation_checkpointing: bool = False,
    use_torch_compile: bool = False,
    master_dtype: Optional[str] = None,
    root_wrap: bool = True,
) -> None:
    """Apply FSDP2 wrapping to the model.  No handle returned — DTensors
    ARE the handle.  Ported from FSDPPolicy._wrap_model.

    If ``block_class_names`` is supplied, it takes precedence and
    ``stage`` is ignored for discovery.  Otherwise we fall back to
    ``_discover_block_classes(model, stage)`` (model __mro__ then stage
    source chain).

    ``root_wrap`` (default ON) adds a root ``fully_shard(model)`` after the
    per-block wrap so the leftover params (embed / final norm / lm_head)
    are sharded + mp_policy'd instead of staying plain replicated tensors.
    The root group deliberately does NOT inherit ``reshard_after_forward``:
    FSDP2's auto policy keeps the root's params materialized after forward,
    which stages rely on for direct post-forward submodule calls (e.g. the
    chunked ``lm_head`` in Qwen3 replay). See ``FSDPConfig.root_wrap`` for
    when to disable it.
    """
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

    fsdp_kwargs: Dict[str, object] = {
        "reshard_after_forward": bool(reshard_after_forward),
    }
    if mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=target_dtype,
            reduce_dtype=torch.float32,
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
    # Keep trainable masters at master_dtype; bf16 pre-casting can erase small optimizer steps.
    for p in model.parameters():
        if isinstance(p, DTensor) or not p.dtype.is_floating_point:
            continue  # already-wrapped params and ints never cast
        if trainable_dtype is not None and p.requires_grad:
            dst = trainable_dtype
        elif not mixed_precision:
            dst = target_dtype
        else:
            continue
        if p.dtype != dst:
            p.data = p.data.to(dst)
            casts += 1

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

    if activation_checkpointing:
        from torch.utils import checkpoint as _ckpt

        def _make_ckpt_forward(orig_fwd: object) -> object:
            def wrapped(*args: object, **kwargs: object) -> object:
                def fn(*a: object) -> object:
                    return orig_fwd(*a, **kwargs)

                return _ckpt.checkpoint(fn, *args, use_reentrant=False)

            return wrapped

        for layer in block_instances:
            layer.forward = _make_ckpt_forward(layer.forward)

    if use_torch_compile:
        for layer in block_instances:
            layer.forward = torch.compile(layer.forward)

    if _current_rank() == 0:
        logger.info(
            "fsdp_wrap: wrapped %d block(s) of class %r "
            "(%s, cpu_offload=%s, mixed_precision=%s, reshard=%s, prefetch=%s, "
            "ac=%s, compile=%s, dtype_casts=%d, master_dtype=%s, root_wrap=%s)",
            len(block_instances),
            tuple(block_class_names),
            fsdp_mode,
            cpu_offload,
            mixed_precision,
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
