"""FSDP-specific sharded-state helpers (torch-native FSDP2)."""

from __future__ import annotations

import logging
from typing import List

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from unirl.train.backend.sharded_state import _maybe_dtensor_to_tensor

logger = logging.getLogger(__name__)


def clip_grad_norm(
    params: List[Parameter],
    max_norm: float,
) -> Tensor:
    """FSDP-safe gradient clipping."""
    try:
        result = torch.nn.utils.clip_grad_norm_(params, max_norm)
        return _maybe_dtensor_to_tensor(result)
    except RuntimeError as exc:
        msg = str(exc)
        fallback_triggers = (
            "No backend type associated with device type cpu",
            "mixed torch.Tensor and DTensor",
        )
        if not any(t in msg for t in fallback_triggers):
            raise
        logger.warning(
            "clip_grad_norm: standard path hit %r; falling back to explicit global-norm clipping.",
            msg.splitlines()[0] if msg else "<no message>",
        )
        return _global_clip_for_sharded_grads(params, max_norm)


def fsdp_offload(model: nn.Module) -> None:
    """Move FSDP-wrapped params + grads to CPU, leaving meta tensors untouched."""
    meta_names = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_names:
        logger.warning(
            "[META-PROBE] fsdp_offload skipping %d meta params (must be frozen aux only): %s%s",
            len(meta_names),
            meta_names[:24],
            " ..." if len(meta_names) > 24 else "",
        )
    model._apply(lambda t: t if t.is_meta else t.cpu())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    logger.debug("fsdp_offload: offloaded params/grads to CPU")


def fsdp_onload(model: nn.Module, device: torch.device) -> None:
    """Move FSDP-wrapped params + grads back to device, leaving meta untouched."""
    model._apply(lambda t: t if t.is_meta else t.to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.debug("fsdp_onload: onloaded params/grads to %s", device)


def _global_clip_for_sharded_grads(
    params: List[Parameter],
    max_grad_norm: float,
) -> Tensor:
    """Explicit global-norm gradient clipping for FSDP DTensor grads."""
    import torch.distributed as dist

    grads: list[Tensor] = []
    shard_group = None
    local_sq_sum = 0.0
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        local_grad = grad
        if hasattr(local_grad, "to_local") and callable(getattr(local_grad, "to_local")):
            if shard_group is None:
                mesh = getattr(local_grad, "device_mesh", None)
                placements = getattr(local_grad, "placements", ())
                shard_dims = [i for i, placement in enumerate(placements) if placement.is_shard()]
                if mesh is not None and shard_dims:
                    if len(shard_dims) != 1:
                        raise RuntimeError(
                            "clip_grad_norm fallback supports exactly one DTensor shard dimension; "
                            f"got placements={placements!r}"
                        )
                    shard_group = mesh.get_group(shard_dims[0])
            local_grad = local_grad.to_local()
        if not isinstance(local_grad, Tensor):
            continue
        local_sq_sum += float(torch.sum(local_grad.detach().float() ** 2).item())
        grads.append(grad)

    if not grads:
        return torch.tensor(0.0)

    reduce_device = torch.device("cpu")
    if torch.cuda.is_available():
        reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")

    total_sq = torch.tensor(local_sq_sum, device=reduce_device, dtype=torch.float32)
    if dist.is_available() and dist.is_initialized():
        if shard_group is None or dist.get_world_size(group=shard_group) > 1:
            dist.all_reduce(total_sq, op=dist.ReduceOp.SUM, group=shard_group)
    global_norm = float(torch.sqrt(total_sq).item())
    clip_coef = float(max_grad_norm) / (global_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return torch.tensor(global_norm, device=reduce_device, dtype=torch.float32)


__all__ = [
    "clip_grad_norm",
    "fsdp_offload",
    "fsdp_onload",
]
