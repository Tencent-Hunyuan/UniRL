"""FSDP-specific sharded-state helpers (torch-native FSDP2).

The FSDP2-generic DCP state-dict helpers are re-exported from
:mod:`unirl.train.backend.sharded_state` (so this module's public surface is
unchanged); only what is specific to the torch-native ``fully_shard`` path lives
here: gradient clipping (with the explicit global-norm fallback for the
cpu_offload corner case) and the meta-skipping offload / onload.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

from unirl.train.backend.sharded_state import (
    StateDict,
    _current_rank,
    _maybe_dtensor_to_tensor,
    gather_optimizer_state_dict,
    gather_state_dict,
    infer_device,
    is_materialized,
    load_model_state_dict,
    load_optimizer_state_dict,
    local_view,
    lora_state_dict,
    move_optimizer_state,
    nft_state_dict,
    trainable_params,
)

logger = logging.getLogger(__name__)


def clip_grad_norm(
    params: List[Parameter],
    max_norm: float,
) -> Tensor:
    """FSDP-safe gradient clipping.

    Tries the standard ``torch.nn.utils.clip_grad_norm_`` first; falls
    back to an explicit global-norm path for known FSDP corner cases
    (mixed regular Tensor + DTensor, or CPU DTensor collectives missing
    under cpu_offload).
    """
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
    """Move FSDP-wrapped params + grads to CPU, leaving meta tensors untouched.

    The 80B meta-init path materializes only the trained decoder + heads (aux
    vae / vit stay on meta via ``with_aux=()``); a plain ``model.cpu()`` would
    raise ``Cannot copy out of meta tensor`` on those. ``_apply`` is what
    ``.cpu()`` delegates to (handles FSDP DTensor shards); skipping meta leaves
    the never-materialized aux alone. No-op difference for fully-materialized
    models (SD3).

    META-PROBE: logs exactly which params stay on meta so the "only frozen aux"
    assumption is verified, not assumed. If a TRAINED / forward-needed module
    (``model.layers.*`` / ``lm_head`` / ``patch_embed`` / ``time_embed`` / heads)
    appears here, materialize missed it and this guard would silently mask the
    bug (deferred meta error or silent-NaN at forward). Expected meta set: only
    ``vae.*`` / ``vision_model.*`` (intentionally never materialized)."""
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
    """Move FSDP-wrapped params + grads back to device, leaving meta untouched.

    Mirror of :func:`fsdp_offload` — never-materialized meta aux stays on meta
    (moving it to a device would raise; it carries no data to move)."""
    model._apply(lambda t: t if t.is_meta else t.to(device))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.debug("fsdp_onload: onloaded params/grads to %s", device)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _global_clip_for_sharded_grads(
    params: List[Parameter],
    max_grad_norm: float,
) -> Tensor:
    """Explicit global-norm gradient clipping for FSDP DTensor grads.

    Ported from the deleted FSDPPolicy._global_clip_for_sharded_grads.
    Handles the FSDP corner case the standard clip_grad_norm_ path can't:
    CPU DTensor collectives missing under cpu_offload. Every grad here is a
    sharded DTensor: the root wrap claims all leftover params, and
    ``fsdp_wrap`` fails fast on trainable params outside every group when
    ``root_wrap`` is disabled — so per-shard square sums SUM-reduce to the
    exact global norm with no replicated double counting.
    """
    import torch.distributed as dist

    grads: list[Tensor] = []
    local_sq_sum = 0.0
    for param in params:
        grad = getattr(param, "grad", None)
        if grad is None:
            continue
        local_grad = grad
        if hasattr(local_grad, "to_local") and callable(getattr(local_grad, "to_local")):
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
        dist.all_reduce(total_sq, op=dist.ReduceOp.SUM)
    global_norm = float(torch.sqrt(total_sq).item())
    clip_coef = float(max_grad_norm) / (global_norm + 1e-6)
    if clip_coef < 1.0:
        for grad in grads:
            grad.mul_(clip_coef)
    return torch.tensor(global_norm, device=reduce_device, dtype=torch.float32)


__all__ = [
    "StateDict",
    "clip_grad_norm",
    "gather_optimizer_state_dict",
    "gather_state_dict",
    "load_model_state_dict",
    "load_optimizer_state_dict",
    "move_optimizer_state",
    "local_view",
    "is_materialized",
    "trainable_params",
    "lora_state_dict",
    "nft_state_dict",
    "fsdp_offload",
    "fsdp_onload",
    "infer_device",
    "_current_rank",
]
