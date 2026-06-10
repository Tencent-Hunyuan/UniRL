from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]


def gather_state_dict(model: nn.Module) -> StateDict:
    """Rank-0 DCP gather.  Returns full state on rank 0, empty on others."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    try:
        full = dict(get_model_state_dict(model, options=options))
    except TypeError:
        full = dict(get_model_state_dict(model))

    if _current_rank() != 0:
        return {}
    return _to_cpu_state_dict(full)


def load_model_state_dict(model: nn.Module, state_dict: StateDict) -> None:
    """Load a full state dict, broadcasting from rank 0 across ranks."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=True,
        cpu_offload=False,
    )
    try:
        set_model_state_dict(model, state_dict, options=options)
    except TypeError:
        set_model_state_dict(model, state_dict)


def local_view(tensor: Tensor) -> Tensor:
    """DTensor -> local shard.  Identity for non-DTensors."""
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    return tensor


def is_materialized(model: nn.Module) -> bool:
    return not any(p.is_meta for p in model.parameters())


def trainable_params(model: nn.Module) -> Iterator[Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def lora_state_dict(
    model: nn.Module,
    full_sd: Optional[StateDict] = None,
) -> StateDict:
    """Adapter-only state for inference export.

    All ranks must call this (the DCP gather is a collective).  Returns
    the filtered dict on rank 0, empty dict on other ranks.
    """
    if full_sd is None:
        full_sd = gather_state_dict(model)
    if _current_rank() != 0:
        return {}
    return {k: v for k, v in full_sd.items() if _is_lora_key(k)}


def nft_state_dict(
    model: nn.Module,
    full_sd: Optional[StateDict] = None,
    shadow_adapter: str = "old",
) -> StateDict:
    """Export the shadow ('old') adapter state for DiffusionNFT checkpoint.

    All ranks must call this (the DCP gather is a collective).  Returns
    the filtered dict on rank 0, empty dict on other ranks.
    """
    if full_sd is None:
        full_sd = gather_state_dict(model)
    if _current_rank() != 0:
        return {}
    token = f".{shadow_adapter}."
    return {k: v for k, v in full_sd.items() if ("lora_A" in k or "lora_B" in k) and token in k}


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


def infer_device(model: nn.Module) -> torch.device:
    """First non-meta parameter's device, else current cuda, else cpu."""
    for param in model.parameters():
        if param.is_meta:
            continue
        return param.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _is_lora_key(key: str) -> bool:
    """True for default-adapter LoRA keys (excludes shadow/old adapter)."""
    return ("lora_A" in key or "lora_B" in key) and ".old." not in key


def _build_state_dict_options(**kwargs: object) -> object:
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    candidates = [
        dict(kwargs),
        {k: v for k, v in kwargs.items() if k != "broadcast_from_rank0"},
        {k: v for k, v in kwargs.items() if k in {"full_state_dict", "cpu_offload"}},
        {},
    ]
    for candidate in candidates:
        try:
            return StateDictOptions(**candidate)
        except TypeError:
            continue
    return StateDictOptions()


def _maybe_dtensor_to_tensor(value: object) -> object:
    if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
        return value.full_tensor()
    return value


def _to_cpu_state_dict(state_dict: StateDict) -> StateDict:
    converted: StateDict = {}
    for key, value in state_dict.items():
        tensor_or_obj = _maybe_dtensor_to_tensor(value)
        if isinstance(tensor_or_obj, torch.Tensor):
            converted[key] = tensor_or_obj.detach().cpu()
        else:
            converted[key] = tensor_or_obj
    return converted


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
    "gather_state_dict",
    "load_model_state_dict",
    "local_view",
    "is_materialized",
    "trainable_params",
    "lora_state_dict",
    "nft_state_dict",
    "fsdp_offload",
    "fsdp_onload",
    "infer_device",
]
