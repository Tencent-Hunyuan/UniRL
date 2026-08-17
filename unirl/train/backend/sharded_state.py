"""FSDP2-generic sharded-state helpers shared by every train backend."""

from __future__ import annotations

import logging
from typing import Dict, Iterator, Optional

import torch
from torch import Tensor, nn
from torch.nn.parameter import Parameter

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]

# ``checkpoint_wrapper`` (activation checkpointing) interposes this segment in
# ``named_parameters()`` names while ``state_dict()`` stays canonical (the
# wrapper's state-dict hooks strip it), so any name-keyed match between module
# params and checkpoint keys must compare canonical names.
_AC_WRAP_SEGMENT = "_checkpoint_wrapped_module."


def _canonical_param_name(name: str) -> str:
    """Strip activation-checkpoint wrapper segments from a parameter name."""
    return name.replace(_AC_WRAP_SEGMENT, "")


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


def load_model_state_dict(
    model: nn.Module,
    state_dict: StateDict,
    *,
    strict: bool = True,
    broadcast_from_rank0: bool = True,
) -> None:
    """Load a full state dict and reshard it into ``model``."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
        strict=strict,
    )
    try:
        set_model_state_dict(model, state_dict, options=options)
    except TypeError:
        set_model_state_dict(model, state_dict)


def gather_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Rank-0 DCP gather of optimizer state.  Full state on rank 0, empty on others."""
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    try:
        full = dict(get_optimizer_state_dict(model, optimizer, options=options))
    except TypeError:
        full = dict(get_optimizer_state_dict(model, optimizer))

    if _current_rank() != 0:
        return {}
    return full


def gather_lora_state_dict(model: nn.Module) -> StateDict:
    """Gather every adapter's LoRA tensors, preserving the model state-dict key format."""
    gathered: StateDict = {}
    for key, value in model.state_dict().items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        if isinstance(value, torch.Tensor) and value.is_meta:
            raise RuntimeError(f"gather_lora_state_dict: LoRA tensor {key!r} is still on meta")
        materialized = _materialize_checkpoint_tensor(value)
        if isinstance(materialized, torch.Tensor):
            materialized = materialized.detach().cpu()
        if _current_rank() == 0:
            gathered[key] = materialized
    if _current_rank() != 0:
        return {}
    return gathered


def load_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: StateDict,
    *,
    broadcast_from_rank0: bool = True,
) -> None:
    """Load a full optimizer state dict and reshard it into ``optimizer``."""
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
    )
    try:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=options)
    except TypeError:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict)


def sharded_model_state_dict(model: nn.Module) -> StateDict:
    """Per-rank sharded model state for DCP."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    options = _build_state_dict_options(full_state_dict=False)
    try:
        return dict(get_model_state_dict(model, options=options))
    except TypeError:
        return dict(get_model_state_dict(model))


def sharded_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Per-rank sharded optimizer state for DCP (symmetric with"""
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    options = _build_state_dict_options(full_state_dict=False)
    try:
        return dict(get_optimizer_state_dict(model, optimizer, options=options))
    except TypeError:
        return dict(get_optimizer_state_dict(model, optimizer))


def load_sharded_model_state_dict(model: nn.Module, state_dict: StateDict, *, strict: bool = True) -> None:
    """Load a per-rank sharded model state read by ``dcp.load`` in place."""
    from torch.distributed.checkpoint.state_dict import set_model_state_dict

    options = _build_state_dict_options(full_state_dict=False, strict=strict)
    try:
        set_model_state_dict(model, state_dict, options=options)
    except TypeError:
        set_model_state_dict(model, state_dict)


def load_sharded_optimizer_state_dict(
    model: nn.Module, optimizer: torch.optim.Optimizer, state_dict: StateDict
) -> None:
    """Load a per-rank sharded optimizer state read by ``dcp.load`` in place."""
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    options = _build_state_dict_options(full_state_dict=False)
    try:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=options)
    except TypeError:
        set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict)


def drop_meta_entries(state_dict: StateDict) -> StateDict:
    """Drop never-materialized (meta) entries from a sharded state dict."""
    kept: StateDict = {}
    for key, value in state_dict.items():
        local = getattr(value, "_local_tensor", value)
        if isinstance(local, torch.Tensor) and local.is_meta:
            continue
        kept[key] = value
    return kept


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: object) -> None:
    """Move every tensor in the optimizer state to ``device`` (the on/offload loop)."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def lora_state_dict(
    model: nn.Module,
    full_sd: Optional[StateDict] = None,
) -> StateDict:
    """Adapter-only state for inference export."""
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
    """Export the shadow ('old') adapter state for DiffusionNFT checkpoint."""
    if full_sd is None:
        full_sd = gather_state_dict(model)
    if _current_rank() != 0:
        return {}
    token = f".{shadow_adapter}."
    return {k: v for k, v in full_sd.items() if ("lora_A" in k or "lora_B" in k) and token in k}


def local_view(tensor: Tensor) -> Tensor:
    """DTensor -> local shard.  Identity for non-DTensors."""
    if hasattr(tensor, "_local_tensor"):
        return tensor._local_tensor
    return tensor


def is_materialized(model: nn.Module) -> bool:
    return not any(p.is_meta for p in model.parameters())


def trainable_params(model: nn.Module) -> Iterator[Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


def infer_device(model: nn.Module) -> torch.device:
    """First non-meta parameter's device, else current cuda, else cpu."""
    for param in model.parameters():
        if param.is_meta:
            continue
        return param.device
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def _is_lora_key(key: str) -> bool:
    """True for default-adapter LoRA keys (excludes shadow/old adapter)."""
    return ("lora_A" in key or "lora_B" in key) and ".old." not in key


def _build_state_dict_options(**kwargs: object) -> object:
    """Construct ``StateDictOptions`` degrading gracefully on older torch."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    candidates = [
        dict(kwargs),
        {k: v for k, v in kwargs.items() if k != "strict"},
        {k: v for k, v in kwargs.items() if k not in {"strict", "broadcast_from_rank0"}},
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
        try:
            return value.full_tensor()
        except Exception:
            return value
    return value


def _materialize_checkpoint_tensor(value: object) -> object:
    if hasattr(value, "full_tensor") and callable(getattr(value, "full_tensor")):
        device = getattr(value, "device", None)
        if (
            getattr(device, "type", None) == "cpu"
            and torch.cuda.is_available()
            and hasattr(value, "cuda")
            and callable(getattr(value, "cuda"))
        ):
            value = value.cuda()
    return _maybe_dtensor_to_tensor(value)


def _to_cpu_state_dict(state_dict: StateDict) -> StateDict:
    converted: StateDict = {}
    for key, value in state_dict.items():
        tensor_or_obj = _maybe_dtensor_to_tensor(value)
        if isinstance(tensor_or_obj, torch.Tensor):
            converted[key] = tensor_or_obj.detach().cpu()
        else:
            converted[key] = tensor_or_obj
    return converted


__all__ = [
    "StateDict",
    "gather_state_dict",
    "load_model_state_dict",
    "gather_optimizer_state_dict",
    "load_optimizer_state_dict",
    "sharded_model_state_dict",
    "sharded_optimizer_state_dict",
    "load_sharded_model_state_dict",
    "load_sharded_optimizer_state_dict",
    "drop_meta_entries",
    "move_optimizer_state",
    "lora_state_dict",
    "nft_state_dict",
    "local_view",
    "is_materialized",
    "trainable_params",
    "infer_device",
]
