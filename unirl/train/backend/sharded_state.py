"""FSDP2-generic sharded-state helpers shared by every train backend."""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Optional

import torch
from torch import nn
from torch.nn.parameter import Parameter

from unirl.distributed.local import local_view

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
    """Rank-0 DCP optimizer gather (empty on other ranks); cold AdamW stays unstepped. See ``../readme.md`` Gotchas."""
    options = _build_state_dict_options(full_state_dict=True, cpu_offload=True)
    return _export_optimizer_state_dict(model, optimizer, options=options, rank0_only=True)


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
    """Load a full optimizer state dict; a cold checkpoint leaves AdamW unstepped. See ``../readme.md`` Gotchas."""
    options = _build_state_dict_options(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
    )
    _apply_optimizer_state_dict(model, optimizer, state_dict, options=options)


def sharded_model_state_dict(model: nn.Module) -> StateDict:
    """Per-rank sharded model state for DCP."""
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    options = _build_state_dict_options(full_state_dict=False)
    try:
        return dict(get_model_state_dict(model, options=options))
    except TypeError:
        return dict(get_model_state_dict(model))


def sharded_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Per-rank sharded optimizer state for DCP; cold AdamW stays unstepped. See ``../readme.md`` Gotchas."""
    options = _build_state_dict_options(full_state_dict=False)
    return _export_optimizer_state_dict(model, optimizer, options=options, rank0_only=False)


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
    """Load sharded optimizer state; a cold checkpoint leaves AdamW unstepped. See ``../readme.md`` Gotchas."""
    options = _build_state_dict_options(full_state_dict=False)
    _apply_optimizer_state_dict(model, optimizer, state_dict, options=options)


def drop_meta_entries(state_dict: StateDict) -> StateDict:
    """Drop never-materialized (meta) entries from a sharded state dict."""
    kept: StateDict = {}
    for key, value in state_dict.items():
        local = local_view(value) if isinstance(value, torch.Tensor) else value
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


def _is_cold_optimizer(optimizer: torch.optim.Optimizer) -> bool:
    if optimizer.state:
        return False
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                return False
    return True


def _optimizer_state_entries(state_dict: StateDict) -> Iterator[Dict[str, object]]:
    state = state_dict.get("state")
    if not isinstance(state, dict):
        return
    for entry in state.values():
        if isinstance(entry, dict) and "step" in entry:
            yield entry


def _zero_optimizer_steps(state_dict: StateDict) -> None:
    for entry in _optimizer_state_entries(state_dict):
        step = entry["step"]
        entry["step"] = torch.zeros_like(step) if isinstance(step, torch.Tensor) else 0


def _optimizer_state_dict_is_cold(state_dict: StateDict) -> bool:
    if not state_dict.get("state"):
        return True
    entries = list(_optimizer_state_entries(state_dict))
    if not entries:
        return False
    for entry in entries:
        step = entry["step"]
        if isinstance(step, torch.Tensor):
            if bool(local_view(step).detach().cpu().any()):
                return False
        elif step != 0:
            return False
    return True


def _restore_cold_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    step_count: Optional[int],
    lrs: List[object],
) -> None:
    optimizer.state.clear()
    for group, lr in zip(optimizer.param_groups, lrs):
        if "lr" in group:
            group["lr"] = lr
    if step_count is not None:
        optimizer._step_count = step_count
    optimizer.zero_grad(set_to_none=True)


def _export_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    options: object,
    rank0_only: bool,
) -> StateDict:
    """Call ``get_optimizer_state_dict`` without advancing a cold AdamW clock. See ``../readme.md`` Gotchas."""
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

    cold = _is_cold_optimizer(optimizer)
    step_count = getattr(optimizer, "_step_count", None)
    lrs = [group.get("lr") for group in optimizer.param_groups]
    try:
        try:
            exported = dict(get_optimizer_state_dict(model, optimizer, options=options))
        except TypeError:
            exported = dict(get_optimizer_state_dict(model, optimizer))
        if cold:
            _zero_optimizer_steps(exported)
        if rank0_only and _current_rank() != 0:
            return {}
        return exported
    finally:
        if cold:
            _restore_cold_optimizer(optimizer, step_count=step_count, lrs=lrs)


def _apply_optimizer_state_dict(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state_dict: StateDict,
    *,
    options: object,
) -> None:
    """Call ``set_optimizer_state_dict`` without leaving a dummy step on a cold AdamW. See ``../readme.md`` Gotchas."""
    from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict

    was_cold = _is_cold_optimizer(optimizer)
    incoming_cold = _optimizer_state_dict_is_cold(state_dict)
    step_count = getattr(optimizer, "_step_count", None)
    lrs = [group.get("lr") for group in optimizer.param_groups]
    applied = False
    try:
        try:
            set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=options)
        except TypeError:
            set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict)
        applied = True
    finally:
        if was_cold and (not applied or incoming_cold):
            _restore_cold_optimizer(optimizer, step_count=step_count, lrs=lrs)


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
    "is_materialized",
    "trainable_params",
    "infer_device",
]
