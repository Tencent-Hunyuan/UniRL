"""FSDP2-generic sharded-state helpers shared by every train backend."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.parameter import Parameter

from unirl.distributed.local import local_view

logger = logging.getLogger(__name__)

StateDict = Dict[str, object]


def _current_rank() -> int:

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


def gather_state_dict(model: nn.Module) -> StateDict:
    """Rank-0 DCP gather.  Returns full state on rank 0, empty on others."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full = dict(get_model_state_dict(model, options=options))

    if _current_rank() != 0:
        return {}
    converted: StateDict = {}
    for key, value in full.items():
        if isinstance(value, DTensor):
            value = value.full_tensor()
        converted[key] = value.detach().cpu() if isinstance(value, torch.Tensor) else value
    return converted


def load_model_state_dict(
    model: nn.Module,
    state_dict: StateDict,
    *,
    strict: bool = True,
    broadcast_from_rank0: bool = True,
) -> object:
    """Load a full state dict and reshard it into ``model``; returns torch's ``(missing_keys, unexpected_keys)``."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
        strict=strict,
    )
    return set_model_state_dict(model, state_dict, options=options)


def gather_optimizer_state_dict(model: nn.Module, optimizer: torch.optim.Optimizer) -> StateDict:
    """Rank-0 DCP optimizer gather; never-updated AdamW params export as step-0 zero state."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    with fresh_adamw_state(optimizer):
        full = get_optimizer_state_dict(model, optimizer, options=options)
    return full if _current_rank() == 0 else {}


def gather_lora_state_dict(model: nn.Module, keep: Callable[[str], bool]) -> StateDict:
    """Gather the LoRA tensors whose keys pass ``keep``, preserving the model state-dict key format."""
    gathered: StateDict = {}
    for key, value in model.state_dict().items():
        if not keep(key):
            continue
        if isinstance(value, torch.Tensor) and value.is_meta:
            raise RuntimeError(f"gather_lora_state_dict: LoRA tensor {key!r} is still on meta")
        if isinstance(value, DTensor):
            if value.device.type == "cpu" and torch.cuda.is_available():
                value = value.cuda()
            value = value.full_tensor()
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
        if _current_rank() == 0:
            gathered[key] = value
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
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_optimizer_state_dict

    options = StateDictOptions(
        full_state_dict=True,
        broadcast_from_rank0=broadcast_from_rank0,
        cpu_offload=False,
    )
    if isinstance(optimizer, torch.optim.AdamW) and "state" in state_dict:
        # Sparse checkpoints list never-updated params without state; a step-0 entry is dropped back to lazy init.
        for group in state_dict["param_groups"]:
            for name in group["params"]:
                state_dict["state"].setdefault(name, {"step": torch.tensor(0.0)})
    set_optimizer_state_dict(model, optimizer, optim_state_dict=state_dict, options=options)
    if isinstance(optimizer, torch.optim.AdamW):
        for param in [p for p, entry in optimizer.state.items() if float(entry["step"]) == 0]:
            del optimizer.state[param]


@contextmanager
def fresh_adamw_state(optimizer: torch.optim.Optimizer) -> Iterator[None]:
    """Temporarily give never-updated AdamW params the step-0 zero state their first update would create."""
    added: List[Parameter] = []
    if isinstance(optimizer, torch.optim.AdamW):
        for group in optimizer.param_groups:
            moments = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq") if group["amsgrad"] else ("exp_avg", "exp_avg_sq")
            for param in group["params"]:
                if param not in optimizer.state:
                    optimizer.state[param] = {
                        "step": torch.tensor(0.0),
                        **{k: torch.zeros_like(param) for k in moments},
                    }
                    added.append(param)
    try:
        yield
    finally:
        for param in added:
            del optimizer.state[param]


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
    """Move optimizer state to ``device``; non-fused/capturable ``step`` stays on CPU as it is read via ``.item()``."""
    for group in optimizer.param_groups:
        step_device = device if group.get("capturable", False) or group.get("fused", False) else "cpu"
        for param in group["params"]:
            state = optimizer.state.get(param, {})
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(step_device if k == "step" else device)


def trainable_params(model: nn.Module) -> Iterator[Parameter]:
    return (p for p in model.parameters() if p.requires_grad)


__all__ = [
    "StateDict",
    "gather_state_dict",
    "load_model_state_dict",
    "gather_optimizer_state_dict",
    "gather_lora_state_dict",
    "load_optimizer_state_dict",
    "fresh_adamw_state",
    "drop_meta_entries",
    "move_optimizer_state",
    "trainable_params",
]
