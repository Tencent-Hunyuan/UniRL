"""Model-agnostic tensor placement helpers for VeOmni extra parallelism."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn


def has_ep_params(model: nn.Module) -> bool:
    """Whether ``model`` carries VeOmni's non-empty EP parameter group."""
    groups = getattr(model, "_extra_parallel_param_groups", None)
    return bool(groups and groups.get("ep"))


def ep_named_parameters(model: nn.Module) -> Tuple[Tuple[str, nn.Parameter], ...]:
    """Return EP parameters in deterministic model order."""
    groups = getattr(model, "_extra_parallel_param_groups", None) or {}
    ep_params = tuple(groups.get("ep", ()))
    ep_ids = {id(param) for param in ep_params}
    named = tuple((name, param) for name, param in model.named_parameters() if id(param) in ep_ids)
    if len(named) != len(ep_params):
        raise RuntimeError(
            "EP placement: could not map every _extra_parallel_param_groups['ep'] "
            f"parameter to an FQN ({len(named)}/{len(ep_params)} mapped)."
        )
    return named


def materialize_local_block(value: torch.Tensor) -> torch.Tensor:
    """Replicate a DTensor over its own mesh, preserving the outer EP block."""
    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:  # pragma: no cover - older torch
        from torch.distributed._tensor import DTensor, Replicate

    if isinstance(value, DTensor):
        work = value
        mesh_device = getattr(work.device_mesh, "device_type", None)
        if work.to_local().device.type == "cpu" and mesh_device != "cpu":
            if not torch.cuda.is_available():
                raise RuntimeError("EP placement: cannot gather a CPU DTensor over an NCCL mesh without CUDA.")
            work = work.cuda()
        placements = [Replicate()] * work.device_mesh.ndim
        return work.redistribute(placements=placements).to_local().detach()
    return value.detach()


def gather_stacked_expert_block(
    local_block: torch.Tensor,
    *,
    ep_size: int,
    ep_group,
) -> torch.Tensor:
    """Gather contiguous ``[E/ep,...]`` blocks and concatenate the expert axis."""
    block = local_block.contiguous()
    if block.device.type == "cpu" and dist.is_initialized() and dist.get_backend(ep_group) == "nccl":
        block = block.cuda()
    gathered = [torch.empty_like(block) for _ in range(ep_size)]
    dist.all_gather(gathered, block, group=ep_group)
    return torch.cat(gathered, dim=0)


def assign_local_block(param: torch.Tensor, block: torch.Tensor) -> None:
    """Copy a full local EP block into a plain tensor or its DTensor shard."""
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:  # pragma: no cover - older torch
        from torch.distributed._tensor import DTensor, distribute_tensor

    with torch.no_grad():
        if isinstance(param, DTensor):
            sharded = distribute_tensor(block, param.device_mesh, param.placements)
            param.to_local().copy_(sharded.to_local())
        else:
            param.copy_(block)


__all__ = [
    "has_ep_params",
    "ep_named_parameters",
    "materialize_local_block",
    "gather_stacked_expert_block",
    "assign_local_block",
]
