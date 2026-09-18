"""EP-sharded expert weight loading for the VeOmni backend."""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn

from unirl.train.backend.veomni.ep.placement import assign_local_block


def load_ep_experts(
    model: nn.Module,
    expert_state_dict: Dict[str, torch.Tensor],
    is_expert_key: Callable[[str], bool],
) -> int:
    """Load fused expert weights into EP-sharded DTensor params."""
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor
    from veomni.distributed.parallel_state import get_parallel_state

    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    ps = get_parallel_state()
    ep_rank = ps.extra_parallel_rank("ep")  # this rank's index in the EP group
    ep_size = ps.ep_size
    n = 0
    for name, param in model.named_parameters():
        if not is_expert_key(name):
            continue
        if not isinstance(param, DTensor):
            if rank0:
                param.data.copy_(expert_state_dict[name].to(device=param.device, dtype=param.dtype))
            n += 1
            continue
        local_experts = param.shape[0]
        block_shape = (local_experts, *param.shape[1:])
        full = expert_state_dict[name].to(device=param.device, dtype=param.dtype) if rank0 else None
        my_block = None
        for j in range(ep_size):
            block = (
                full[j * local_experts : (j + 1) * local_experts].contiguous()
                if rank0
                else torch.empty(block_shape, dtype=param.dtype, device=param.device)
            )
            if dist.is_initialized():
                dist.broadcast(block, src=0)
            if ep_rank == j:
                my_block = block
        assign_local_block(param, my_block)
        del full, my_block
        n += 1
    return n


def register_unsharded_param_hooks(model: nn.Module) -> Dict[str, int]:
    """All-gather sharded weights for root modules called OUTSIDE the FSDP forward."""
    from torch.distributed.tensor import DTensor

    full_cache: Dict[int, nn.Parameter] = {}  # id(module) -> full all-gathered weight
    sharded_cache: Dict[int, nn.Parameter] = {}  # id(module) -> sharded weight (mid-call only)

    def _pre(m, args):
        w = m.weight
        if isinstance(w, DTensor):
            full = full_cache.get(id(m))
            if full is None:
                full = nn.Parameter(w.full_tensor(), requires_grad=False)
                full_cache[id(m)] = full
            sharded_cache[id(m)] = w
            m.weight = full

    def _post(m, args, output):
        w = sharded_cache.pop(id(m), None)
        if w is not None:
            m.weight = w

    counts = {"wte": 0, "ln_f": 0, "lm_head": 0}
    for name, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(w, DTensor):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if isinstance(mod, nn.Embedding):
            kind = "wte"
        elif leaf == "ln_f":
            kind = "ln_f"
        elif leaf == "lm_head":
            kind = "lm_head"
        else:
            continue
        mod.register_forward_pre_hook(_pre)
        mod.register_forward_hook(_post)
        counts[kind] += 1
    return counts


__all__ = ["load_ep_experts", "register_unsharded_param_hooks"]
