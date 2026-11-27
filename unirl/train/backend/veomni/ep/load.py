"""EP-sharded expert weight loading for the VeOmni backend."""

from __future__ import annotations

import glob
import json
import os
from typing import Callable, Dict, Tuple

import torch
from torch import nn

from unirl.train.backend.veomni.ep.models import hi3, qwen3_moe
from unirl.train.backend.veomni.ep.placement import assign_local_block


def load_ep_experts_sliced(
    model: nn.Module,
    pretrained_path: str,
    is_expert_key: Callable[[str], bool],
) -> int:
    """Memory-optimal per-rank Safetensors EP expert loader (no Rank-0 full stack or broadcast)."""
    from safetensors import safe_open
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_size = int(ps.ep_size) if getattr(ps, "ep_enabled", False) else 1
    ep_rank = (
        int(ps.extra_parallel_rank("ep"))
        if (ep_size > 1 and hasattr(ps, "extra_parallel_rank"))
        else (int(ps.ep_rank) if ep_size > 1 else 0)
    )

    index_path = os.path.join(pretrained_path, "model.safetensors.index.json")
    single_path = os.path.join(pretrained_path, "model.safetensors")

    handles: Dict[str, object] = {}
    key_to_handle: Dict[str, object] = {}

    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f).get("weight_map", {})
        for key, fname in weight_map.items():
            shard_path = os.path.join(pretrained_path, fname)
            if shard_path not in handles:
                handles[shard_path] = safe_open(shard_path, framework="pt", device="cpu")
            key_to_handle[key] = handles[shard_path]
    elif os.path.isfile(single_path):
        handles[single_path] = safe_open(single_path, framework="pt", device="cpu")
        for key in handles[single_path].keys():
            key_to_handle[key] = handles[single_path]
    else:
        shards = sorted(glob.glob(os.path.join(pretrained_path, "*.safetensors")))
        if not shards:
            raise FileNotFoundError(f"load_ep_experts_sliced: no *.safetensors under {pretrained_path!r}")
        for shard_path in shards:
            handles[shard_path] = safe_open(shard_path, framework="pt", device="cpu")
            for key in handles[shard_path].keys():
                key_to_handle[key] = handles[shard_path]

    available_keys = set(key_to_handle)

    def get_tensor(key: str) -> torch.Tensor:
        return key_to_handle[key].get_tensor(key)

    n = 0
    for name, param in model.named_parameters():
        if not is_expert_key(name):
            continue
        expected_shape = tuple(param.shape)
        block = hi3.build_local_fused_block(
            fused_param_name=name,
            expected_shape=expected_shape,
            ep_rank=ep_rank,
            available_keys=available_keys,
            get_tensor=get_tensor,
        )
        if block is None:
            block = qwen3_moe.build_local_fused_block(
                fused_param_name=name,
                expected_shape=expected_shape,
                ep_rank=ep_rank,
                available_keys=available_keys,
                get_tensor=get_tensor,
            )
        if block is None:
            raise RuntimeError(
                f"load_ep_experts_sliced: could not rebuild local expert block for {name!r} "
                f"(expected_shape={expected_shape}, ep_rank={ep_rank}) from {pretrained_path!r}."
            )
        block = block.to(device=param.device, dtype=param.dtype)
        assign_local_block(param, block)
        del block
        n += 1
    return n


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

    # Map id(module) -> (local_storage_ptr, local_tensor_version, cached_full_parameter)
    full_cache: Dict[int, Tuple[int, int, nn.Parameter]] = {}
    sharded_cache: Dict[int, nn.Parameter] = {}  # id(module) -> sharded weight (mid-call only)

    def _pre(m, args):
        w = m._parameters.get("weight")
        if isinstance(w, DTensor):
            sharded_cache[id(m)] = w
            if w.requires_grad:
                # Differentiable full_tensor() keeps autograd connected to the DTensor shard;
                # store directly in _parameters so nn.Module.__getattr__ returns the live Tensor.
                full_cache.pop(id(m), None)
                m._parameters["weight"] = w.full_tensor()  # type: ignore[assignment]
            else:
                local_t = w._local_tensor
                version_key = (int(local_t.data_ptr()), int(local_t._version))
                cached = full_cache.get(id(m))
                if cached is None or (cached[0], cached[1]) != version_key:
                    full = nn.Parameter(w.full_tensor().detach(), requires_grad=False)
                    full_cache[id(m)] = (version_key[0], version_key[1], full)
                else:
                    full = cached[2]
                m._parameters["weight"] = full

    def _post(m, args, output):
        w = sharded_cache.pop(id(m), None)
        if w is not None:
            m._parameters["weight"] = w

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


__all__ = ["load_ep_experts", "load_ep_experts_sliced", "register_unsharded_param_hooks"]
