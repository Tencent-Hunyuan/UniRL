"""EP-sharded expert weight loading for the VeOmni backend.

Why this can't go through DCP: VeOmni shards a fused ``[E,...]`` expert tensor so
each rank's param dim-0 is ALREADY its local experts (``E/ep`` — the EP split sits
*outside* the DTensor; the DTensor only carries the ``ep_fsdp`` shard on dim 1).
``set_model_state_dict``'s broadcast then copies the full ``[E,...]`` straight onto
the local ``[E/ep,...]`` shard and fails (size ``E`` vs ``E/ep``). So load here:
broadcast rank-0's full tensor, slice this rank's expert block by ``ep_rank``, and
re-shard that block with the param's own mesh/placement.

Model-agnostic: any model whose fused expert tensors are EP-sharded loads them the
same way. The per-model bits — which params are experts, and the per-expert ->
fused conversion — stay in the model package (:mod:`.models`) and are passed in
(the ``is_expert_key`` predicate + the already-fused ``expert_state_dict``).
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
from torch import nn


def load_ep_experts(
    model: nn.Module,
    expert_state_dict: Dict[str, torch.Tensor],
    is_expert_key: Callable[[str], bool],
) -> int:
    """Load fused expert weights into EP-sharded DTensor params.

    ``expert_state_dict`` is the rank-0 full dict (``{}`` on other ranks);
    ``is_expert_key`` selects which of ``model``'s params are EP experts (the
    model package owns the naming). Returns the number of expert params loaded.
    """
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor, distribute_tensor
    from veomni.distributed.parallel_state import get_parallel_state

    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    ps = get_parallel_state()
    ep_rank = ps.extra_parallel_rank("ep")  # this rank's index in the EP group
    ep_size = ps.ep_size
    n = 0
    for name, param in model.named_parameters():
        if not is_expert_key(name):
            continue
        if not isinstance(param, DTensor):  # ep_size==1: plain replicated param
            if rank0:
                param.data.copy_(expert_state_dict[name].to(device=param.device, dtype=param.dtype))
            n += 1
            continue
        # The param's dim-0 is ALREADY this rank's local experts (E/ep). Send each
        # ep_rank's [E/ep,...] block separately (one broadcast per EP group) and keep
        # only the block this rank owns, then re-shard it with the param's own
        # mesh/placement (handles the dim-1 ep_fsdp shard). Per-block (not full-[E,...])
        # so every rank's transient stays [E/ep,...] — the EP memory saving must hold
        # at LOAD time too, else an 80B model OOMs here even when the sharded steady
        # state fits. Broadcast-only (no scatter) to work on every NCCL build.
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
        sharded = distribute_tensor(my_block, param.device_mesh, param.placements)
        param.to_local().copy_(sharded.to_local())
        del full, my_block, sharded
        n += 1
    return n


def register_unsharded_param_hooks(model: nn.Module) -> Dict[str, int]:
    """All-gather sharded weights for root modules called OUTSIDE the FSDP forward.

    VeOmni's root ``fully_shard`` turns the model's root-level params — the token
    embedding (``wte``), the final norm (``ln_f``) and the output head
    (``lm_head``) — into DTensors. Only the wrapped decoder *layers* get their own
    FSDP forward hook that all-gathers their params; these root params are not in
    any layer, so when HI3's ``ForCausalMM`` wrapper calls them directly (build
    ``inputs_embeds = wte(input_ids)`` for the ViT scatter; ``ln_f``/``lm_head``
    to read out logits) the weight is still a sharded DTensor while the activation
    is a plain tensor → ``aten.* got mixed Tensor and DTensor``.

    Register forward pre/post hooks on exactly those three (wte, ln_f, lm_head)
    that swap the DTensor weight for its full all-gathered tensor for the call's
    duration (cached once — all three are frozen under LoRA), then restore it. A
    no-op when the weight is already a plain tensor (so it never double-gathers).
    Returns a per-kind count of the modules hooked.

    The full and sharded weights are cached OFF the module (in closure dicts keyed
    by ``id(module)``), not as module attributes: assigning an ``nn.Parameter`` to
    a module silently registers it, so a ``_full_w`` attribute would leak a full,
    unsharded duplicate of wte/ln_f/lm_head into ``named_parameters()`` /
    ``state_dict()`` (polluting checkpoint + weight-sync enumeration). ``m.weight``
    itself is still swapped (FSDP's forward reads ``self.weight``), but only for
    the call's duration; the persistent ``_parameters['weight']`` is always the
    sharded DTensor.
    """
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
        elif leaf == "ln_f":  # the model's final RMSNorm (per-layer norms unshard via FSDP)
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
