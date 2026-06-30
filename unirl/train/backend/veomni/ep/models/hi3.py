"""HunyuanImage 3.0 expert-parallel (EP) wiring for the VeOmni backend.

HI3 ships its 64 experts as an ``nn.ModuleList`` of per-expert ``HunyuanMLP``
(``...mlp.experts.{j}.gate_and_up_proj.weight`` / ``...down_proj.weight``).
VeOmni's EP shards a *fused* expert tensor whose dim-0 is the expert axis via
``Shard(0)`` — so the per-expert weights are stacked into
``...mlp.experts.gate_and_up_proj`` ``[E, 2I, H]`` and
``...mlp.experts.down_proj`` ``[E, H, I]`` (I = moe_intermediate_size).

Pieces (all GPU-verified — see jobs/hi3_ep/):

* :class:`FusedHunyuanMoE` — drop-in replacement for ``HunyuanMoE`` whose expert
  compute routes through veomni's ``group_gemm_fused_moe_forward`` (Triton
  grouped GEMM + all_to_all dispatch under EP). Reuses the original router
  (``gate``) and shared expert (``shared_mlp``) — neither is expert-parallel.
* :func:`fuse_expert_state_dict` — checkpoint converter (per-expert -> fused),
  applied at load time.
* :func:`get_hi3_parallel_plan` — the ``ParallelPlan`` naming the fused tensors
  with ``Shard(0)``; ``VeOmniBackend`` asserts ``get_parallel_plan()`` exists
  when ``ep_size > 1``.

The HALF-SWAP (critical, verified): HI3 computes ``x1 * silu(x2)`` (silu on the
SECOND half of gate_and_up); veomni's merged-fc1 computes ``silu(first)*second``
(silu on the FIRST half). So gate_and_up's two halves are swapped when producing
the fused weight — without it, outputs diverge (~0.22 rel). down_proj is unchanged.

The generic EP-sharded weight load lives in :func:`unirl.train.backend.veomni.ep.load_ep_experts`;
this module supplies the HI3 expert-key predicate (:func:`is_fused_expert_key`)
and the per-expert -> fused converter it consumes.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import torch
from torch import nn

# Per-expert weight key: <prefix>.experts.{idx}.{gate_and_up_proj|down_proj}.weight
_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_and_up_proj|down_proj)\.weight$")

# FQN globs relative to the module VeOmniBackend wraps (the bare decoder,
# transformer.model) -> ``layers.*`` (NOT ``model.layers.*``).
_EP_PLAN = {
    "layers.*.mlp.experts.gate_and_up_proj": 0,
    "layers.*.mlp.experts.down_proj": 0,
}


def _swap_gate_up_halves(gate_and_up: torch.Tensor) -> torch.Tensor:
    """Swap the two output halves of a fused gate_and_up tensor ``[..., 2I, H]``.

    HI3 (silu on 2nd half) -> veomni merged-fc1 (silu on 1st half). Verified
    necessary: jobs/hi3_ep/stage2a_level2_kernel.py.
    """
    half = gate_and_up.shape[-2] // 2
    return torch.cat([gate_and_up[..., half:, :], gate_and_up[..., :half, :]], dim=-2)


def fuse_expert_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack per-expert weights into fused ``[E, ...]`` tensors (load-time converter).

    ``<prefix>.experts.{j}.{proj}.weight`` -> ``<prefix>.experts.{proj}`` of shape
    ``[E, *weight.shape]`` (experts stacked in index order). ``gate_and_up_proj`` is
    additionally half-swapped to veomni's silu-first convention; ``down_proj`` keeps
    its layout. Non-expert keys (router, shared_mlp, attention, norms) pass through.
    """
    groups: Dict[tuple, Dict[int, torch.Tensor]] = {}
    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        m = _EXPERT_RE.match(key)
        if m is None:
            out[key] = value
            continue
        groups.setdefault((m["prefix"], m["proj"]), {})[int(m["idx"])] = value

    for (prefix, proj), per_idx in groups.items():
        indices = sorted(per_idx)
        if indices != list(range(len(indices))):
            raise ValueError(
                f"fuse_expert_state_dict: non-contiguous experts for {prefix}.{proj}: "
                f"got {indices[:8]}{'...' if len(indices) > 8 else ''}"
            )
        stacked = torch.stack([per_idx[j] for j in indices], dim=0)  # [E, *wshape]
        if proj == "gate_and_up_proj":
            stacked = _swap_gate_up_halves(stacked).contiguous()
        out[f"{prefix}.{proj}"] = stacked
    return out


def get_hi3_parallel_plan():
    """Return the VeOmni ``ParallelPlan`` for HI3 expert parallelism (Shard(0))."""
    from torch.distributed._tensor import Shard
    from veomni.distributed.parallel_plan import ParallelPlan

    ep_plan = {fqn: Shard(dim) for fqn, dim in _EP_PLAN.items()}
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


class FusedExperts(nn.Module):
    """Holds the fused expert weights so their FQN is ``experts.gate_and_up_proj``
    / ``experts.down_proj`` (what :data:`_EP_PLAN` + the converter target)."""

    def __init__(self, num_experts: int, hidden: int, inter: int, dtype, device):
        super().__init__()
        self.gate_and_up_proj = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden, dtype=dtype, device=device))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter, dtype=dtype, device=device))


class FusedHunyuanMoE(nn.Module):
    """EP drop-in for HI3's ``HunyuanMoE``: fused experts via veomni grouped GEMM
    + all_to_all; original router + shared expert reused verbatim.

    Routing matches HI3's training path (dropless ``easy_topk``). Requires
    ``veomni.distributed.parallel_state`` initialized (ep_size>=1) — the backend
    does this before wrapping.
    """

    def __init__(
        self,
        gate: nn.Module,
        shared_mlp: Optional[nn.Module],
        num_experts: int,
        hidden: int,
        inter: int,
        dtype,
        device,
    ):
        super().__init__()
        self.gate = gate
        self.shared_mlp = shared_mlp
        self.num_experts = num_experts
        self.experts = FusedExperts(num_experts, hidden, inter, dtype, device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from torch.distributed.tensor import DTensor
        from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward

        bsz, seq, hidden = hidden_states.shape
        shared = self.shared_mlp(hidden_states) if self.shared_mlp is not None else None
        topk_weights, topk_idx = self.gate(hidden_states, topk_impl="easy")
        topk_weights = topk_weights.to(hidden_states.dtype)
        # The EP-sharded expert params are DTensors (each rank's local experts).
        # The Triton grouped-GEMM kernel needs raw local tensors, not DTensors.
        gate_up = self.experts.gate_and_up_proj
        down = self.experts.down_proj
        if isinstance(gate_up, DTensor):
            gate_up = gate_up.to_local()
        if isinstance(down, DTensor):
            down = down.to_local()
        y = group_gemm_fused_moe_forward(
            num_experts=self.num_experts,
            routing_weights=topk_weights,
            selected_experts=topk_idx,
            hidden_states=hidden_states.reshape(-1, hidden).contiguous(),
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=down,
            fc1_1_2_weight=gate_up,
        ).view(bsz, seq, hidden)
        return y + shared if shared is not None else y


def is_fused_expert_key(key: str) -> bool:
    """A fused-expert param key (``*.experts.gate_and_up_proj`` / ``.down_proj``).

    HI3-specific naming; passed to the generic
    ``unirl.train.backend.veomni.ep.load_ep_experts`` as its expert selector.
    """
    return key.endswith((".experts.gate_and_up_proj", ".experts.down_proj"))


def replace_hunyuan_moe_with_fused(decoder: nn.Module) -> int:
    """In-place: swap every ``HunyuanMoE`` ``mlp`` in ``decoder.layers`` for a
    :class:`FusedHunyuanMoE`, and attach ``get_parallel_plan`` to ``decoder``.

    Meta-safe — builds the fused module from the original's dims/dtype/device
    (read off ``experts[0]`` without materializing), adopting the original
    ``gate`` / ``shared_mlp`` submodules so their checkpoint keys are unchanged.
    Run on the meta model BEFORE ``veomni_parallelize`` so the fused expert
    params (not the per-expert ModuleList) are what FSDP/EP shards. Returns the
    number of layers swapped.
    """
    n = 0
    for layer in getattr(decoder, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if type(mlp).__name__ != "HunyuanMoE":
            continue
        w = mlp.experts[0].gate_and_up_proj.weight  # [2I, H]
        two_i, hidden = w.shape
        layer.mlp = FusedHunyuanMoE(
            gate=mlp.gate,
            shared_mlp=getattr(mlp, "shared_mlp", None),
            num_experts=mlp.num_experts,
            hidden=hidden,
            inter=two_i // 2,
            dtype=w.dtype,
            device=w.device,
        )
        n += 1
    decoder.get_parallel_plan = get_hi3_parallel_plan  # type: ignore[attr-defined]
    return n


__all__ = [
    "fuse_expert_state_dict",
    "get_hi3_parallel_plan",
    "FusedHunyuanMoE",
    "FusedExperts",
    "replace_hunyuan_moe_with_fused",
    "is_fused_expert_key",
]
