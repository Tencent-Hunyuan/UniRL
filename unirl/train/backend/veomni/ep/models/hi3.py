"""HunyuanImage 3.0 expert-parallel (EP) wiring for the VeOmni backend."""

from __future__ import annotations

import re
from typing import Dict, Optional

import torch
from torch import nn

_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_and_up_proj|down_proj)\.weight$")

_EP_PLAN = {
    "layers.*.mlp.experts.gate_and_up_proj": 0,
    "layers.*.mlp.experts.down_proj": 0,
}


def _swap_gate_up_halves(gate_and_up: torch.Tensor) -> torch.Tensor:
    """Swap the two output halves of a fused gate_and_up tensor ``[..., 2I, H]``."""
    half = gate_and_up.shape[-2] // 2
    return torch.cat([gate_and_up[..., half:, :], gate_and_up[..., :half, :]], dim=-2)


def fuse_expert_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Stack per-expert weights into fused ``[E, ...]`` tensors (load-time converter)."""
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
        stacked = torch.stack([per_idx[j] for j in indices], dim=0)
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
    """Holds the fused expert weights so their FQN matches what :data:`_EP_PLAN` and the converter target."""

    def __init__(self, num_experts: int, hidden: int, inter: int, dtype, device):
        super().__init__()
        self.gate_and_up_proj = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden, dtype=dtype, device=device))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter, dtype=dtype, device=device))


class FusedHunyuanMoE(nn.Module):
    """EP drop-in for HI3's ``HunyuanMoE``: fused experts via veomni grouped GEMM + all_to_all."""

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
    """A fused-expert param key (``*.experts.gate_and_up_proj`` / ``.down_proj``)."""
    return key.endswith((".experts.gate_and_up_proj", ".experts.down_proj"))


def replace_hunyuan_moe_with_fused(decoder: nn.Module) -> int:
    """In-place: swap every ``HunyuanMoE`` mlp for a :class:`FusedHunyuanMoE` and attach ``get_parallel_plan``."""
    n = 0
    for layer in getattr(decoder, "layers", []):
        mlp = getattr(layer, "mlp", None)
        if type(mlp).__name__ != "HunyuanMoE":
            continue
        w = mlp.experts[0].gate_and_up_proj.weight
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
