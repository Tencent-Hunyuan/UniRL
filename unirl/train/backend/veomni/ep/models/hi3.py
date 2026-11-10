"""HunyuanImage 3.0 expert-parallel (EP) wiring for the VeOmni backend."""

from __future__ import annotations

import re
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from unirl.train.backend.veomni.ep.router_replay import get_active_replay, maybe_replay_indices

_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_and_up_proj|down_proj)\.weight$")

_EP_PLAN = {
    "layers.*.mlp.experts.gate_and_up_proj": 0,
    "layers.*.mlp.experts.down_proj": 0,
}

_SHARED_EXPERT_STREAMS: Dict[int, torch.cuda.Stream] = {}


def _get_shared_expert_stream(device: torch.device) -> torch.cuda.Stream:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    stream = _SHARED_EXPERT_STREAMS.get(idx)
    if stream is None:
        stream = torch.cuda.Stream(device=device)
        _SHARED_EXPERT_STREAMS[idx] = stream
    return stream


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
            half = stacked.shape[-2] // 2
            stacked = torch.cat([stacked[..., half:, :], stacked[..., :half, :]], dim=-2).contiguous()
        out[f"{prefix}.{proj}"] = stacked
    return out


def get_hi3_parallel_plan():
    """Return the VeOmni ``ParallelPlan`` for HI3 expert parallelism (Shard(0))."""
    from torch.distributed.tensor import Shard
    from veomni.distributed.parallel_plan import ParallelPlan

    ep_plan = {fqn: Shard(dim) for fqn, dim in _EP_PLAN.items()}
    return ParallelPlan(extra_parallel_plan={"ep": ep_plan})


def _resolve_veomni_moe_op(moe_implementation: str = "fused_triton"):
    try:
        from veomni.ops import OP_REGISTRY, VeomniOp

        impl = moe_implementation
        available = OP_REGISTRY.list_available("moe_experts", "standard")
        if impl not in available:
            impl = "fused_triton" if "fused_triton" in available else "eager"
        return VeomniOp("moe_experts", "standard", impl)
    except (ImportError, AttributeError, TypeError, ValueError, KeyError):
        return None


class FusedExperts(nn.Module):
    """Holds the fused expert weights so their FQN matches what :data:`_EP_PLAN` and the converter target."""

    def __init__(
        self,
        num_experts: int,
        hidden: int,
        inter: int,
        dtype,
        device,
        *,
        moe_implementation: str = "fused_triton",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden
        self.intermediate_dim = inter
        self.act_fn = F.silu
        self.moe_implementation = moe_implementation
        self.gate_and_up_proj = nn.Parameter(torch.empty(num_experts, 2 * inter, hidden, dtype=dtype, device=device))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden, inter, dtype=dtype, device=device))
        self.veomni_moe = _resolve_veomni_moe_op(moe_implementation)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        from torch.distributed.tensor import DTensor

        gate_up = self.gate_and_up_proj
        down = self.down_proj
        if isinstance(gate_up, DTensor):
            gate_up = gate_up.to_local()
        if isinstance(down, DTensor):
            down = down.to_local()

        if self.veomni_moe is not None:
            unused = gate_up.new_empty(0)
            return self.veomni_moe(
                hidden_states,
                top_k_weights,
                top_k_index,
                unused,
                unused,
                down,
                gate_up,
                num_experts=self.num_experts,
            )

        if self.moe_implementation == "fused_quack":
            from veomni.ops.kernels.moe.quack_gemm import quack_gemm_fused_moe_forward as moe_fn
        else:
            from veomni.ops.kernels.moe.group_gemm import group_gemm_fused_moe_forward as moe_fn

        return moe_fn(
            num_experts=self.num_experts,
            routing_weights=top_k_weights,
            selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_weight=None,
            fc1_2_weight=None,
            fc2_weight=down,
            fc1_1_2_weight=gate_up,
        )


class FusedHunyuanMoE(nn.Module):
    """EP drop-in for HI3's ``HunyuanMoE``: fused experts via VeOmni grouped GEMM + all_to_all + stream overlap."""

    _supports_native_router_replay: bool = True

    def __init__(
        self,
        gate: nn.Module,
        shared_mlp: Optional[nn.Module],
        num_experts: int,
        hidden: int,
        inter: int,
        dtype,
        device,
        *,
        moe_implementation: str = "fused_triton",
        overlap_shared_expert: bool = True,
    ):
        super().__init__()
        self.gate = gate
        self.shared_mlp = shared_mlp
        self.num_experts = num_experts
        self.overlap_shared_expert = bool(overlap_shared_expert)
        self.experts = FusedExperts(
            num_experts,
            hidden,
            inter,
            dtype,
            device,
            moe_implementation=moe_implementation,
        )

    def _apply_router_replay(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if get_active_replay() is None:
            return topk_weights, topk_idx

        replayed_idx = maybe_replay_indices(self.gate, topk_weights, topk_idx)
        if not torch.equal(replayed_idx, topk_idx):
            scores: Optional[torch.Tensor] = None
            hidden = hidden_states.shape[-1]
            scoring_func = getattr(self.gate, "scoring_func", "softmax")
            if hasattr(self.gate, "wg") and isinstance(self.gate.wg, nn.Module):
                wg_w = getattr(self.gate.wg, "weight", None)
                in_dtype = wg_w.dtype if isinstance(wg_w, torch.Tensor) else hidden_states.dtype
                logits = self.gate.wg(hidden_states.reshape(-1, hidden).to(in_dtype)).float()
                scores = torch.sigmoid(logits) if scoring_func == "sigmoid" else torch.softmax(logits, dim=-1)
            elif hasattr(self.gate, "weight") and isinstance(self.gate.weight, torch.Tensor):
                logits = F.linear(hidden_states.reshape(-1, hidden).float(), self.gate.weight.float())
                scores = torch.sigmoid(logits) if scoring_func == "sigmoid" else torch.softmax(logits, dim=-1)
            if scores is not None:
                flat_idx = replayed_idx.reshape(scores.shape[0], -1).long()
                gathered = scores.gather(1, flat_idx)
                if getattr(self.gate, "norm_topk_prob", False):
                    gathered = gathered / gathered.sum(dim=-1, keepdim=True).clamp_min(1e-20)
                scaling = float(getattr(self.gate, "routed_scaling_factor", 1.0))
                if scaling != 1.0:
                    gathered = gathered * scaling
                topk_weights = gathered.to(dtype=topk_weights.dtype).view_as(topk_weights)
        return topk_weights, replayed_idx

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq, hidden = hidden_states.shape
        use_overlap = self.shared_mlp is not None and self.overlap_shared_expert and hidden_states.is_cuda

        if use_overlap:
            main_stream = torch.cuda.current_stream(hidden_states.device)
            side_stream = _get_shared_expert_stream(hidden_states.device)
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                hidden_states.record_stream(side_stream)
                shared = self.shared_mlp(hidden_states)  # type: ignore[misc]
                shared.record_stream(main_stream)
        else:
            shared = self.shared_mlp(hidden_states) if self.shared_mlp is not None else None

        topk_weights, topk_idx = self.gate(hidden_states, topk_impl="easy")
        topk_weights, topk_idx = self._apply_router_replay(hidden_states, topk_weights, topk_idx)
        topk_weights = topk_weights.to(hidden_states.dtype)

        y = self.experts(
            hidden_states.reshape(-1, hidden).contiguous(),
            topk_idx.reshape(-1, topk_idx.shape[-1]),
            topk_weights.reshape(-1, topk_weights.shape[-1]),
        ).view(bsz, seq, hidden)

        if use_overlap:
            torch.cuda.current_stream(hidden_states.device).wait_stream(side_stream)

        return y + shared if shared is not None else y


def is_fused_expert_key(key: str) -> bool:
    """A fused-expert param key (``*.experts.gate_and_up_proj`` / ``.down_proj``)."""
    return key.endswith((".experts.gate_and_up_proj", ".experts.down_proj"))


def replace_hunyuan_moe_with_fused(
    decoder: nn.Module,
    *,
    moe_implementation: str = "fused_triton",
    overlap_shared_expert: bool = True,
) -> int:
    """In-place: swap every ``HunyuanMoE`` mlp for a :class:`FusedHunyuanMoE` and attach ``get_parallel_plan``."""
    n = 0
    for layer in decoder.layers:
        mlp = layer.mlp
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
            moe_implementation=moe_implementation,
            overlap_shared_expert=overlap_shared_expert,
        )
        n += 1
    if n == 0:
        raise RuntimeError(f"replace_hunyuan_moe_with_fused: found 0 HunyuanMoE layers on {type(decoder).__name__}")
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
