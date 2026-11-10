"""HunyuanImage 3.0 expert-parallel (EP) wiring for the VeOmni backend."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Set
from typing import Dict, Literal, Optional

import torch
from torch import nn

_PEFT_PREFIX = "base_model.model."
_GATE_AND_UP_SUFFIXES = (".experts.gate_and_up_proj", ".experts.gate_up_proj")
_DOWN_SUFFIX = ".experts.down_proj"
_EXPERT_RE = re.compile(r"^(?P<prefix>.*\.experts)\.(?P<idx>\d+)\.(?P<proj>gate_and_up_proj|down_proj)\.weight$")

_EP_PLAN = {
    "layers.*.mlp.experts.gate_and_up_proj": 0,
    "layers.*.mlp.experts.gate_up_proj": 0,
    "layers.*.mlp.experts.down_proj": 0,
}

HunyuanFusedExpertKind = Literal["gate_and_up", "down"]


def normalize_fused_param_name(name: str) -> str:
    """Normalize PEFT's envelope/base-layer hop to the fused model FQN."""
    return name.removeprefix(_PEFT_PREFIX).replace(".base_layer.", ".").removesuffix(".weight")


def fused_expert_kind(name: str) -> HunyuanFusedExpertKind | None:
    """Classify a fused HunyuanImage-3.0 expert parameter name."""
    name = normalize_fused_param_name(name)
    if name.endswith(_GATE_AND_UP_SUFFIXES):
        return "gate_and_up"
    if name.endswith(_DOWN_SUFFIX):
        return "down"
    return None


def is_fused_expert_param(name: str) -> bool:
    """Whether ``name`` is a fused HI3 expert parameter (protocol hook for ``ep/experts.py``)."""
    return fused_expert_kind(name) is not None


def _swap_gate_up_halves(gate_and_up: torch.Tensor) -> torch.Tensor:
    """Swap the two output halves of a fused gate_and_up tensor ``[..., 2I, H]``."""
    if gate_and_up.ndim < 2 or gate_and_up.shape[-2] % 2 != 0:
        raise RuntimeError(f"HI3 layout: invalid fused gate_and_up shape {tuple(gate_and_up.shape)}.")
    half = gate_and_up.shape[-2] // 2
    return torch.cat([gate_and_up[..., half:, :], gate_and_up[..., :half, :]], dim=-2)


def _candidate_expert_prefixes(prefix: str) -> tuple[str, ...]:
    """Return checkpoint prefix candidates for bare-decoder or wrapped model FQNs."""
    candidates = [prefix]
    if prefix.startswith("model."):
        candidates.append(prefix.removeprefix("model."))
    else:
        candidates.append(f"model.{prefix}")
    return tuple(dict.fromkeys(candidates))


def build_local_fused_block(
    *,
    fused_param_name: str,
    expected_shape: tuple[int, ...],
    ep_rank: int,
    available_keys: Set[str],
    get_tensor: Callable[[str], torch.Tensor],
) -> torch.Tensor | None:
    """Build this EP rank's fused HI3 expert block directly from per-expert HF keys."""
    kind = fused_expert_kind(fused_param_name)
    if kind is None:
        return None

    name = normalize_fused_param_name(fused_param_name)
    prefix = name.rsplit(".experts.", 1)[0]
    num_local = int(expected_shape[0])
    start = ep_rank * num_local
    experts = range(start, start + num_local)
    proj = "gate_and_up_proj" if kind == "gate_and_up" else "down_proj"

    resolved_keys: list[str] | None = None
    for cand_prefix in _candidate_expert_prefixes(prefix):
        keys = [f"{cand_prefix}.experts.{expert}.{proj}.weight" for expert in experts]
        present = [key for key in keys if key in available_keys]
        if not present:
            continue
        if len(present) != len(keys):
            missing = next(key for key in keys if key not in available_keys)
            raise RuntimeError(
                f"HI3 layout: incomplete checkpoint for {name!r}: "
                f"{len(present)}/{len(keys)} keys present (e.g. missing {missing!r})."
            )
        resolved_keys = keys
        break

    if resolved_keys is None:
        return None

    block = torch.stack([get_tensor(key) for key in resolved_keys], dim=0)
    if kind == "gate_and_up":
        block = _swap_gate_up_halves(block).contiguous()

    if tuple(block.shape) != expected_shape:
        raise RuntimeError(
            f"HI3 layout: rebuilt {name!r} shape {tuple(block.shape)} != expected {expected_shape} "
            f"(ep_rank={ep_rank}, num_local={num_local})."
        )
    return block


def iter_hf_expert_tensors(
    fused_param_name: str,
    stacked: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield HF per-expert tensors from a full VeOmni fused HI3 stack."""
    kind = fused_expert_kind(fused_param_name)
    if kind is None:
        raise RuntimeError(f"HI3 layout: unsupported fused parameter {fused_param_name!r}.")
    name = normalize_fused_param_name(fused_param_name)
    prefix = name.rsplit(".experts.", 1)[0]

    if kind == "gate_and_up":
        unswapped = _swap_gate_up_halves(stacked)
        for expert in range(int(unswapped.shape[0])):
            yield f"{prefix}.experts.{expert}.gate_and_up_proj.weight", unswapped[expert].contiguous()
    else:
        for expert in range(int(stacked.shape[0])):
            yield f"{prefix}.experts.{expert}.down_proj.weight", stacked[expert].contiguous()


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
    from torch.distributed.tensor import Shard
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
    """A fused-expert param key (``*.experts.gate_and_up_proj`` / ``.gate_up_proj`` / ``.down_proj``)."""
    norm = normalize_fused_param_name(key)
    return norm.endswith((".experts.gate_and_up_proj", ".experts.gate_up_proj", ".experts.down_proj"))


def replace_hunyuan_moe_with_fused(decoder: nn.Module) -> int:
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
        )
        n += 1
    if n == 0:
        raise RuntimeError(f"replace_hunyuan_moe_with_fused: found 0 HunyuanMoE layers on {type(decoder).__name__}")
    decoder.get_parallel_plan = get_hi3_parallel_plan  # type: ignore[attr-defined]
    return n


__all__ = [
    "HunyuanFusedExpertKind",
    "normalize_fused_param_name",
    "fused_expert_kind",
    "is_fused_expert_param",
    "build_local_fused_block",
    "iter_hf_expert_tensors",
    "fuse_expert_state_dict",
    "get_hi3_parallel_plan",
    "FusedHunyuanMoE",
    "FusedExperts",
    "replace_hunyuan_moe_with_fused",
    "is_fused_expert_key",
]
