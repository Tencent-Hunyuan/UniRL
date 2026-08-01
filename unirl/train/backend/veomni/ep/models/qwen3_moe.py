"""Qwen3-MoE expert checkpoint layout.

This is UniRL's single runtime copy of VeOmni's
``Qwen3MoeCheckpointTensorConverter`` semantics. I/O and collectives stay in
their callers; this module owns only key naming and pure HF ↔ fused tensor
conversion so load and rollout sync cannot drift independently.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Set
from typing import Literal

import torch

_PEFT_PREFIX = "base_model.model."
_GATE_UP_SUFFIX = ".experts.gate_up_proj"
_DOWN_SUFFIX = ".experts.down_proj"

FusedExpertKind = Literal["gate_up", "down"]


def normalize_fused_param_name(name: str) -> str:
    """Normalize PEFT's envelope/base-layer hop to the fused model FQN."""
    return name.removeprefix(_PEFT_PREFIX).replace(".base_layer.", ".").removesuffix(".weight")


def fused_expert_kind(name: str) -> FusedExpertKind | None:
    """Classify a fused Qwen3-MoE expert parameter name."""
    name = normalize_fused_param_name(name)
    if name.endswith(_GATE_UP_SUFFIX):
        return "gate_up"
    if name.endswith(_DOWN_SUFFIX):
        return "down"
    return None


def is_fused_expert_param(name: str) -> bool:
    return fused_expert_kind(name) is not None


def fuse_gate_up_pair(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """HF ``gate`` + ``up`` ``[I,H]`` → VeOmni ``gate_up`` ``[2I,H]``."""
    if tuple(gate.shape) != tuple(up.shape):
        raise RuntimeError(f"Qwen3-MoE layout: gate shape {tuple(gate.shape)} != up shape {tuple(up.shape)}.")
    return torch.cat([gate, up], dim=0)


def split_gate_up_pair(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`fuse_gate_up_pair`."""
    if gate_up.ndim < 2 or gate_up.shape[-2] % 2:
        raise RuntimeError(f"Qwen3-MoE layout: invalid fused gate_up shape {tuple(gate_up.shape)}.")
    width = gate_up.shape[-2] // 2
    return gate_up[..., :width, :], gate_up[..., width:, :]


def build_local_fused_block(
    *,
    fused_param_name: str,
    expected_shape: tuple[int, ...],
    ep_rank: int,
    available_keys: Set[str],
    get_tensor: Callable[[str], torch.Tensor],
) -> torch.Tensor | None:
    """Build this EP rank's fused block from HF per-expert checkpoint keys.

    Returns ``None`` only when the checkpoint contains none of the relevant HF
    keys. Any partial projection/expert set is corruption and raises.
    """
    kind = fused_expert_kind(fused_param_name)
    if kind is None:
        return None

    name = normalize_fused_param_name(fused_param_name)
    prefix = name.rsplit(".experts.", 1)[0]
    num_local = int(expected_shape[0])
    start = ep_rank * num_local
    experts = range(start, start + num_local)

    def _resolve(keys: list[str]) -> list[str] | None:
        present = [key for key in keys if key in available_keys]
        if not present:
            return None
        if len(present) != len(keys):
            missing = next(key for key in keys if key not in available_keys)
            raise RuntimeError(
                f"Qwen3-MoE layout: incomplete checkpoint for {name!r}: "
                f"{len(present)}/{len(keys)} keys present (e.g. missing {missing!r})."
            )
        return keys

    if kind == "down":
        keys = _resolve([f"{prefix}.experts.{expert}.down_proj.weight" for expert in experts])
        if keys is None:
            return None
        block = torch.stack([get_tensor(key) for key in keys])
    else:
        gate_keys = _resolve([f"{prefix}.experts.{expert}.gate_proj.weight" for expert in experts])
        up_keys = _resolve([f"{prefix}.experts.{expert}.up_proj.weight" for expert in experts])
        if gate_keys is None and up_keys is None:
            return None
        if gate_keys is None or up_keys is None:
            missing_projection = "gate_proj" if gate_keys is None else "up_proj"
            raise RuntimeError(
                f"Qwen3-MoE layout: checkpoint has only one gate/up projection; missing {missing_projection}."
            )
        block = torch.stack(
            [
                fuse_gate_up_pair(get_tensor(gate_key), get_tensor(up_key))
                for gate_key, up_key in zip(gate_keys, up_keys)
            ]
        )

    if tuple(block.shape) != expected_shape:
        raise RuntimeError(
            f"Qwen3-MoE layout: rebuilt {name!r} shape {tuple(block.shape)} != expected {expected_shape} "
            f"(ep_rank={ep_rank}, num_local={num_local})."
        )
    return block


def iter_hf_expert_tensors(
    fused_param_name: str,
    stacked: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield HF per-expert tensors from a full VeOmni fused stack."""
    kind = fused_expert_kind(fused_param_name)
    if kind is None:
        raise RuntimeError(f"Qwen3-MoE layout: unsupported fused parameter {fused_param_name!r}.")
    name = normalize_fused_param_name(fused_param_name)
    prefix = name.rsplit(".experts.", 1)[0]

    if kind == "gate_up":
        gate, up = split_gate_up_pair(stacked)
        for expert in range(int(stacked.shape[0])):
            yield f"{prefix}.experts.{expert}.gate_proj.weight", gate[expert].contiguous()
            yield f"{prefix}.experts.{expert}.up_proj.weight", up[expert].contiguous()
    else:
        for expert in range(int(stacked.shape[0])):
            yield f"{prefix}.experts.{expert}.down_proj.weight", stacked[expert].contiguous()


__all__ = [
    "FusedExpertKind",
    "normalize_fused_param_name",
    "fused_expert_kind",
    "is_fused_expert_param",
    "fuse_gate_up_pair",
    "split_gate_up_pair",
    "build_local_fused_block",
    "iter_hf_expert_tensors",
]
