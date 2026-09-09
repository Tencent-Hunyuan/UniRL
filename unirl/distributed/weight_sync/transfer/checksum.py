"""Shared tensor-fingerprint helpers for trainer ↔ rollout-worker value-correctness checks."""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Tuple, Union

import torch


def fingerprint_tensor(t: torch.Tensor) -> str:
    """Return a 16-char hex SHA-256 prefix over ``(dtype, shape, all bytes)``."""
    data = t.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(data.dtype).encode())
    h.update(str(tuple(data.shape)).encode())
    h.update(data.view(torch.uint8).flatten().numpy().tobytes())
    return h.hexdigest()[:16]


def compute_param_checksums(
    named_tensors: Union[
        Dict[str, torch.Tensor],
        Iterable[Tuple[str, torch.Tensor]],
    ],
) -> Dict[str, str]:
    """Hash each tensor in a name→tensor dict (or ``[(name, tensor), ...]``)."""
    items = named_tensors.items() if isinstance(named_tensors, dict) else named_tensors
    return {name: fingerprint_tensor(t) for name, t in items}


def _is_lora_b_name(name: str) -> bool:
    """Heuristic: PEFT names ``lora_B`` matrices with the ``.lora_B.`` substring (or trailing ``.lora_B.weight``)."""
    return ".lora_B." in name or name.endswith(".lora_B.weight")


def compute_lora_checksums_post_optimize(
    lora_tensors: Dict[str, torch.Tensor],
    peft_config: Dict,
) -> Dict[str, str]:
    """Hash each LoRA tensor as it appears after the worker's ``lora.optimize()``."""
    r = float(peft_config.get("r", peft_config.get("rank", 8)))
    alpha = float(peft_config.get("lora_alpha", peft_config.get("alpha", r)))
    scale = alpha / r if r else 1.0
    out: Dict[str, str] = {}
    for name, t in lora_tensors.items():
        scaled = t * scale if _is_lora_b_name(name) else t
        out[name] = fingerprint_tensor(scaled)
    return out


__all__ = [
    "fingerprint_tensor",
    "compute_param_checksums",
    "compute_lora_checksums_post_optimize",
]
