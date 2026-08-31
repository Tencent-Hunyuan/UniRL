"""Declare MiniMax-H3's fused SwiGLU so the diffusion LoRA manager binds gate/up adapters to it."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTINEL = "_unirl_h3_packed_lora_mapping"

# ``DiffusionLoRAManager._compute_packed_modules_mapping`` reduces each row to its two leaf
# names, so the sub-projections must be listed in the fused layer's own slice order.
_H3_PACKED_LORA_ROWS = (
    (".attn.qkv_proj", ".attn.to_q", "q"),
    (".attn.qkv_proj", ".attn.to_k", "k"),
    (".attn.qkv_proj", ".attn.to_v", "v"),
    (".mlp.fc1", ".mlp.gate_proj", 0),
    (".mlp.fc1", ".mlp.up_proj", 1),
)


def patch_minimax_h3_packed_lora_mapping() -> None:
    """Extend ``MiniMaxH3DiTModel.stacked_params_mapping`` to cover the fused qkv and fc1."""
    try:
        from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import (
            MiniMaxH3DiTModel,
        )
    except ImportError:
        return

    if getattr(MiniMaxH3DiTModel, _SENTINEL, False):
        return

    declared = getattr(MiniMaxH3DiTModel, "stacked_params_mapping", ()) or ()
    if not isinstance(declared, (list, tuple)):
        declared = ()

    # Declared rows come first: the manager keeps the first spelling it sees for a
    # given (packed, sub) leaf pair, so upstream stays authoritative where it overlaps.
    merged = list(declared)
    known = {(str(row[0]), str(row[1])) for row in merged if isinstance(row, (list, tuple)) and len(row) >= 2}
    for row in _H3_PACKED_LORA_ROWS:
        if (row[0], row[1]) not in known:
            merged.append(row)

    MiniMaxH3DiTModel.stacked_params_mapping = tuple(merged)
    setattr(MiniMaxH3DiTModel, _SENTINEL, True)
    logger.info(
        "Patched MiniMaxH3DiTModel.stacked_params_mapping: %d declared row(s) -> %d",
        len(declared),
        len(merged),
    )


__all__ = ["patch_minimax_h3_packed_lora_mapping"]
