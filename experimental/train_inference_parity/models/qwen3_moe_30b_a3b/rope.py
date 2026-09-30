"""Deterministic Qwen3 rotary embedding implementation."""

from __future__ import annotations

import torch

from .context import exact_enabled


def rotary_forward(self, x, position_ids):
    if not exact_enabled():
        original = getattr(rotary_forward, "_unirl_parity_original", None)
        if not callable(original):
            raise RuntimeError("Qwen3 parity RoPE original is unavailable")
        return original(self, x, position_ids)
    base = self.config.rope_parameters["rope_theta"]
    dimension = getattr(self.config, "head_dim", None) or (self.config.hidden_size // self.config.num_attention_heads)
    inverse = 1.0 / (base ** (torch.arange(0, dimension, 2, dtype=torch.float32, device=x.device) / dimension))
    expanded_frequency = inverse[None, :, None].expand(
        position_ids.shape[0],
        -1,
        1,
    )
    expanded_positions = position_ids[:, None, :].float()
    with torch.autocast(device_type=x.device.type, enabled=False):
        frequencies = (expanded_frequency.float() @ expanded_positions.float()).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
    return (
        (embedding.cos() * self.attention_scaling).to(x.dtype),
        (embedding.sin() * self.attention_scaling).to(x.dtype),
    )


__all__ = ["rotary_forward"]
