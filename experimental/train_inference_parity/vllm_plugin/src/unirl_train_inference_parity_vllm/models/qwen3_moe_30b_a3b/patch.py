"""Install the public-reference Qwen3-MoE vLLM patch."""

from __future__ import annotations

import importlib

import torch

from .dense import make_attention, make_logits_processor
from .moe import make_moe_block

_INSTALLED = False


def _install_rope() -> None:
    import vllm.model_executor.layers.rotary_embedding.base as rope

    def inverse_frequency(self, base):
        exponent = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
        return 1.0 / (base ** (exponent / self.rotary_dim))

    def apply(self, positions, query, key=None):
        positions = positions.flatten()
        cache = self._match_cos_sin_cache_dtype(query).index_select(0, positions)
        cosine, sine = cache.chunk(2, dim=-1)
        tokens = int(positions.shape[0])

        def rotate(value):
            original_shape = value.shape
            value = value.view(tokens, -1, self.head_size)
            first, second = torch.chunk(value[..., : self.rotary_dim], 2, dim=-1)
            cos = cosine.unsqueeze(-2).to(value.dtype)
            sin = sine.unsqueeze(-2).to(value.dtype)
            rotated = torch.cat(
                (first * cos - second * sin, second * cos + first * sin),
                dim=-1,
            )
            return torch.cat(
                (rotated, value[..., self.rotary_dim :]),
                dim=-1,
            ).reshape(original_shape)

        return rotate(query), None if key is None else rotate(key)

    rope.RotaryEmbeddingBase._compute_inv_freq = inverse_frequency
    rope.RotaryEmbedding.forward_cuda = apply


def install_qwen3_moe_patch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    module = importlib.import_module("vllm.model_executor.models.qwen3_moe")
    _install_rope()
    module.Qwen3MoeSparseMoeBlock = make_moe_block(module.Qwen3MoeSparseMoeBlock)
    module.Qwen3MoeAttention = make_attention(module.Qwen3MoeAttention)
    module.LogitsProcessor = make_logits_processor(module.LogitsProcessor)
    module._unirl_train_inference_parity = True
    _INSTALLED = True


__all__ = ["install_qwen3_moe_patch"]
