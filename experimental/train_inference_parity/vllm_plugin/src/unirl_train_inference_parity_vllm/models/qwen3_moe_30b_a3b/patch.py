"""Install the public-reference Qwen3-MoE vLLM patch."""

from __future__ import annotations

import importlib

import torch

from ...common.providers import preflight_providers
from ...compat import require_symbol
from ...registry import PatchResult, SymbolResult, symbol_result, value_result
from .dense import make_attention, make_logits_processor
from .moe import make_moe_block

_INSTALLED = False


def _install_rope() -> tuple[SymbolResult, ...]:
    import vllm.model_executor.layers.rotary_embedding.base as rope

    original_inverse_frequency = rope.RotaryEmbeddingBase._compute_inv_freq
    original_forward_cuda = rope.RotaryEmbedding.forward_cuda

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
    return (
        symbol_result(
            ("vllm.model_executor.layers.rotary_embedding.base.RotaryEmbeddingBase._compute_inv_freq"),
            inverse_frequency,
            before=original_inverse_frequency,
            actual=rope.RotaryEmbeddingBase._compute_inv_freq,
        ),
        symbol_result(
            ("vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_cuda"),
            apply,
            before=original_forward_cuda,
            actual=rope.RotaryEmbedding.forward_cuda,
        ),
    )


def preflight_qwen3_moe_patch(strict: bool) -> None:
    preflight_providers(strict=strict)
    contracts = (
        (
            "vllm.model_executor.layers.rotary_embedding.base",
            "RotaryEmbeddingBase._compute_inv_freq",
            (("self", "<required>"), ("base", "<required>")),
            ("vllm.model_executor.layers.rotary_embedding.base.RotaryEmbeddingBase._compute_inv_freq"),
        ),
        (
            "vllm.model_executor.layers.rotary_embedding.base",
            "RotaryEmbedding.forward_cuda",
            (
                ("self", "<required>"),
                ("positions", "<required>"),
                ("query", "<required>"),
                ("key", "None"),
            ),
            ("vllm.model_executor.layers.rotary_embedding.base.RotaryEmbedding.forward_cuda"),
        ),
        (
            "vllm.model_executor.models.qwen3_moe",
            "Qwen3MoeSparseMoeBlock",
            (("vllm_config", "<required>"), ("prefix", "''")),
            "vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock",
        ),
        (
            "vllm.model_executor.models.qwen3_moe",
            "Qwen3MoeSparseMoeBlock.forward",
            (("self", "<required>"), ("hidden_states", "<required>")),
            ("vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock.forward"),
        ),
        (
            "vllm.model_executor.models.qwen3_moe",
            "Qwen3MoeAttention",
            (
                ("hidden_size", "<required>"),
                ("num_heads", "<required>"),
                ("num_kv_heads", "<required>"),
                ("rope_parameters", "<required>"),
                ("max_position_embeddings", "8192"),
                ("head_dim", "None"),
                ("rms_norm_eps", "1e-06"),
                ("qkv_bias", "False"),
                ("cache_config", "None"),
                ("quant_config", "None"),
                ("prefix", "''"),
                ("dual_chunk_attention_config", "None"),
            ),
            "vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention",
        ),
        (
            "vllm.model_executor.models.qwen3_moe",
            "Qwen3MoeAttention.forward",
            (
                ("self", "<required>"),
                ("positions", "<required>"),
                ("hidden_states", "<required>"),
            ),
            ("vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention.forward"),
        ),
        (
            "vllm.model_executor.models.qwen3_moe",
            "LogitsProcessor",
            (
                ("vocab_size", "<required>"),
                ("org_vocab_size", "None"),
                ("scale", "1.0"),
                ("logits_as_input", "False"),
                ("soft_cap", "None"),
            ),
            "vllm.model_executor.layers.logits_processor.LogitsProcessor",
        ),
        (
            "vllm.model_executor.models.qwen3_moe",
            "LogitsProcessor._get_logits",
            (
                ("self", "<required>"),
                ("hidden_states", "<required>"),
                ("lm_head", "<required>"),
                ("embedding_bias", "<required>"),
            ),
            ("vllm.model_executor.layers.logits_processor.LogitsProcessor._get_logits"),
        ),
        (
            "vllm.model_executor.layers.fused_moe.moe_permute_unpermute",
            "moe_permute",
            (
                ("hidden_states", "<required>"),
                ("a1q_scale", "<required>"),
                ("topk_ids", "<required>"),
                ("n_expert", "<required>"),
                ("n_local_expert", "-1"),
                ("expert_map", "None"),
                ("permuted_hidden_states", "None"),
            ),
            ("vllm.model_executor.layers.fused_moe.moe_permute_unpermute.moe_permute"),
        ),
        (
            "vllm.distributed.parallel_state",
            "GroupCoordinator.all_gather",
            (
                ("self", "<required>"),
                ("input_", "<required>"),
                ("dim", "-1"),
            ),
            "vllm.distributed.parallel_state.GroupCoordinator.all_gather",
        ),
        (
            "vllm.model_executor.layers.linear",
            "ReplicatedLinear.forward",
            (("self", "<required>"), ("x", "<required>")),
            "vllm.model_executor.layers.linear.ReplicatedLinear.forward",
        ),
        (
            "vllm.model_executor.layers.fused_moe.router.base_router",
            "BaseRouter.select_experts",
            (
                ("self", "<required>"),
                ("hidden_states", "<required>"),
                ("router_logits", "<required>"),
                ("input_ids", "KEYWORD_ONLY", "None"),
            ),
            "vllm.model_executor.layers.fused_moe.router.base_router.BaseRouter.select_experts",
        ),
        (
            "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
            "FusedTopKRouter._compute_routing",
            (
                ("self", "<required>"),
                ("hidden_states", "<required>"),
                ("router_logits", "<required>"),
                ("indices_type", "<required>"),
                ("input_ids", "KEYWORD_ONLY", "None"),
            ),
            ("vllm.model_executor.layers.fused_moe.router.fused_topk_router.FusedTopKRouter._compute_routing"),
        ),
    )
    for module, symbol, parameters, origin in contracts:
        require_symbol(
            module,
            symbol,
            parameters=parameters,
            origin=origin,
            strict=strict,
        )


def install_qwen3_moe_patch(*, strict: bool) -> PatchResult:
    if not strict:
        raise ValueError("the Qwen3-MoE public-reference installer requires strict=True")
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("Qwen3-MoE parity patch installed twice")
    module = importlib.import_module("vllm.model_executor.models.qwen3_moe")
    original_moe_block = module.Qwen3MoeSparseMoeBlock
    original_attention = module.Qwen3MoeAttention
    original_logits_processor = module.LogitsProcessor
    original_marker = getattr(module, "_unirl_train_inference_parity", "<missing>")
    moe_block = make_moe_block(original_moe_block)
    attention = make_attention(original_attention)
    logits_processor = make_logits_processor(original_logits_processor)
    rope_results = _install_rope()
    module.Qwen3MoeSparseMoeBlock = moe_block
    module.Qwen3MoeAttention = attention
    module.LogitsProcessor = logits_processor
    module._unirl_train_inference_parity = True
    _INSTALLED = True
    return PatchResult(
        name="qwen3_moe_30b_a3b",
        symbols=(
            *rope_results,
            symbol_result(
                ("vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock"),
                moe_block,
                before=original_moe_block,
                actual=module.Qwen3MoeSparseMoeBlock,
            ),
            symbol_result(
                "vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention",
                attention,
                before=original_attention,
                actual=module.Qwen3MoeAttention,
            ),
            symbol_result(
                "vllm.model_executor.models.qwen3_moe.LogitsProcessor",
                logits_processor,
                before=original_logits_processor,
                actual=module.LogitsProcessor,
            ),
            value_result(
                ("vllm.model_executor.models.qwen3_moe._unirl_train_inference_parity"),
                "literal:true",
                before=original_marker,
                actual=module._unirl_train_inference_parity,
                verified=module._unirl_train_inference_parity is True,
            ),
        ),
    )


__all__ = ["install_qwen3_moe_patch", "preflight_qwen3_moe_patch"]
