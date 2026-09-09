"""Public-reference FSDP actor adaptor for Qwen3-30B-A3B."""

from __future__ import annotations

import contextlib
import contextvars
import importlib.machinery
import math
import sys
import types

import torch
import torch.nn.functional as F

from unirl.models.qwen3.ar import register_exact_actor_provider

_EXACT = contextvars.ContextVar("unirl_parity_exact", default=False)
_LIBRARY = None
_INSTALLED = False


def _providers():
    from unirl_train_inference_parity_vllm.common import providers

    return providers


@contextlib.contextmanager
def exact_mode(enabled: bool = True):
    token = _EXACT.set(bool(enabled))
    try:
        yield
    finally:
        _EXACT.reset(token)


def _native_linear(input, weight, bias=None):
    output = torch.matmul(input, weight.t())
    return output if bias is None else output + bias


def _linear_cuda(input, weight, bias=None):
    if not (
        _EXACT.get()
        and input.is_cuda
        and weight.is_cuda
        and input.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and input.ndim in (2, 3)
        and weight.ndim == 2
    ):
        return _native_linear(input, weight, bias)
    original_shape = input.shape
    output = _providers().linear(
        input.reshape(-1, original_shape[-1]),
        weight,
        bias,
    )
    return output.reshape(*original_shape[:-1], output.shape[-1])


def _linear_backward_cuda(input, grad_output, weight, output_mask):
    input_2d = input.reshape(-1, input.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
    grad_input = (
        torch.matmul(grad_output, weight) if output_mask[0] else torch.empty(0, device=input.device, dtype=input.dtype)
    )
    grad_weight = (
        torch.matmul(grad_2d.t(), input_2d)
        if output_mask[1]
        else torch.empty(0, device=weight.device, dtype=weight.dtype)
    )
    grad_bias = (
        grad_2d.sum(dim=0) if output_mask[2] else torch.empty(0, device=grad_output.device, dtype=grad_output.dtype)
    )
    return grad_input, grad_weight, grad_bias


def _softmax_cuda(input, dim, dtype=None):
    source = input if dtype is None else input.to(dtype)
    return _providers().softmax(source.contiguous(), dim=dim)


def _softmax_internal_cuda(input, dim, half_to_float):
    return _softmax_cuda(input.float() if half_to_float else input, dim)


def register_aten() -> None:
    global _LIBRARY
    if _LIBRARY is not None:
        return
    library = torch.library.Library("aten", "IMPL")
    library.impl("aten::linear", _linear_cuda, "CUDA")
    library.impl("aten::linear_backward", _linear_backward_cuda, "CUDA")
    try:
        library.impl("aten::softmax", _softmax_cuda, "CUDA")
        library.impl("aten::_softmax", _softmax_internal_cuda, "CUDA")
    except RuntimeError as error:
        if "already a kernel registered" not in str(error):
            raise
    _LIBRARY = library


def exact_context():
    register_aten()
    return exact_mode(True)


class _LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        output = _providers().log_softmax(input, dim=int(dim))
        ctx.save_for_backward(output)
        ctx.dim = int(dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (output,) = ctx.saved_tensors
        grad_input = grad_output - output.exp() * grad_output.sum(
            dim=ctx.dim,
            keepdim=True,
        )
        return grad_input, None


def _log_softmax(input, dim=-1):
    return _LogSoftmax.apply(input, dim)


class _RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, eps):
        ctx.save_for_backward(hidden, weight)
        ctx.eps = float(eps)
        return _providers().rms_norm(hidden, weight.to(hidden.dtype), float(eps))

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight = ctx.saved_tensors
        needs_hidden, needs_weight = ctx.needs_input_grad[:2]
        with torch.enable_grad():
            hidden_ref = hidden.detach().requires_grad_(needs_hidden)
            weight_ref = weight.detach().requires_grad_(needs_weight)
            variance = hidden_ref.float().square().mean(dim=-1, keepdim=True)
            output = (hidden_ref.float() * torch.rsqrt(variance + ctx.eps) * weight_ref.float()).to(hidden.dtype)
            requested = [
                tensor
                for tensor, needed in (
                    (hidden_ref, needs_hidden),
                    (weight_ref, needs_weight),
                )
                if needed
            ]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_output,
                allow_unused=True,
            )
        iterator = iter(computed)
        return (
            next(iterator) if needs_hidden else None,
            next(iterator) if needs_weight else None,
            None,
        )


def _rmsnorm_forward(self, hidden_states):
    return _RmsNorm.apply(
        hidden_states,
        self.weight,
        float(self.variance_epsilon),
    )


def _router_forward(self, hidden_states):
    hidden = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden, self.weight)
    probabilities = F.softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return logits, weights.float(), indices


def _dispatch(hidden, topk_ids, experts):
    tokens, topk = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    order = torch.argsort(flat_ids, stable=True)
    rows = (
        hidden[:, None, :]
        .expand(tokens, topk, hidden.shape[-1])
        .reshape(-1, hidden.shape[-1])
        .index_select(0, order)
        .contiguous()
    )
    counts = torch.bincount(flat_ids, minlength=experts)
    return rows, counts, order


def _grouped_linear(inputs, weights, counts, *, public):
    outputs = []
    offset = 0
    for expert in range(int(weights.shape[0])):
        count = int(counts[expert].item())
        if count:
            rows = inputs[offset : offset + count].contiguous()
            outputs.append(
                _providers().linear(rows, weights[expert], None) if public else F.linear(rows, weights[expert])
            )
        offset += count
    return torch.cat(outputs, dim=0)


def _combine(contributions, order, tokens, topk, hidden):
    row_map = torch.empty(tokens * topk, dtype=torch.long, device=order.device)
    row_map[order] = torch.arange(tokens * topk, device=order.device)
    row_map = row_map.view(tokens, topk)
    total = torch.zeros((tokens, hidden), dtype=torch.float32, device=contributions.device)
    for slot in range(topk):
        value = contributions.index_select(0, row_map[:, slot])
        total = total + value.float()
    total = total.to(torch.bfloat16).float()
    return total.to(torch.bfloat16)


def _moe_forward(hidden, topk_ids, topk_weights, w13, w2, *, public):
    tokens, hidden_size = hidden.shape
    topk = int(topk_ids.shape[1])
    experts = int(w13.shape[0])
    dispatched, counts, order = _dispatch(hidden, topk_ids, experts)
    tp = int(__import__("os").environ.get("UNIRL_PARITY_TRAIN_TP", "4"))
    intermediate = int(w13.shape[1]) // 2
    local_intermediate = intermediate // tp
    fc1_parts = []
    for rank in range(tp):
        begin, end = rank * local_intermediate, (rank + 1) * local_intermediate
        rank_weight = torch.cat(
            (w13[:, begin:end], w13[:, intermediate + begin : intermediate + end]),
            dim=1,
        ).contiguous()
        fc1_parts.append(_grouped_linear(dispatched, rank_weight, counts, public=public))
    packed = torch.cat(fc1_parts, dim=-1).view(-1, tp, 2, local_intermediate)
    activation = (F.silu(packed[:, :, 0]) * packed[:, :, 1]).reshape(-1, intermediate)
    local_hidden = hidden_size // tp
    fc2_parts = []
    for rank in range(tp):
        rank_weight = w2[
            :,
            rank * local_hidden : (rank + 1) * local_hidden,
            :,
        ].contiguous()
        fc2_parts.append(_grouped_linear(activation, rank_weight, counts, public=public))
    fc2 = torch.cat(fc2_parts, dim=-1)
    sorted_weights = topk_weights.reshape(-1).index_select(0, order)
    contributions = (fc2 * sorted_weights[:, None]).to(torch.bfloat16)
    return _combine(
        contributions,
        order,
        tokens,
        topk,
        hidden_size,
    )


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, topk_ids, topk_weights, w13, w2):
        ctx.save_for_backward(hidden, topk_ids, topk_weights, w13, w2)
        return _moe_forward(
            hidden,
            topk_ids,
            topk_weights,
            w13,
            w2,
            public=True,
        )

    @staticmethod
    def backward(ctx, grad_output):
        hidden, topk_ids, topk_weights, w13, w2 = ctx.saved_tensors
        needs = ctx.needs_input_grad
        sources = (hidden, topk_weights, w13, w2)
        rebuilt = [
            tensor.detach().requires_grad_(needed)
            for tensor, needed in zip(
                sources,
                (needs[0], needs[2], needs[3], needs[4]),
                strict=True,
            )
        ]
        with torch.enable_grad():
            output = _moe_forward(
                rebuilt[0],
                topk_ids,
                rebuilt[1],
                rebuilt[2],
                rebuilt[3],
                public=False,
            )
            requested = [tensor for tensor in rebuilt if tensor.requires_grad]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_output,
                allow_unused=True,
            )
        iterator = iter(computed)
        gradients = [next(iterator) if tensor.requires_grad else None for tensor in rebuilt]
        return gradients[0], None, gradients[1], gradients[2], gradients[3]


def _sparse_moe_forward(self, hidden_states):
    shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, shape[-1])
    _logits, weights, indices = self.gate(hidden)
    output = _Moe.apply(
        hidden,
        indices,
        weights,
        self.experts.gate_up_proj,
        self.experts.down_proj,
    )
    return output.view(shape)


def _rotary_forward(self, x, position_ids):
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


def _math_attention_backward(q, k, v, grad_output, *, scale, causal):
    with torch.enable_grad():
        qr = q.detach().requires_grad_(q.requires_grad)
        kr = k.detach().requires_grad_(k.requires_grad)
        vr = v.detach().requires_grad_(v.requires_grad)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                qr.transpose(1, 2),
                kr.transpose(1, 2),
                vr.transpose(1, 2),
                is_causal=causal,
                scale=scale,
                enable_gqa=qr.shape[-2] != kr.shape[-2],
            ).transpose(1, 2)
        return torch.autograd.grad(
            output,
            (qr, kr, vr),
            grad_output,
            allow_unused=True,
        )


class _Fa3Fixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        batch, sequence = q.shape[:2]
        cu = torch.arange(
            0,
            (batch + 1) * sequence,
            sequence,
            dtype=torch.int32,
            device=q.device,
        )
        output = flash_attn_varlen_func(
            q=q.reshape(-1, *q.shape[2:]),
            k=k.reshape(-1, *k.shape[2:]),
            v=v.reshape(-1, *v.shape[2:]),
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=sequence,
            max_seqlen_k=sequence,
            softmax_scale=scale,
            causal=causal,
            deterministic=True,
            num_splits=1,
            fa_version=3,
        ).view_as(q)
        ctx.save_for_backward(q, k, v)
        ctx.scale, ctx.causal = scale, causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (
            *_math_attention_backward(
                *ctx.saved_tensors,
                grad_output,
                scale=ctx.scale,
                causal=ctx.causal,
            ),
            None,
            None,
        )


class _Fa3Varlen(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_q, cu_k, max_q, max_k, scale, causal):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        output = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=scale,
            causal=causal,
            deterministic=True,
            num_splits=1,
            fa_version=3,
        )
        ctx.save_for_backward(q, k, v, cu_q, cu_k)
        ctx.scale, ctx.causal = scale, causal
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, cu_q, cu_k = ctx.saved_tensors
        q_parts, k_parts, v_parts = [], [], []
        for index in range(int(cu_q.numel()) - 1):
            qb, qe = int(cu_q[index]), int(cu_q[index + 1])
            kb, ke = int(cu_k[index]), int(cu_k[index + 1])
            grads = _math_attention_backward(
                q[qb:qe].unsqueeze(0),
                k[kb:ke].unsqueeze(0),
                v[kb:ke].unsqueeze(0),
                grad_output[qb:qe].unsqueeze(0),
                scale=ctx.scale,
                causal=ctx.causal,
            )
            q_parts.append(grads[0].squeeze(0))
            k_parts.append(grads[1].squeeze(0))
            v_parts.append(grads[2].squeeze(0))
        return (
            torch.cat(q_parts),
            torch.cat(k_parts),
            torch.cat(v_parts),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_kwargs):
    if dropout_p:
        raise ValueError("parity FA3 requires dropout=0")
    scale = float(softmax_scale or (1.0 / math.sqrt(q.shape[-1])))
    return _Fa3Fixed.apply(q, k, v, scale, bool(causal))


def _flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    **kwargs,
):
    if float(kwargs.get("dropout_p", 0.0) or 0.0):
        raise ValueError("parity FA3 requires dropout=0")
    scale = float(softmax_scale or (1.0 / math.sqrt(q.shape[-1])))
    return _Fa3Varlen.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        scale,
        bool(causal),
    )


def _install_fa3() -> None:
    module = types.ModuleType("flash_attn_interface")
    module.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface",
        loader=None,
    )
    module.flash_attn_func = _flash_attn_func
    module.flash_attn_varlen_func = _flash_attn_varlen_func
    module.flash_attn_with_kvcache = None
    sys.modules["flash_attn_interface"] = module
    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    def available():
        return torch.cuda.is_available()

    available.cache_clear = lambda: None
    import_utils.is_flash_attn_3_available.cache_clear()
    import_utils.is_flash_attn_3_available = available
    transformers_utils.is_flash_attn_3_available = available
    flash_utils.is_flash_attn_3_available = available
    modeling_utils.FLASH_ATTENTION_COMPATIBILITY_MATRIX[3]["general_availability_check"] = available
    modeling_utils.FLASH_ATTENTION_COMPATIBILITY_MATRIX[3]["pkg_availability_check"] = lambda *args, **kwargs: True


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    register_aten()
    register_exact_actor_provider(
        context_factory=exact_context,
        log_softmax=_log_softmax,
    )
    _install_fa3()
    from transformers.models.qwen3_moe import modeling_qwen3_moe as modeling

    modeling.Qwen3MoeTopKRouter.forward = _router_forward
    modeling.Qwen3MoeRMSNorm.forward = _rmsnorm_forward
    modeling.Qwen3MoeSparseMoeBlock.forward = _sparse_moe_forward
    modeling.Qwen3MoeRotaryEmbedding.forward = _rotary_forward
    _INSTALLED = True


install()


__all__ = ["exact_mode", "install", "register_aten"]
