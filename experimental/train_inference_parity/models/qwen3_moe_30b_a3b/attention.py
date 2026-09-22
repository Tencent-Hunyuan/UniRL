"""vLLM FA3 facade with deterministic forward and finite surrogate backward."""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import math
import types

import torch
import torch.nn.functional as F

from .context import exact_enabled


@functools.lru_cache(maxsize=1)
def probe_fa3_backend():
    """Import and validate the real bundled backend before advertising FA3."""
    try:
        backend = importlib.import_module("vllm.vllm_flash_attn")
    except Exception as error:
        raise RuntimeError("Qwen3 parity FA3 requires importable vllm.vllm_flash_attn") from error
    if not callable(getattr(backend, "flash_attn_varlen_func", None)):
        raise RuntimeError("vllm.vllm_flash_attn does not expose flash_attn_varlen_func")
    return backend


def fa3_available() -> bool:
    try:
        probe_fa3_backend()
    except RuntimeError:
        return False
    return torch.cuda.is_available()


def _require_bfloat16(*tensors: torch.Tensor) -> None:
    invalid = [str(tensor.dtype) for tensor in tensors if tensor.dtype != torch.bfloat16]
    if invalid:
        raise ValueError(f"Qwen3 parity FA3 requires BF16 tensors, got {invalid}")


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
        gradients = torch.autograd.grad(
            output,
            (qr, kr, vr),
            grad_output,
            allow_unused=True,
        )
    return gradients


class _Fa3Fixed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, causal):
        _require_bfloat16(q, k, v)
        flash_attn_varlen_func = probe_fa3_backend().flash_attn_varlen_func
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
        _require_bfloat16(q, k, v)
        flash_attn_varlen_func = probe_fa3_backend().flash_attn_varlen_func
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
            gradients = _math_attention_backward(
                q[qb:qe].unsqueeze(0),
                k[kb:ke].unsqueeze(0),
                v[kb:ke].unsqueeze(0),
                grad_output[qb:qe].unsqueeze(0),
                scale=ctx.scale,
                causal=ctx.causal,
            )
            q_parts.append(gradients[0].squeeze(0))
            k_parts.append(gradients[1].squeeze(0))
            v_parts.append(gradients[2].squeeze(0))
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


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_kwargs):
    if not exact_enabled():
        raise RuntimeError("Qwen3 parity FA3 facade may only run inside stage.exact_context()")
    if dropout_p:
        raise ValueError("parity FA3 requires dropout=0")
    scale = float(softmax_scale or (1.0 / math.sqrt(q.shape[-1])))
    return _Fa3Fixed.apply(q, k, v, scale, bool(causal))


def flash_attn_varlen_func(
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
    if not exact_enabled():
        raise RuntimeError("Qwen3 parity FA3 facade may only run inside stage.exact_context()")
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


def make_flash_attn_interface() -> types.ModuleType:
    """Build the Transformers-facing facade after the backend probe passed."""
    probe_fa3_backend()
    module = types.ModuleType("flash_attn_interface")
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn_interface", loader=None)
    module.flash_attn_func = flash_attn_func
    module.flash_attn_varlen_func = flash_attn_varlen_func
    module.flash_attn_with_kvcache = None
    return module


__all__ = ["fa3_available", "make_flash_attn_interface", "probe_fa3_backend"]
