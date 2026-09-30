"""Guarded linear and direct reduction providers for exact parity."""

from __future__ import annotations

import torch

from .context import exact_enabled


def providers():
    from unirl_train_inference_parity_vllm.common import providers as implementations

    return implementations


def native_linear(input, weight, bias=None):
    output = torch.matmul(input, weight.t())
    return output if bias is None else output + bias


def linear_cuda(input, weight, bias=None):
    """Use the BI provider only inside an exact BF16 CUDA region."""
    if not (
        exact_enabled()
        and input.is_cuda
        and weight.is_cuda
        and input.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and input.ndim in (2, 3)
        and weight.ndim == 2
    ):
        return native_linear(input, weight, bias)
    original_shape = input.shape
    output = providers().linear(
        input.reshape(-1, original_shape[-1]),
        weight,
        bias,
    )
    return output.reshape(*original_shape[:-1], output.shape[-1])


def linear_backward_cuda(input, grad_output, weight, output_mask):
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


class _LogSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim):
        output = providers().log_softmax(input, dim=int(dim))
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


def exact_log_softmax(input, dim=-1):
    """Bitwise selected-token forward provider with an analytic gradient."""
    if not exact_enabled():
        raise RuntimeError("Qwen3 parity log-softmax may only run inside stage.exact_context()")
    return _LogSoftmax.apply(input, dim)


__all__ = [
    "exact_log_softmax",
    "linear_backward_cuda",
    "linear_cuda",
    "native_linear",
    "providers",
]
