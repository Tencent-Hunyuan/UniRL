"""Public vLLM/Torch providers used by the reference profile."""

from __future__ import annotations

import torch

from ..compat import require_symbol


def preflight_providers(*, strict: bool) -> None:
    module = "vllm.model_executor.layers.batch_invariant"
    contracts = {
        "linear_batch_invariant": (
            ("input", "<required>"),
            ("weight", "<required>"),
            ("bias", "None"),
        ),
        "rms_norm_batch_invariant": (
            ("input", "<required>"),
            ("weight", "<required>"),
            ("eps", "1e-06"),
        ),
        "log_softmax": (("input", "<required>"), ("dim", "-1")),
        "softmax_batch_invariant": (
            ("input", "<required>"),
            ("dim", "<required>"),
            ("dtype", "None"),
        ),
        "mean_batch_invariant": (
            ("input", "<required>"),
            ("dim", "<required>"),
            ("keepdim", "False"),
            ("dtype", "None"),
        ),
    }
    for name, parameters in contracts.items():
        require_symbol(
            module,
            name,
            parameters=parameters,
            origin=f"{module}.{name}",
            strict=strict,
        )


def linear(input: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    from vllm.model_executor.layers.batch_invariant import linear_batch_invariant

    return linear_batch_invariant(input.contiguous(), weight.contiguous(), bias)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from vllm.model_executor.layers.batch_invariant import rms_norm_batch_invariant

    return rms_norm_batch_invariant(input.contiguous(), weight.contiguous(), eps)


def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor:
    from vllm.model_executor.layers.batch_invariant import log_softmax as implementation

    return implementation(input, dim=dim)


def softmax(input: torch.Tensor, dim: int = -1, dtype=None) -> torch.Tensor:
    from vllm.model_executor.layers.batch_invariant import softmax_batch_invariant

    return softmax_batch_invariant(input, dim=dim, dtype=dtype)


def mean(input: torch.Tensor, dim, keepdim=False, dtype=None) -> torch.Tensor:
    from vllm.model_executor.layers.batch_invariant import mean_batch_invariant

    return mean_batch_invariant(input, dim=dim, keepdim=keepdim, dtype=dtype)


__all__ = [
    "linear",
    "log_softmax",
    "mean",
    "preflight_providers",
    "rms_norm",
    "softmax",
]
