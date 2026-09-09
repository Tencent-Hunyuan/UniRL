"""Process-global precision settings without enabling vLLM's global BI mode."""

from __future__ import annotations

_INSTALLED = False


def install_precision_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import torch

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
        False,
        False,
    )
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
        False,
        False,
    )
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    _INSTALLED = True


__all__ = ["install_precision_contract"]
