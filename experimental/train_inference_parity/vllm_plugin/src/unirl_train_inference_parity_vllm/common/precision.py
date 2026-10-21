"""Process-global precision settings without enabling vLLM's global BI mode."""

from __future__ import annotations

from ..compat import require_symbol, require_value
from ..registry import SymbolResult, value_result

_INSTALLED = False


def preflight_precision_contract(*, strict: bool) -> None:
    for symbol in (
        "matmul.fp32_precision",
        "matmul.allow_bf16_reduced_precision_reduction",
        "matmul.allow_fp16_reduced_precision_reduction",
    ):
        require_value(
            "torch.backends.cuda",
            symbol,
            expected_type=str if symbol.endswith("fp32_precision") else bool,
        )
    for symbol in ("conv.fp32_precision", "rnn.fp32_precision"):
        require_value("torch.backends.cudnn", symbol, expected_type=str)
    require_symbol(
        "torch.backends.cuda",
        "preferred_blas_library",
        parameters=(("backend", "None"),),
        origin="torch.backends.cuda.preferred_blas_library",
        strict=strict,
    )


def install_precision_contract() -> tuple[SymbolResult, ...]:
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("precision parity contract installed twice")
    import torch

    before = {
        "matmul_fp32": torch.backends.cuda.matmul.fp32_precision,
        "conv_fp32": torch.backends.cudnn.conv.fp32_precision,
        "rnn_fp32": torch.backends.cudnn.rnn.fp32_precision,
        "bf16_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "blas": torch.backends.cuda.preferred_blas_library(),
    }
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.preferred_blas_library(backend="cublaslt")
    _INSTALLED = True
    return (
        value_result(
            "torch.backends.cuda.matmul.fp32_precision",
            "literal:'ieee'",
            before=before["matmul_fp32"],
            actual=torch.backends.cuda.matmul.fp32_precision,
            verified=torch.backends.cuda.matmul.fp32_precision == "ieee",
        ),
        value_result(
            "torch.backends.cudnn.conv.fp32_precision",
            "literal:'ieee'",
            before=before["conv_fp32"],
            actual=torch.backends.cudnn.conv.fp32_precision,
            verified=torch.backends.cudnn.conv.fp32_precision == "ieee",
        ),
        value_result(
            "torch.backends.cudnn.rnn.fp32_precision",
            "literal:'ieee'",
            before=before["rnn_fp32"],
            actual=torch.backends.cudnn.rnn.fp32_precision,
            verified=torch.backends.cudnn.rnn.fp32_precision == "ieee",
        ),
        value_result(
            "torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction",
            "literal:false",
            before=before["bf16_reduction"],
            actual=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            verified=not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        ),
        value_result(
            "torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction",
            "literal:false",
            before=before["fp16_reduction"],
            actual=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            verified=not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        ),
        value_result(
            "torch.backends.cuda.preferred_blas_library",
            "torch._C._BlasBackend.Cublaslt",
            before=before["blas"],
            actual=torch.backends.cuda.preferred_blas_library(),
            verified=torch.backends.cuda.preferred_blas_library().name == "Cublaslt",
        ),
    )


__all__ = ["install_precision_contract", "preflight_precision_contract"]
