"""Route vLLM RMSNorm layers to the public BI primitive."""

from __future__ import annotations

from ..compat import require_symbol
from ..registry import SymbolResult, symbol_result

_INSTALLED = False


def preflight_norm_patch(*, strict: bool) -> None:
    require_symbol(
        "vllm.model_executor.layers.layernorm",
        "RMSNorm.forward_cuda",
        parameters=(("self", "<required>"), ("x", "<required>"), ("residual", "None")),
        origin="vllm.model_executor.layers.layernorm.RMSNorm.forward_cuda",
        strict=strict,
    )


def install_norm_patch() -> tuple[SymbolResult, ...]:
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("RMSNorm parity patch installed twice")
    from vllm.model_executor.layers.layernorm import RMSNorm

    from .providers import rms_norm

    original = RMSNorm.forward_cuda

    def forward_cuda(self, x, residual=None):
        if self.variance_size_override is not None:
            raise RuntimeError("parity RMSNorm does not support variance_size_override")
        if residual is not None:
            added = x + residual
            return rms_norm(added, self.weight.data, self.variance_epsilon), added
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    forward_cuda._unirl_parity_original = original
    RMSNorm.forward_cuda = forward_cuda
    _INSTALLED = True
    return (
        symbol_result(
            "vllm.model_executor.layers.layernorm.RMSNorm.forward_cuda",
            forward_cuda,
            before=original,
            actual=RMSNorm.forward_cuda,
        ),
    )


__all__ = ["install_norm_patch", "preflight_norm_patch"]
