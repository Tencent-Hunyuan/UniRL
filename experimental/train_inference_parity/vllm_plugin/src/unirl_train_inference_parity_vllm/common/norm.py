"""Route vLLM RMSNorm layers to the public BI primitive."""

from __future__ import annotations

_INSTALLED = False


def install_norm_patch() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.model_executor.layers.layernorm import RMSNorm

    from .providers import rms_norm

    original = RMSNorm.forward_cuda

    def forward_cuda(self, x, residual=None):
        if self.variance_size_override is not None:
            return self.forward_native(x, residual)
        if residual is not None:
            added = x + residual
            return rms_norm(added, self.weight.data, self.variance_epsilon), added
        return rms_norm(x, self.weight.data, self.variance_epsilon)

    forward_cuda._unirl_parity_original = original
    RMSNorm.forward_cuda = forward_cuda
    _INSTALLED = True


__all__ = ["install_norm_patch"]
