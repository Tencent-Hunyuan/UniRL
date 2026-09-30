"""Irreversible ATen registration for a dedicated parity-training process."""

from __future__ import annotations

import torch

from .context import require_parity_opt_in
from .linear_ops import linear_backward_cuda, linear_cuda

_LIBRARY = None


def _cuda_registration(operator: str) -> str | None:
    return next(
        (line for line in torch._C._dispatch_dump_table(operator).splitlines() if line.startswith("CUDA:")),
        None,
    )


def preflight_aten() -> None:
    """Reject another extension's process-global CUDA overrides."""
    conflicts = {}
    for operator in ("aten::linear", "aten::linear_backward"):
        registration = _cuda_registration(operator)
        if registration is not None and not any(
            marker in registration for marker in ("/pytorch/", "site-packages/torch/")
        ):
            conflicts[operator] = registration
    if conflicts:
        raise RuntimeError(f"unexpected pre-existing CUDA linear providers: {conflicts}")


def register_aten() -> None:
    """Register guarded linear kernels for the lifetime of this process."""
    global _LIBRARY
    require_parity_opt_in()
    if _LIBRARY is not None:
        return
    preflight_aten()
    library = torch.library.Library("aten", "IMPL")
    library.impl("aten::linear", linear_cuda, "CUDA")
    library.impl("aten::linear_backward", linear_backward_cuda, "CUDA")
    _LIBRARY = library


def aten_registered() -> bool:
    return _LIBRARY is not None


__all__ = ["aten_registered", "preflight_aten", "register_aten"]
