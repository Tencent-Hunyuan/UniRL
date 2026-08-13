"""VeOmni-specific sharded-state helpers."""

from __future__ import annotations

import logging

import torch
from torch import Tensor, nn

from unirl.train.backend.sharded_state import _maybe_dtensor_to_tensor

logger = logging.getLogger(__name__)


def clip_grad_norm(model: nn.Module, max_norm: float) -> Tensor:
    """Gradient clipping via VeOmni's FSDP2 clip (EP-aware under Phase 2)."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.fsdp2 import clip_grad_norm as _veomni_clip_grad_norm

    result = _veomni_clip_grad_norm(model, max_norm)
    return _maybe_dtensor_to_tensor(result)


def veomni_offload(model: nn.Module) -> None:
    """Move the parallelized model to CPU via VeOmni (reshards the root first)."""
    meta_names = [n for n, p in model.named_parameters() if p.is_meta]
    if meta_names:
        raise RuntimeError(
            f"veomni_offload: {len(meta_names)} params still on meta "
            f"(e.g. {meta_names[:4]}); VeOmniBackend v1 requires a fully-"
            "materialized trainable module."
        )
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.offloading import offload_model_to_cpu

    offload_model_to_cpu(model)
    logger.debug("veomni_offload: offloaded params/grads to CPU")


def veomni_onload(model: nn.Module, device: torch.device) -> None:
    """Move the parallelized model back to ``device`` via VeOmni."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.offloading import load_model_to_gpu

    load_model_to_gpu(model, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    logger.debug("veomni_onload: onloaded params/grads to %s", device)


__all__ = [
    "clip_grad_norm",
    "veomni_offload",
    "veomni_onload",
]
