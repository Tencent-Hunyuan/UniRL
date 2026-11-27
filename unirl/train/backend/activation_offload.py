"""Parameter-storage-aware CPU activation offload for FSDP2 and VeOmni blocks."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import ActivationWrapper
from torch.distributed.tensor import DTensor


def _local_storage_ptr(tensor: torch.Tensor) -> int:
    """Return the underlying non-empty storage address for a Tensor or DTensor."""
    local = tensor._local_tensor if isinstance(tensor, DTensor) else tensor
    if local.is_meta:
        return 0
    storage = local.untyped_storage()
    return storage.data_ptr() if storage.nbytes() > 0 else 0


@contextmanager
def activation_offload_context(module: nn.Module) -> Iterator[None]:
    """Offload CUDA activations saved during ``module`` forward without copying parameter views."""
    param_ptrs = {ptr for p in module.parameters() if (ptr := _local_storage_ptr(p)) != 0}
    is_pinnable = torch.cuda.is_available()

    def _pack(tensor: torch.Tensor) -> tuple[str, Any]:
        if (
            not tensor.is_cuda
            or isinstance(tensor, (nn.Parameter, DTensor))
            or _local_storage_ptr(tensor) in param_ptrs
        ):
            return ("keep", tensor)
        pin = is_pinnable and not tensor.is_sparse
        packed = torch.empty(tensor.size(), dtype=tensor.dtype, layout=tensor.layout, pin_memory=pin)
        packed.copy_(tensor, non_blocking=pin)
        return ("cpu", (tensor.device, packed))

    def _unpack(state: tuple[str, Any]) -> torch.Tensor:
        kind, payload = state
        if kind == "keep":
            return payload
        device, packed = payload
        return packed.to(device, non_blocking=is_pinnable)

    with torch.autograd.graph.saved_tensors_hooks(_pack, _unpack):
        yield


class ParameterAwareOffloadWrapper(ActivationWrapper):
    """``ActivationWrapper`` that offloads saved CUDA activations while skipping parameter views."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped module under ``activation_offload_context``."""
        with activation_offload_context(self._checkpoint_wrapped_module):
            return self._checkpoint_wrapped_module(*args, **kwargs)


def parameter_aware_offload_wrapper(module: nn.Module) -> nn.Module:
    """Wrap ``module`` so saved CUDA activations offload to pinned CPU memory."""
    return ParameterAwareOffloadWrapper(module)


__all__ = [
    "ParameterAwareOffloadWrapper",
    "activation_offload_context",
    "parameter_aware_offload_wrapper",
]
