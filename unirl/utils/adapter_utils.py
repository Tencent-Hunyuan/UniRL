"""
LoRA adapter utilities.

Provides context managers for switching between LoRA adapters,
compatible with both regular and FSDP-wrapped models.
"""

from contextlib import contextmanager

import torch.nn as nn


@contextmanager
def switch_adapter(model: nn.Module, adapter_name: str):
    """
    Context manager to temporarily switch LoRA adapter.

    Compatible with:
    - Direct PEFT models (model.set_adapter)
    - FSDP-wrapped models (model.module.set_adapter)

    Args:
        model: Model with LoRA adapters
        adapter_name: Name of adapter to switch to

    Yields:
        None (model is temporarily using the specified adapter)
    """
    if hasattr(model, "set_adapter"):
        original_adapter = getattr(model, "active_adapter", "default")
        try:
            model.set_adapter(adapter_name)
            yield
        finally:
            model.set_adapter(original_adapter)
    elif hasattr(model, "module") and hasattr(model.module, "set_adapter"):
        # FSDP-wrapped model
        original_adapter = getattr(model.module, "active_adapter", "default")
        try:
            model.module.set_adapter(adapter_name)
            yield
        finally:
            model.module.set_adapter(original_adapter)
    else:
        # No adapter support, just yield (no-op)
        yield
