"""Model adapters for the ``sglang_diffusion`` engine.

Importing this package registers every concrete adapter (the ``@register_adapter``
side-effects fire), so ``get_adapter(model_family)`` resolves after import.
"""

from unirl.rollout.engine.sglang_diffusion.adapters.base import (
    ModelAdapter,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.sglang_diffusion.adapters.image_dit import ImageDiTAdapter

# Concrete adapters — imported for their registration side-effects.
from unirl.rollout.engine.sglang_diffusion.adapters.sd3 import SD3Adapter
from unirl.rollout.engine.sglang_diffusion.adapters.flux import (
    FluxAdapter,
    Flux2KleinAdapter,
)
from unirl.rollout.engine.sglang_diffusion.adapters.video import (
    HunyuanVideoAdapter,
    MochiAdapter,
)

__all__ = [
    "ModelAdapter",
    "ImageDiTAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
    "SD3Adapter",
    "FluxAdapter",
    "Flux2KleinAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
]
