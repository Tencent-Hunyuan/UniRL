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
from unirl.rollout.engine.sglang_diffusion.adapters.flux import (
    Flux2KleinAdapter,
    FluxAdapter,
)
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter

# Concrete adapters — imported for their registration side-effects.
from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image import QwenImageAdapter
from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image_edit_plus import (
    QwenImageEditPlusAdapter,
)
from unirl.rollout.engine.sglang_diffusion.adapters.sd3 import SD3Adapter
from unirl.rollout.engine.sglang_diffusion.adapters.video import (
    HunyuanVideoAdapter,
    MochiAdapter,
    VideoAdapter,
)
from unirl.rollout.engine.sglang_diffusion.adapters.z_image import ZImageAdapter

__all__ = [
    "ModelAdapter",
    "ImageAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
    "SD3Adapter",
    "FluxAdapter",
    "Flux2KleinAdapter",
    "QwenImageAdapter",
    "VideoAdapter",
    "QwenImageEditPlusAdapter",
    "MochiAdapter",
    "HunyuanVideoAdapter",
    "ZImageAdapter",
]
