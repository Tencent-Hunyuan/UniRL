"""Adapter registry — importing this package registers all 8 modalities.

The 8 v1 modalities partition into 4 output shapes:

- ``ar_dit``    — t2i, it2i (two tracks: "ar" root + "image" child)
- ``ar_only``   — i2t, t2t, ar_recaption (single "ar" track)
- ``dit_image`` — sd35_t2i, dit_recaption (single "image" track)
- ``dit_video`` — t2v (single "video" track; the concrete IS the shape)
"""

from unirl.rollout.engine.vllm_omni_v2.adapters.ar_dit import (
    ArDiTAdapter,
    It2iAdapter,
    T2iAdapter,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.ar_only import (
    ArOnlyAdapter,
    ArRecaptionAdapter,
    I2tAdapter,
    T2tAdapter,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.base import (
    ModelAdapter,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.dit_image import (
    DiTImageAdapter,
    DitRecaptionAdapter,
    Sd35T2iAdapter,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.dit_video import DiTVideoAdapter

__all__ = [
    "ArDiTAdapter",
    "ArOnlyAdapter",
    "ArRecaptionAdapter",
    "DiTImageAdapter",
    "DiTVideoAdapter",
    "DitRecaptionAdapter",
    "I2tAdapter",
    "It2iAdapter",
    "ModelAdapter",
    "Sd35T2iAdapter",
    "T2iAdapter",
    "T2tAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
