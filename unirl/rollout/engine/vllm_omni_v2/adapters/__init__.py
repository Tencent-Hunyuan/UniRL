"""Adapter registry — importing this package registers all 8 modalities.

The 8 v1 modalities partition into 4 output shapes; 6 of them are
HunyuanImage-3 and live in family-named ``hi3_*`` modules:

- ``hi3_ar_dit``        — t2i, it2i (HI3 two tracks: "ar" root + "image" child)
- ``hi3_ar_only``       — i2t, t2t, ar_recaption (HI3 single "ar" track)
- ``hi3_dit_recaption`` — dit_recaption (HI3 single "image" track;
  subclasses ``DiTImageAdapter``)
- ``dit_image``         — sd35_t2i (single "image" track)
- ``dit_video``         — t2v (single "video" track; the concrete IS the shape)
"""

from unirl.rollout.engine.vllm_omni_v2.adapters.base import (
    ModelAdapter,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.dit_image import (
    DiTImageAdapter,
    Sd35T2iAdapter,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.dit_video import DiTVideoAdapter
from unirl.rollout.engine.vllm_omni_v2.adapters.hi3_ar_dit import (
    Hi3ArDiTAdapter,
    It2iAdapter,
    T2iAdapter,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.hi3_ar_only import (
    ArRecaptionAdapter,
    Hi3ArOnlyAdapter,
    I2tAdapter,
    T2tAdapter,
)
from unirl.rollout.engine.vllm_omni_v2.adapters.hi3_dit_recaption import DitRecaptionAdapter

__all__ = [
    "ArRecaptionAdapter",
    "DiTImageAdapter",
    "DiTVideoAdapter",
    "DitRecaptionAdapter",
    "Hi3ArDiTAdapter",
    "Hi3ArOnlyAdapter",
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
