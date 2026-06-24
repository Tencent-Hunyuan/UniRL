"""Adapter registry — importing this package registers all 9 modalities.

Modality adapters are grouped by model family and composed from input/output
sub-adapters (the binder constructs both in ``__init__`` and delegates the
two conversion verbs):

- ``hi3``  — hi3_t2i, hi3_it2i, hi3_i2t, hi3_t2t, hi3_ar_recaption, hi3_dit_recaption
- ``sd3``  — sd3_t2i
- ``hv15`` — hv15_t2v
- ``qwen_image`` — qwen_image_t2i
- ``bagel`` — bagel_t2i

``dit`` holds the universal single-stage DiT skeletons
(:class:`DitInputAdapter` / :class:`DitOutputAdapter`) the families derive
from; family-specific sub-adapters carry the family prefix and live in the
family file.
"""

from unirl.rollout.engine.vllm_omni.adapters.bagel import (
    BagelInputAdapter,
    BagelOutputAdapter,
    BagelT2iAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.base import (
    ModelAdapter,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.vllm_omni.adapters.dit import DitInputAdapter, DitOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.hi3 import (
    Hi3ArRecaptionAdapter,
    Hi3ArRecaptionOutputAdapter,
    Hi3DitRecaptionAdapter,
    Hi3DitRecaptionInputAdapter,
    Hi3DitRecaptionOutputAdapter,
    Hi3I2tAdapter,
    Hi3ImageOutputAdapter,
    Hi3InputAdapter,
    Hi3It2iAdapter,
    Hi3T2iAdapter,
    Hi3T2tAdapter,
    Hi3TextOutputAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.hv15 import (
    Hv15InputAdapter,
    Hv15T2vAdapter,
    Hv15VideoOutputAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.qwen_image import (
    QwenImageInputAdapter,
    QwenImageOutputAdapter,
    QwenImageT2iAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3OutputAdapter, Sd3T2iAdapter

__all__ = [
    "DitInputAdapter",
    "DitOutputAdapter",
    "BagelInputAdapter",
    "BagelOutputAdapter",
    "BagelT2iAdapter",
    "Hi3ArRecaptionAdapter",
    "Hi3ArRecaptionOutputAdapter",
    "Hi3DitRecaptionAdapter",
    "Hi3DitRecaptionInputAdapter",
    "Hi3DitRecaptionOutputAdapter",
    "Hi3I2tAdapter",
    "Hi3ImageOutputAdapter",
    "Hi3InputAdapter",
    "Hi3It2iAdapter",
    "Hi3T2iAdapter",
    "Hi3T2tAdapter",
    "Hi3TextOutputAdapter",
    "Hv15InputAdapter",
    "Hv15T2vAdapter",
    "Hv15VideoOutputAdapter",
    "ModelAdapter",
    "QwenImageInputAdapter",
    "QwenImageOutputAdapter",
    "QwenImageT2iAdapter",
    "Sd3OutputAdapter",
    "Sd3T2iAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
