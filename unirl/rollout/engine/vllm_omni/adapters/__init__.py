"""Adapter registry — importing this package registers all 12 modalities."""

from unirl.rollout.engine.vllm_omni.adapters.bagel import (
    BagelAdapter,
    BagelInputAdapter,
    BagelIt2iAdapter,
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
from unirl.rollout.engine.vllm_omni.adapters.qwen3_omni import (
    Qwen3OmniThinkerAdapter,
    Qwen3OmniThinkerInputAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.qwen_image import (
    QwenImageGroupedInputAdapter,
    QwenImageInputAdapter,
    QwenImageOutputAdapter,
    QwenImageT2iAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3InputAdapter, Sd3OutputAdapter, Sd3T2iAdapter

__all__ = [
    "DitInputAdapter",
    "DitOutputAdapter",
    "BagelAdapter",
    "BagelInputAdapter",
    "BagelIt2iAdapter",
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
    "QwenImageGroupedInputAdapter",
    "Qwen3OmniThinkerAdapter",
    "Qwen3OmniThinkerInputAdapter",
    "QwenImageInputAdapter",
    "QwenImageOutputAdapter",
    "QwenImageT2iAdapter",
    "Sd3InputAdapter",
    "Sd3OutputAdapter",
    "Sd3T2iAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
