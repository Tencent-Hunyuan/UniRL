"""Adapter registry — importing this package registers all 12 modalities."""

from unirl.rollout.engine.vllm_omni.adapters.bagel import (
    BagelAdapter,
    BagelIt2iAdapter,
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
    Hi3DitRecaptionAdapter,
    Hi3DitRecaptionOutputAdapter,
    Hi3I2tAdapter,
    Hi3ImageOutputAdapter,
    Hi3InputAdapter,
    Hi3It2iAdapter,
    Hi3T2iAdapter,
    Hi3T2tAdapter,
    Hi3TextOutputAdapter,
)
from unirl.rollout.engine.vllm_omni.adapters.hv15 import Hv15T2vAdapter, Hv15VideoOutputAdapter
from unirl.rollout.engine.vllm_omni.adapters.qwen3_omni import Qwen3OmniThinkerAdapter
from unirl.rollout.engine.vllm_omni.adapters.qwen_image import QwenImageOutputAdapter, QwenImageT2iAdapter
from unirl.rollout.engine.vllm_omni.adapters.sd3 import Sd3OutputAdapter, Sd3T2iAdapter

__all__ = [
    "DitInputAdapter",
    "DitOutputAdapter",
    "BagelAdapter",
    "BagelIt2iAdapter",
    "BagelT2iAdapter",
    "Hi3ArRecaptionAdapter",
    "Hi3DitRecaptionAdapter",
    "Hi3DitRecaptionOutputAdapter",
    "Hi3I2tAdapter",
    "Hi3ImageOutputAdapter",
    "Hi3InputAdapter",
    "Hi3It2iAdapter",
    "Hi3T2iAdapter",
    "Hi3T2tAdapter",
    "Hi3TextOutputAdapter",
    "Hv15T2vAdapter",
    "Hv15VideoOutputAdapter",
    "ModelAdapter",
    "Qwen3OmniThinkerAdapter",
    "QwenImageOutputAdapter",
    "QwenImageT2iAdapter",
    "Sd3OutputAdapter",
    "Sd3T2iAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
