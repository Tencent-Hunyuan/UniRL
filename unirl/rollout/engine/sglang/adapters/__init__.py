"""The adapter package — registry + the two registered families.

Importing this package populates the registry (the ``text`` / ``vlm`` imports
fire the ``@register_adapter`` side-effects); ``config.__post_init__`` validates
``model_family`` against :func:`registered_adapters`.
"""

from unirl.rollout.engine.sglang.adapters.base import (
    MMEncoding,
    ModelAdapter,
    PreparedInputs,
    get_adapter,
    register_adapter,
    registered_adapters,
)
from unirl.rollout.engine.sglang.adapters.text import TextLMAdapter
from unirl.rollout.engine.sglang.adapters.vlm import VLMAdapter

__all__ = [
    "MMEncoding",
    "ModelAdapter",
    "PreparedInputs",
    "TextLMAdapter",
    "VLMAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
