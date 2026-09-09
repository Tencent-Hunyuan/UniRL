"""Pure helpers the adapter's conversion methods call."""

from unirl.rollout.engine.sglang.utils.conditions import pack_prompt_condition
from unirl.rollout.engine.sglang.utils.conversations import (
    build_text_conversations,
    build_vision_conversations,
    unique_group_indices,
)
from unirl.rollout.engine.sglang.utils.images import pil_to_base64
from unirl.rollout.engine.sglang.utils.sampling import ResolvedSampling, resolve_sampling
from unirl.rollout.engine.sglang.utils.thinking import split_thinking_tags

__all__ = [
    "ResolvedSampling",
    "build_text_conversations",
    "build_vision_conversations",
    "pack_prompt_condition",
    "pil_to_base64",
    "resolve_sampling",
    "split_thinking_tags",
    "unique_group_indices",
]
