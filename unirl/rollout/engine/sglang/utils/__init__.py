"""Pure helpers the adapter's conversion methods call.

No engine state, no runtime, no I/O — everything here unit-tests with canned
data. The conversion *logic* lives on the base adapter (:mod:`..adapters.text`);
these are the generic mechanics it leans on.
"""

from unirl.rollout.engine.sglang.utils.conditions import pack_prompt_condition
from unirl.rollout.engine.sglang.utils.images import pil_to_base64
from unirl.rollout.engine.sglang.utils.sampling import ResolvedSampling, resolve_sampling
from unirl.rollout.engine.sglang.utils.thinking import split_thinking_tags

__all__ = [
    "ResolvedSampling",
    "pack_prompt_condition",
    "pil_to_base64",
    "resolve_sampling",
    "split_thinking_tags",
]
