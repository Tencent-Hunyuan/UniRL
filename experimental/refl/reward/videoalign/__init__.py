"""Recipe-local VideoAlign reward (Qwen2-VL VQ/MQ/TA scorer).

Self-contained port of the VideoAlign reward family for the REFL WAN recipe.
The reward model code (``Qwen2VLRewardModelBT``), prompt templates, config
dataclasses and checkpoint-loading helpers all live under
:mod:`experimental.refl.reward.videoalign.model`, matching the recipe-local
layout used by the WAN22 face reward.

Public API
----------
- :class:`VideoAlignRewardScorer` — REFL-compatible reward backend.
- :class:`VideoAlignSpec`         — typed config dataclass.
"""

from .scorer import VideoAlignRewardScorer, VideoAlignSpec

__all__ = ["VideoAlignRewardScorer", "VideoAlignSpec"]
