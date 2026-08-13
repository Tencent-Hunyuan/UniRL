"""Recipe-local VideoAlign reward (Qwen2-VL VQ/MQ/TA scorer)."""

from .scorer import VideoAlignRewardScorer, VideoAlignSpec

__all__ = ["VideoAlignRewardScorer", "VideoAlignSpec"]
