"""Built-in reward scorer implementations."""

from .aesthetic import AestheticRewardScorer
from .base import LocalRewardBackend
from .clap import CLAPRewardScorer
from .clip import ClipRewardScorer
from .geneval2 import GenEval2RewardScorer
from .hpsv2 import HPSv2RewardScorer
from .hpsv3 import HPSv3RewardScorer
from .hpsv3pp import HPSv3PPRewardScorer
from .image_reward import ImageRewardScorer
from .imagebind import ImageBindRewardScorer
from .mc_exact_match import MCExactMatchRewardScorer
from .ocr import OCRRewardScorer
from .pickscore import PickScoreRewardScorer
from .registry import available_builtin_reward_models, resolve_builtin_reward_scorer_class
from .t2av_composite import T2AVCompositeScorer
from .video import VideoRewardScorer
from .video_clip_delta import VideoCLIPDeltaScorer
from .video_pickscore import VideoPickScoreScorer
from .videoalign import VideoAlignRewardScorer

__all__ = [
    "AestheticRewardScorer",
    "LocalRewardBackend",
    "CLAPRewardScorer",
    "ClipRewardScorer",
    "GenEval2RewardScorer",
    "HPSv2RewardScorer",
    "HPSv3RewardScorer",
    "HPSv3PPRewardScorer",
    "ImageRewardScorer",
    "ImageBindRewardScorer",
    "MCExactMatchRewardScorer",
    "OCRRewardScorer",
    "PickScoreRewardScorer",
    "T2AVCompositeScorer",
    "VideoCLIPDeltaScorer",
    "VideoAlignRewardScorer",
    "VideoPickScoreScorer",
    "VideoRewardScorer",
    "available_builtin_reward_models",
    "resolve_builtin_reward_scorer_class",
]
