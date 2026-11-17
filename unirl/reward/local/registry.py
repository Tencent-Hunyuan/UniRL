"""Built-in reward scorer / spec registry."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple, Type

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend

_BUILTIN_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "aesthetic": ("unirl.reward.local.aesthetic", "AestheticRewardScorer", "AestheticSpec"),
    "clip": ("unirl.reward.local.clip", "ClipRewardScorer", "ClipSpec"),
    "hpsv2": ("unirl.reward.local.hpsv2", "HPSv2RewardScorer", "HPSv2Spec"),
    "ocr": ("unirl.reward.local.ocr", "OCRRewardScorer", "OCRSpec"),
    "pickscore": ("unirl.reward.local.pickscore", "PickScoreRewardScorer", "PickScoreSpec"),
    "geneval2": ("unirl.reward.local.geneval2", "GenEval2RewardScorer", "GenEval2Spec"),
    "hpsv3": ("unirl.reward.local.hpsv3", "HPSv3RewardScorer", "HPSv3Spec"),
    "hpsv3pp": ("unirl.reward.local.hpsv3pp", "HPSv3PPRewardScorer", "HPSv3PPSpec"),
    "image_reward": ("unirl.reward.local.image_reward", "ImageRewardScorer", "ImageRewardSpec"),
    "videopickscore": ("unirl.reward.local.video_pickscore", "VideoPickScoreScorer", "VideoPickScoreSpec"),
    "videoclipdelta": ("unirl.reward.local.video_clip_delta", "VideoCLIPDeltaScorer", "VideoCLIPDeltaSpec"),
    "videoalign": ("unirl.reward.local.videoalign", "VideoAlignRewardScorer", "VideoAlignSpec"),
    "mc_exact_match": ("unirl.reward.local.mc_exact_match", "MCExactMatchRewardScorer", "MCExactMatchSpec"),
    "clap": ("unirl.reward.local.clap", "CLAPRewardScorer", "CLAPSpec"),
    "imagebind": ("unirl.reward.local.imagebind", "ImageBindRewardScorer", "ImageBindSpec"),
    "t2av_composite": ("unirl.reward.local.t2av_composite", "T2AVCompositeScorer", "T2AVCompositeSpec"),
    "per_domain": ("unirl.reward.local.per_domain", "PerDomainRewardScorer", "PerDomainSpec"),
}


def available_builtin_reward_models() -> Tuple[str, ...]:
    """Return built-in reward model names."""
    return tuple(sorted(_BUILTIN_REGISTRY.keys()))


def _resolve_builtin_entry(model_name: str) -> Tuple[str, str, str]:
    key = "" if model_name is None else str(model_name).strip().lower()
    if key not in _BUILTIN_REGISTRY:
        raise ValueError(f"Unknown model_name: {model_name}. Available: {list(available_builtin_reward_models())}")
    return _BUILTIN_REGISTRY[key]


def resolve_builtin_reward_scorer_class(model_name: str) -> Type[RewardBackend]:
    """Resolve a built-in reward scorer class by canonical model name."""
    module_name, scorer_attr, _ = _resolve_builtin_entry(model_name)
    module = importlib.import_module(module_name)
    scorer_cls = getattr(module, scorer_attr)
    if not isinstance(scorer_cls, type) or not issubclass(scorer_cls, RewardBackend):
        raise TypeError(f"Configured scorer {module_name}.{scorer_attr} is not a RewardBackend.")
    return scorer_cls


def resolve_builtin_reward_spec_class(model_name: str) -> Type[BaseRewardComponentSpec]:
    """Resolve a built-in reward Spec class by canonical model name."""
    module_name, _, spec_attr = _resolve_builtin_entry(model_name)
    module = importlib.import_module(module_name)
    spec_cls = getattr(module, spec_attr)
    if not isinstance(spec_cls, type) or not issubclass(spec_cls, BaseRewardComponentSpec):
        raise TypeError(f"Configured spec {module_name}.{spec_attr} is not a BaseRewardComponentSpec.")
    return spec_cls


__all__ = [
    "available_builtin_reward_models",
    "resolve_builtin_reward_scorer_class",
    "resolve_builtin_reward_spec_class",
]
