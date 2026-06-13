"""Registry mapping scorer-name → BaseScorer subclass.

Each concrete scorer module calls `register(name, cls)` at import time.
If a scorer's optional dep is missing its ImportError is reported as a
warning (so users can see why a reward didn't register) but swallowed —
the service only hard-fails when that reward is actually configured.
"""

from __future__ import annotations

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import BaseScorer

logger = get_logger(__name__)

_REGISTRY: dict[str, type[BaseScorer]] = {}


def register(name: str, cls: type[BaseScorer]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"scorer already registered: {name}")
    _REGISTRY[name] = cls


def get_scorer_cls(name: str) -> type[BaseScorer]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown scorer: {name}. available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available_scorers() -> list[str]:
    return sorted(_REGISTRY)


def _try_import(module_path: str) -> None:
    try:
        __import__(module_path)
    except ImportError as e:
        logger.warning(
            "scorer module %s not registered (optional dependency missing or broken): %s",
            module_path,
            e,
        )


# Scorer name → module path mapping. Used by ScorerActor to import
# the correct scorer module inside its Ray venv (where the deps exist).
# Main process does NOT import these — they may depend on packages
# (transformers, vllm, hpsv2, ...) that only exist in per-scorer venvs.
SCORER_MODULES: dict[str, str] = {
    "clip": "reward_service.scorers.clip",
    "pickscore": "reward_service.scorers.pickscore",
    "imagereward": "reward_service.scorers.imagereward",
    "hpsv2": "reward_service.scorers.hpsv2_scorer",
    "hpsv3": "reward_service.scorers.hpsv3_scorer",
    "unified_reward": "reward_service.scorers.unified_reward",
    "geneval2": "reward_service.scorers.geneval2",
    "geneval": "reward_service.scorers.geneval",
    "ocr": "reward_service.scorers.ocr",
    "videoalign": "reward_service.scorers.videoalign",
    "wise": "reward_service.scorers.wise",
    "editreward": "reward_service.scorers.editreward",
}
