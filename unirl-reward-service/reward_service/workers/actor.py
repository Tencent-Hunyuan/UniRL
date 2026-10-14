"""Ray actor wrapping a single scorer instance.

Each actor owns its GPU(s) exclusively (num_gpus set at actor options
from the reward config). The actor is a thin forwarder: it instantiates
the scorer on first use, then delegates score() calls. Scorer classes
live outside Ray and are pickled by name to avoid serializing heavy
model state.
"""

from __future__ import annotations

import os
from typing import Any

import ray

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import ScoreItem
from reward_service.scorers.registry import SCORER_MODULES, _try_import, get_scorer_cls

logger = get_logger(__name__)

# Packages logged at actor startup to verify venv isolation.
_VENV_PROBE_PACKAGES = ("transformers", "vllm", "hpsv2", "image_reward")


def _pin_cuda_to_ray_gpus() -> None:
    """Restrict CUDA visibility to the GPUs Ray assigned to this actor.

    Ray's ``num_gpus`` parameter is a *logical* scheduler reservation — it
    does not automatically set ``CUDA_VISIBLE_DEVICES``. Without this call
    the CUDA runtime can still enumerate every physical GPU on the host,
    so any other process releasing memory on an unrelated card can trigger
    vLLM's memory-profiling invariant check (initial_free > current_free).

    Must be called early in the actor ``__init__``, before any CUDA context
    is created (i.e. before importing vLLM or loading a model).
    """
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        return  # CPU-only actor — leave CUDA_VISIBLE_DEVICES untouched
    visible = ",".join(str(g) for g in gpu_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    logger.debug("ScorerActor: pinned CUDA_VISIBLE_DEVICES=%s", visible)


@ray.remote
class ScorerActor:
    def __init__(self, scorer_name: str, params: dict[str, Any]) -> None:
        _pin_cuda_to_ray_gpus()
        self._log_venv_info(scorer_name)
        # Import the scorer module inside the actor process (which has the
        # per-scorer venv with the right dependencies). This triggers the
        # module-level register() call. Main process never imports these.
        module_path = SCORER_MODULES.get(scorer_name)
        if module_path:
            _try_import(module_path)
        cls = get_scorer_cls(scorer_name)
        logger.info("ScorerActor initializing scorer=%s params=%s", scorer_name, params)
        self.scorer = cls(**params)
        self.scorer_name = scorer_name

    @staticmethod
    def _log_venv_info(scorer_name: str) -> None:
        """Log the Python executable and key package versions for debugging
        per-scorer venv isolation."""
        import sys

        is_venv = "/runtime_resources/pip/" in sys.executable
        logger.info(
            "ScorerActor[%s] python=%s venv=%s",
            scorer_name,
            sys.executable,
            is_venv,
        )
        for pkg in _VENV_PROBE_PACKAGES:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                logger.info("ScorerActor[%s]   %s==%s", scorer_name, pkg, ver)
            except ImportError:
                pass

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        return self.scorer.score(items)

    def sub_metric_names(self) -> tuple[str, ...]:
        return tuple(self.scorer.sub_metric_names)

    def ping(self) -> str:
        return f"{self.scorer_name}:ready"
