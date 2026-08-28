"""SenseNova-U1.5 pipeline extensions for vLLM-Omni rollout."""

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "RLSenseNovaU1Pipeline":
        from unirl.rollout.engine.vllm_omni.pipelines.sensenova_u1.pipeline import (
            RLSenseNovaU1Pipeline,
        )

        return RLSenseNovaU1Pipeline
    raise AttributeError(name)


__all__ = ["RLSenseNovaU1Pipeline"]
