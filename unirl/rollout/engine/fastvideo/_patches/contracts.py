"""Extend FastVideo's ``ForwardBatch.RLData`` with the SDE-window fields UniRL resolves engine-side."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional


def patch_contracts() -> None:
    """Add UniRL's resolved SDE-window fields to ``ForwardBatch.RLData`` idempotently."""
    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    original = ForwardBatch.RLData
    existing = {field.name for field in fields(original)}
    if {"sde_step_indices", "sde_type"} <= existing:
        original._unirl_fastvideo_contract = True
        return

    missing_base = {"enabled", "collect_log_probs", "store_trajectory", "keep_trajectory_on_cpu"} - existing
    if missing_base:
        raise RuntimeError(f"FastVideo RLData is incompatible; missing base fields {sorted(missing_base)}")

    @dataclass
    class UniRLFastVideoRLData(original):  # type: ignore[valid-type,misc]
        """Stock FastVideo RLData plus UniRL's resolved transition contract."""

        sde_step_indices: Optional[list[int]] = None
        sde_type: Any = "dance"

    UniRLFastVideoRLData.__name__ = "RLData"
    UniRLFastVideoRLData.__qualname__ = "ForwardBatch.RLData"
    UniRLFastVideoRLData.__module__ = original.__module__
    UniRLFastVideoRLData._unirl_fastvideo_contract = True
    ForwardBatch.RLData = UniRLFastVideoRLData


__all__ = ["patch_contracts"]
