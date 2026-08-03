"""FastVideo request/response contract extensions required by UniRL."""

from __future__ import annotations

from dataclasses import dataclass, fields


def patch_contracts() -> None:
    """Add the SDE-window fields to ``ForwardBatch.RLData`` idempotently."""

    from fastvideo.pipelines.pipeline_batch_info import ForwardBatch

    original = ForwardBatch.RLData
    existing = {field.name for field in fields(original)}
    required = {"sde_step_indices", "sde_type"}
    if required <= existing:
        setattr(original, "_unirl_fastvideo_contract", True)
        return

    missing_base = {"enabled", "collect_log_probs", "store_trajectory"} - existing
    if missing_base:
        raise RuntimeError(f"FastVideo RLData is incompatible; missing base fields {sorted(missing_base)}")

    @dataclass
    class UniRLFastVideoRLData(original):
        """Stock FastVideo RLData plus UniRL's resolved transition contract."""

        sde_step_indices: list[int] | None = None
        sde_type: str = "dance"

    UniRLFastVideoRLData.__name__ = "RLData"
    UniRLFastVideoRLData.__qualname__ = "ForwardBatch.RLData"
    UniRLFastVideoRLData.__module__ = original.__module__
    setattr(UniRLFastVideoRLData, "_unirl_fastvideo_contract", True)
    ForwardBatch.RLData = UniRLFastVideoRLData


__all__ = ["patch_contracts"]
