"""Prompt-only track construction for Self-Forcing video distillation."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import Texts
from unirl.types.sample import Part


class WAN21SelfForcingPromptTrackBuilder(Remote):
    """Encode UCF-style prompts without decoding or VAE-encoding target videos."""

    def __init__(self, *, pipeline: Any, real_guidance_scale: float = 3.0) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.real_guidance_scale = float(real_guidance_scale)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def build(self, records: List[Dict[str, Any]]) -> Part:
        if not records:
            raise ValueError("WAN21SelfForcingPromptTrackBuilder.build: empty record shard.")
        texts = Texts(texts=[str(record["prompt"]) for record in records])
        with torch.no_grad():
            conditions = self.pipeline.build_conditions(
                texts,
                guidance_scale=self.real_guidance_scale,
            )
        return Part(
            sample_ids=[
                str(record.get("sample_id", f"self-forcing:{index}"))
                for index, record in enumerate(records)
            ],
            conditions=conditions.to_dict(),
            metadata=[dict(record.get("metadata") or {}) for record in records],
        )


__all__ = ["WAN21SelfForcingPromptTrackBuilder"]
