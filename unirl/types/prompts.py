from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.rollout_req import PrimitiveValue


@dataclass
class RolloutInputs(Batch):
    primitives: Dict[str, PrimitiveValue] = concat_field(default_factory=dict)
    sample_ids: List[str] = concat_field(default_factory=list)
    group_ids: List[str] = concat_field(default_factory=list)
    metadata: List[Optional[Dict[str, Any]]] = concat_field(default_factory=list)

    def expand(self, samples_per_prompt: int) -> RolloutInputs:
        """Expand each prompt into samples_per_prompt entries."""
        k = int(samples_per_prompt)
        if k < 1:
            raise ValueError(f"samples_per_prompt must be >= 1, got {k}")
        if k == 1:
            return self

        expanded = self.repeat_interleave(k)
        expanded.sample_ids = [f"prompt:{gid}:sample:{j}" for gid in self.group_ids for j in range(k)]
        return expanded
