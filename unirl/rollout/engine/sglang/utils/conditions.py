"""Prompt-condition packing — the right-pad mechanics ``build_conditions`` calls."""

from __future__ import annotations

from typing import List, Optional

import torch

from unirl.types.conditions import TextTokenCondition


def pack_prompt_condition(
    per_sample_prompt_ids: List[List[int]],
    *,
    pad_token_id: int,
) -> Optional[TextTokenCondition]:
    """Pack per-sample prompt token ids into a :class:`TextTokenCondition`.

    Right-padded to the in-batch max with ``pad_token_id``; ``attention_mask``
    zeros out the pad positions. This is what the AR replay consumes at train
    time to teacher-force over ``prompt + response``. Returns ``None`` when no
    sample carries prompt ids.
    """
    if not any(per_sample_prompt_ids):
        return None
    max_plen = max(len(p) for p in per_sample_prompt_ids)
    batch = len(per_sample_prompt_ids)
    input_ids = torch.full((batch, max_plen), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((batch, max_plen), dtype=torch.long)
    for i, p in enumerate(per_sample_prompt_ids):
        n_real = len(p)
        if n_real > 0:
            input_ids[i, :n_real] = torch.tensor(p, dtype=torch.long)
            attention_mask[i, :n_real] = 1
    return TextTokenCondition(input_ids=input_ids, attention_mask=attention_mask)


__all__ = ["pack_prompt_condition"]
