"""Optional validation of prompt-group boundaries under fixed row chunking."""

from itertools import groupby
from typing import Sequence


def validate_group_chunks(group_ids: Sequence[str], forward_batch_size: int) -> None:
    """Allow whole groups or aligned equal sub-groups without changing chunk sizes."""
    seen = set()
    start = 0
    for group_id, rows in groupby(group_ids):
        if group_id in seen:
            raise ValueError("Group-boundary validation requires contiguous prompt groups.")
        seen.add(group_id)
        size = sum(1 for _ in rows)
        end = start + size
        if size > forward_batch_size:
            aligned = start % forward_batch_size == 0 and size % forward_batch_size == 0
        else:
            aligned = start // forward_batch_size == (end - 1) // forward_batch_size
        if not aligned:
            raise ValueError(
                f"Prompt group {group_id!r} at rows [{start}, {end}) is not aligned with "
                f"forward_batch_size={forward_batch_size}; choose a group-aligned batch size "
                "or disable validate_group_boundaries."
            )
        start = end
