"""Request-side prompt helpers (pure)."""

from __future__ import annotations

from typing import Dict, List, Tuple


def deexpand_prompts_from_groups(
    prompts: List[str],
    group_ids: List[str],
) -> Tuple[List[str], int]:
    """Collapse K-expanded prompts back to unique prompts when groups agree.

    Returns ``(unique_prompts, k)`` where ``k`` is the repeat count to set as
    ``num_outputs_per_prompt`` on SGLang (one text-encode pass per group instead
    of K). Falls through to ``(prompts, 1)`` when the structure doesn't admit a
    clean collapse: heterogeneous K per group, mismatched prompt strings within a
    group, or empty groups.
    """
    n = len(prompts)
    if n == 0 or len(group_ids) != n:
        return list(prompts), 1

    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for i, gid in enumerate(group_ids):
        if gid not in groups:
            order.append(gid)
            groups[gid] = []
        groups[gid].append(i)

    if not groups:
        return list(prompts), 1

    k_per_group = {gid: len(idxs) for gid, idxs in groups.items()}
    k_values = set(k_per_group.values())
    if len(k_values) != 1:
        return list(prompts), 1

    k = next(iter(k_values))
    if k <= 1:
        return list(prompts), 1

    unique_prompts: List[str] = []
    for gid in order:
        idxs = groups[gid]
        base = prompts[idxs[0]]
        if any(prompts[i] != base for i in idxs[1:]):
            return list(prompts), 1
        unique_prompts.append(base)
    return unique_prompts, k


__all__ = ["deexpand_prompts_from_groups"]
