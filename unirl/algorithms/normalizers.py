"""Group-id helpers for advantage normalization scopes."""

from typing import Any, Dict, List, Optional


def _normalize_group_id(group_id: Any) -> Optional[str]:
    if group_id is None:
        return None
    text = str(group_id).strip()
    return text if text else None


def build_group_index_map(group_ids: List[str]) -> Dict[str, List[int]]:
    ordered_groups: Dict[str, List[int]] = {}
    for sample_idx, raw_group_id in enumerate(group_ids):
        gid = _normalize_group_id(raw_group_id)
        if gid is None:
            continue
        ordered_groups.setdefault(gid, []).append(sample_idx)
    return ordered_groups
