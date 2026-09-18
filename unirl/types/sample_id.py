"""Sample-id path grammar — lineage encoded in the id."""

from __future__ import annotations

from typing import Optional


def parent_id(sid: str) -> Optional[str]:
    """The parent id — ``sid`` with its last ``/segment`` stripped; ``None`` for a root (an id with no ``/``)."""
    return sid.rsplit("/", 1)[0] if "/" in sid else None


def _last_segment(sid: str) -> Optional[str]:
    """The trailing lineage segment (after the last ``/``); ``None`` for a root."""
    return sid.rsplit("/", 1)[1] if "/" in sid else None


def branch_of(sid: str) -> Optional[int]:
    """Sibling/branch index from the last segment; ``None`` for a root."""
    seg = _last_segment(sid)
    if seg is None:
        return None
    if not seg.isdigit():
        raise ValueError(f"branch_of: malformed lineage segment {seg!r} in id {sid!r}")
    return int(seg)


def child_id(pid: str, j: int) -> str:
    """A child id: the parent path ``pid`` extended by one ``{j}`` branch segment."""
    return f"{pid}/{j}"


def ancestor_id(sid: str, depth: int) -> str:
    """Id of the ancestor at lineage depth ``depth`` — the id's first ``depth + 1`` segments; fail-loud out of range."""
    segs = sid.split("/")
    if depth < 0 or depth >= len(segs):
        raise ValueError(f"ancestor_id: depth {depth} out of range for id {sid!r} (depth {len(segs) - 1})")
    return "/".join(segs[: depth + 1])


__all__ = [
    "parent_id",
    "branch_of",
    "child_id",
    "ancestor_id",
]
