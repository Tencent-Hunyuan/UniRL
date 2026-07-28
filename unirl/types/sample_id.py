"""Sample-id path grammar — lineage encoded in the id (LIN-446).

See ``unirl/types/README.md``. A sample id is a ``/``-delimited path: a root id
(no ``/``) plus one ``{branch}`` segment per fork — e.g. ``"p0/1/2"`` (root ``p0``;
branch 1; then branch 2). The path *is* the lineage: a part's parent id is the child
id with its last segment stripped, so lineage is recovered by id (position-
independent) and survives split/concat/balance with no remapping — unlike a
positional parent index.

These helpers are the str-based id API (no ``SampleId`` class). Parsing is fail-loud:
a malformed segment raises rather than returning a wrong value, since the point of
the path scheme is to harden lineage against silent corruption. Kept torch-free so
the grammar can be exercised in isolation.
"""

from __future__ import annotations

from typing import Optional


def parent_id(sid: str) -> Optional[str]:
    """The parent id — ``sid`` with its last ``/segment`` stripped; ``None`` for a
    root (an id with no ``/``)."""
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
    """Id of the ancestor at lineage depth ``depth`` — the id's first
    ``depth + 1`` segments (``0`` = root; a sample's own depth returns the id
    itself). Fail-loud when ``depth`` is negative or exceeds the sample's depth."""
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
