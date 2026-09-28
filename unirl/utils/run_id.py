"""Resolve stable, filesystem-safe identifiers shared by a Ray run."""

from __future__ import annotations

import os
import re


def resolve_run_id(explicit: str | None = None) -> str:
    """Return a filesystem-safe id shared by every rank in one run."""
    raw = str(explicit or os.environ.get("UNIRL_RUN_ID") or "").strip() or _ray_job_id()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("._")
    if not safe:
        raise ValueError(f"run id {raw!r} has no filesystem-safe characters.")
    return safe


def _ray_job_id() -> str:
    """Current Ray job id. Recent releases hand back the hex string, older ones a ``JobID``."""
    try:
        import ray

        job_id = ray.get_runtime_context().get_job_id()
    except Exception as exc:
        raise RuntimeError(
            "No run-unique id is available, so stale or concurrent runs would collide. "
            "Set UNIRL_RUN_ID (or pass run_id=...) when there is no Ray job context."
        ) from exc
    as_hex = getattr(job_id, "hex", None)
    return str(as_hex() if callable(as_hex) else job_id)


__all__ = ["resolve_run_id"]
