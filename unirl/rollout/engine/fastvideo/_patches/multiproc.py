"""Propagate the UniRL patch suite into FastVideo's spawn-started workers."""

from __future__ import annotations

from unirl.rollout.engine.fastvideo._patches.compat import import_fastvideo_module, require_attr


def _worker_main_with_patches(*args, **kwargs):
    """Spawn-safe FastVideo worker entrypoint that installs the runtime patches first."""
    from unirl.rollout.engine.fastvideo._patches.hijack import patch_worker_runtime

    patch_worker_runtime()
    from fastvideo.worker.multiproc_executor import WorkerMultiprocProc

    original = getattr(WorkerMultiprocProc, "_unirl_original_worker_main", WorkerMultiprocProc.worker_main)
    if original is _worker_main_with_patches:
        raise RuntimeError("FastVideo worker entrypoint patch lost the original worker_main")
    return original(*args, **kwargs)


def patch_multiproc() -> None:
    """Replace ``WorkerMultiprocProc.worker_main`` so each spawn child re-installs the patches."""
    module = import_fastvideo_module("fastvideo.worker.multiproc_executor", "MultiprocExecutor workers")
    WorkerMultiprocProc = require_attr(module, "WorkerMultiprocProc", "MultiprocExecutor workers")

    current = WorkerMultiprocProc.worker_main
    if current is _worker_main_with_patches:
        return
    WorkerMultiprocProc._unirl_original_worker_main = current
    WorkerMultiprocProc.worker_main = staticmethod(_worker_main_with_patches)


__all__ = ["patch_multiproc"]
