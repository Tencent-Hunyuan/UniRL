"""Propagate patches into workers and recover TCPStore startup collisions."""

from __future__ import annotations

import contextlib
import faulthandler
import logging
import signal
from typing import Any, Callable

_MAX_PORT_ATTEMPTS = 5
_ORIGINAL_INIT_EXECUTOR: Callable[..., Any] | None = None


def _patched_worker_main(*args: Any, **kwargs: Any) -> Any:
    from unirl.rollout.engine.fastvideo._patches.hijack import FastVideoHijack

    FastVideoHijack.hijack()

    import psutil
    from fastvideo.worker import multiproc_executor as module

    log_queue = kwargs.pop("log_queue", None)
    if log_queue is not None:
        handler = module._make_queue_log_handler(log_queue)
        logging.getLogger("fastvideo").addHandler(handler)
        kwargs["_initial_log_handler"] = handler

    shutdown_requested = False

    def signal_handler(signum, frame):
        del signum, frame
        nonlocal shutdown_requested
        if not shutdown_requested:
            shutdown_requested = True
            raise SystemExit()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    module.kill_itself_when_parent_died()
    faulthandler.enable()
    parent_process = psutil.Process().parent()

    worker = None
    ready_pipe = kwargs.pop("ready_pipe")
    rank = kwargs.get("rank")
    try:
        worker = module.WorkerMultiprocProc(*args, **kwargs)
        ready_pipe.send({"status": module.WorkerMultiprocProc.READY_STR})
        ready_pipe.close()
        ready_pipe = None
        worker.worker_busy_loop()
    except Exception as exc:  # noqa: BLE001
        if ready_pipe is not None:
            module.logger.exception("WorkerMultiprocProc failed to start.")
            with contextlib.suppress(Exception):
                ready_pipe.send(
                    {
                        "status": "ERROR",
                        "error": str(exc),
                        "traceback": module.get_exception_traceback(),
                        "rank": rank,
                    }
                )
        else:
            module.logger.exception("WorkerMultiprocProc failed.")
            if parent_process:
                parent_process.send_signal(signal.SIGQUIT)
        shutdown_requested = True
        module.logger.error("Worker %s hit an exception: %s", rank, module.get_exception_traceback())
    finally:
        if ready_pipe is not None:
            ready_pipe.close()
        if worker is not None:
            worker.shutdown()


setattr(_patched_worker_main, "_unirl_fastvideo_spawn", True)


def _patched_init_executor(self) -> None:
    if _ORIGINAL_INIT_EXECUTOR is None or _ORIGINAL_INIT_EXECUTOR is _patched_init_executor:
        raise RuntimeError("FastVideo executor retry patch could not resolve stock _init_executor")

    for attempt in range(1, _MAX_PORT_ATTEMPTS + 1):
        try:
            _ORIGINAL_INIT_EXECUTOR(self)
            return
        except Exception as exc:  # noqa: BLE001
            message = str(exc).lower()
            address_in_use = "eaddrinuse" in message or "address already in use" in message
            if not address_in_use or attempt == _MAX_PORT_ATTEMPTS:
                raise
            self.fastvideo_args.master_port = None


setattr(_patched_init_executor, "_unirl_fastvideo_port_retry", True)


def patch_multiproc() -> None:
    """Make spawn initialization errors recoverable and retry fresh ports."""

    global _ORIGINAL_INIT_EXECUTOR

    from fastvideo.worker.multiproc_executor import MultiprocExecutor, WorkerMultiprocProc

    current_worker_main = WorkerMultiprocProc.worker_main
    if not getattr(current_worker_main, "_unirl_fastvideo_spawn", False):
        WorkerMultiprocProc.worker_main = staticmethod(_patched_worker_main)

    current_init = MultiprocExecutor._init_executor
    if not getattr(current_init, "_unirl_fastvideo_port_retry", False):
        _ORIGINAL_INIT_EXECUTOR = current_init
        MultiprocExecutor._init_executor = _patched_init_executor


__all__ = ["patch_multiproc"]
