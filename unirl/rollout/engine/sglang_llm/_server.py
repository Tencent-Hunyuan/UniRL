"""Process and HTTP helpers for the SGLang LLM engine.

Pulled out of :mod:`unirl.rollout.engine.sglang_llm.engine` so the
engine class stays focused on the typed ``generate(req)`` path and the
weight-sync forwards. These helpers have no engine state and are
import-cheap.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Bind-to-zero free port (OS-assigned, ephemeral range).

    sglang server / router need a port deterministically close to the
    bind moment; ``ray/utils/net.get_free_port`` does range-scan from
    10000+ instead, more useful for stable rendezvous ports.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def kill_process_tree(pid: int) -> None:
    """Send SIGTERM to ``pid`` and its descendants."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def wait_router_ready(
    router_url: str,
    *,
    timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
    process: Optional[multiprocessing.Process] = None,
) -> None:
    """Poll router until it accepts HTTP connections or timeout."""
    deadline = time.monotonic() + timeout_s
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        if process is not None and not process.is_alive():
            raise RuntimeError("sglang router process died during startup")
        try:
            req = urllib.request.Request(f"{router_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                return
        except urllib.error.HTTPError as exc:
            # Anything other than 503 means the server bound the port and
            # is responding; that's enough for "router is up".
            if exc.code != 503:
                return
            last_error = exc
        except Exception as exc:
            last_error = exc
            logger.debug("SGLang router health check failed for %s.", router_url, exc_info=True)
        time.sleep(poll_interval_s)
    detail = f"; last error: {last_error!r}" if last_error is not None else ""
    raise TimeoutError(f"sglang router at {router_url} not ready within {timeout_s}s{detail}")


def wait_server_healthy(
    base_url: str,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
    is_alive_fn: Optional[Callable[[], bool]] = None,
) -> None:
    """Poll server ``/health_generate`` until 200 OK or timeout."""
    deadline = time.monotonic() + timeout_s
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health_generate", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            logger.debug("SGLang server health check failed for %s.", base_url, exc_info=True)
        if is_alive_fn is not None and not is_alive_fn():
            raise RuntimeError("SGLang SRT server process terminated unexpectedly.")
        time.sleep(poll_interval_s)
    detail = f"; last error: {last_error!r}" if last_error is not None else ""
    raise TimeoutError(f"SGLang SRT server at {base_url} did not become healthy within {timeout_s}s{detail}")


def run_router_process(router_args: Any) -> None:
    """Entry point for the sglang router daemon process.

    Must be module-level (not a closure) so ``multiprocessing.Process``
    can pickle it on the spawn-start-method path.
    """
    from sglang_router.launch_router import launch_router

    launch_router(router_args)


__all__ = [
    "find_free_port",
    "kill_process_tree",
    "wait_router_ready",
    "wait_server_healthy",
    "run_router_process",
]
