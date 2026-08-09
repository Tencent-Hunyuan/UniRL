"""Turn abrupt driver termination into an orderly teardown.

A training driver that dies on an unhandled ``SIGTERM`` never runs its
``finally`` blocks, so the rollout engine's subprocess tree is never asked to
shut down. On the anchored vLLM-Omni topology that stranded four ``Worker_TP``
processes holding ~7.3 GiB of device memory each until they were reaped by
hand.

The engine-side fate-sharing watchdog
(:func:`unirl.rollout.engine.vllm_omni.patches.runtime.install_fate_sharing`)
is the hard backstop for that. This module is the *graceful* half: it gives the
driver a chance to shut the engine down properly — releasing KV cache, letting
Ray actors exit cleanly, flushing loggers — instead of relying on the tree
being force-killed.

Three guarantees, in order of preference:

1. A termination signal unwinds the main thread through the existing
   ``finally`` blocks, so teardown runs the way it does on a clean exit.
2. Teardown runs at most once, whichever path reaches it (normal return,
   exception, signal, or ``atexit``).
3. The process exits even if teardown wedges — a hard deadline and a
   second-signal escape hatch both bypass Python entirely.
"""

from __future__ import annotations

import atexit
import functools
import logging
import os
import signal
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_HARD_EXIT_GRACE_SECONDS = 120.0

_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


def _parent_of(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat = handle.read()
    except OSError:
        return None
    try:
        return int(stat[stat.rindex(b")") + 2 :].split()[1])
    except (ValueError, IndexError):
        return None


def _describe(pid: int) -> str:
    for path in (f"/proc/{pid}/cmdline", f"/proc/{pid}/comm"):
        try:
            with open(path, "rb") as handle:
                text = handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        if text:
            return text
    return ""


def descendants_of(root_pid: int) -> List[int]:
    """Transitive children of ``root_pid``, deepest first.

    Empty where procfs is unavailable: the callers treat descendant reaping as
    a backstop, so degrading to "found nothing" is preferable to raising out of
    a teardown path.
    """
    children: Dict[int, List[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return []
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        parent = _parent_of(pid)
        if parent is not None:
            children.setdefault(parent, []).append(pid)

    ordered: List[int] = []
    frontier = list(children.get(root_pid, ()))
    while frontier:
        pid = frontier.pop(0)
        ordered.append(pid)
        frontier.extend(children.get(pid, ()))
    ordered.reverse()
    return ordered


def terminate_descendants(root_pid: int, *, name_prefix: str, grace: float = 15.0) -> int:
    """Reap surviving descendants whose name starts with ``name_prefix``.

    Returns how many had to be signalled. The prefix keeps this narrow — a
    process tree shutdown that also swept up, say, dataloader workers would be
    a much bigger hammer than the problem needs.
    """
    stragglers = [pid for pid in descendants_of(root_pid) if _describe(pid).startswith(name_prefix)]
    if not stragglers:
        return 0

    for pid in stragglers:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass

    deadline = time.monotonic() + grace
    remaining = list(stragglers)
    while remaining and time.monotonic() < deadline:
        time.sleep(0.5)
        remaining = [pid for pid in remaining if os.path.exists(f"/proc/{pid}")]

    for pid in remaining:
        logger.warning("Descendant %d (%s) ignored SIGTERM; killing", pid, _describe(pid)[:80])
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(stragglers)


def run_with_timeout(fn: Callable[[], None], *, timeout: float, what: str) -> bool:
    """Run a teardown step, giving up on it after ``timeout`` seconds.

    Returns whether it finished. A teardown step that blocks forever is worse
    than one that fails: the steps after it — killing the Ray actors, dropping
    the placement groups — are the ones that actually free the GPUs. The call
    is not cancelled, just abandoned on a daemon thread, so the caller can move
    on to those.
    """
    done = threading.Event()

    def runner() -> None:
        try:
            fn()
        except BaseException:  # noqa: BLE001 - reported, never propagated into teardown
            logger.exception("%s failed", what)
        finally:
            done.set()

    worker = threading.Thread(target=runner, name=f"teardown-{what}", daemon=True)
    worker.start()
    if done.wait(timeout):
        return True
    logger.warning("%s did not finish within %.0fs; continuing without it", what, timeout)
    return False


class ShutdownRequested(BaseException):
    """Raised in the main thread when a termination signal arrives.

    Derives from ``BaseException`` rather than ``Exception`` on purpose: the
    training loop is full of ``except Exception`` guards that would otherwise
    swallow the request and carry on stepping.
    """


class GracefulShutdown:
    """Context manager that runs ``teardown`` on every exit path.

    Usage::

        with GracefulShutdown(trainer.shutdown, name="ar-train"):
            trainer.train(...)
    """

    def __init__(
        self,
        teardown: Callable[[], None],
        *,
        grace_seconds: float = _HARD_EXIT_GRACE_SECONDS,
        name: str = "run",
    ) -> None:
        self._teardown = teardown
        self._grace = grace_seconds
        self._name = name
        self._lock = threading.Lock()
        self._done = False
        self._signal_count = 0
        self._exit_code: Optional[int] = None
        self._previous: Dict[int, object] = {}
        self._timer: Optional[threading.Timer] = None

    def __enter__(self) -> "GracefulShutdown":
        self.claim_signals()
        self._reclaim_after_ray_init()
        atexit.register(self.run_teardown)
        return self

    def claim_signals(self) -> None:
        """Take ownership of the termination signals. Safe to call repeatedly.

        Repeatable because we are not the only ones installing handlers:
        ``ray.init()`` unconditionally overwrites SIGTERM with its own
        ``sys.exit`` handler and then lets the core worker install absl's
        failure-signal handler on top, which prints a stack trace and re-raises
        with the default disposition. Whoever registers last wins, and the
        default disposition kills the driver outright — no ``finally``, no
        engine shutdown, GPUs still held.
        """
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in _SIGNALS:
            try:
                previous = signal.getsignal(sig)
                if previous == self._on_signal:
                    continue
                if previous is signal.SIG_IGN:
                    continue
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                continue
            self._previous.setdefault(sig, previous)
            logger.info("Claimed %s for graceful shutdown (was %r)", signal.Signals(sig).name, previous)

    def _reclaim_after_ray_init(self) -> None:
        """Re-take the signals as soon as Ray finishes initialising.

        Ray is auto-initialised lazily on first use, which here happens deep
        inside device-pool construction — long after ``__enter__`` and well
        before the training loop. Hooking ``ray.init`` is what makes the
        handover ordering deterministic instead of a race we lose silently.
        """
        try:
            import ray
        except ImportError:
            return
        if getattr(ray.init, "_unirl_reclaims_signals", False):
            return

        original = ray.init

        @functools.wraps(original)
        def init_and_reclaim(*args, **kwargs):
            result = original(*args, **kwargs)
            self.claim_signals()
            return result

        init_and_reclaim._unirl_reclaims_signals = True  # type: ignore[attr-defined]
        ray.init = init_and_reclaim  # type: ignore[assignment]

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.run_teardown()
        for sig, handler in self._previous.items():
            try:
                signal.signal(sig, handler)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass
        if self._exit_code is not None:
            os._exit(self._exit_code)
        return False

    def _on_signal(self, signum: int, frame) -> None:
        self._signal_count += 1
        name = signal.Signals(signum).name
        if self._signal_count > 1:
            os._exit(128 + signum)
        self._exit_code = 128 + signum
        logger.warning(
            "%s received during %s — shutting down (send %s again to exit immediately)",
            name,
            self._name,
            name,
        )
        self._arm_hard_exit(128 + signum)
        raise ShutdownRequested(name)

    def _arm_hard_exit(self, exit_code: int) -> None:
        """Bound teardown in wall-clock time, outside the interpreter's control.

        A daemon timer thread is the only mechanism that still fires when the
        main thread is stuck in a C call that never yields the GIL back to
        Python — which is exactly what a hung collective or a blocking Ray RPC
        looks like.
        """
        if self._timer is not None:
            return
        self._timer = threading.Timer(self._grace, lambda: os._exit(exit_code))
        self._timer.daemon = True
        self._timer.start()

    def run_teardown(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
        self._arm_hard_exit(self._exit_code if self._exit_code is not None else 1)
        try:
            self._teardown()
        except BaseException:  # noqa: BLE001 - teardown failure must not mask the original error
            logger.exception("Teardown for %s failed", self._name)
        finally:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
