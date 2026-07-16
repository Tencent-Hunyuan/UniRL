"""SandboxTool — a persistent Python-REPL subprocess tool (LIN-533).

The first out-of-process :class:`~unirl.rollout.loop.tools.tool.StatefulTool`: each session owns a
long-lived ``python`` subprocess whose global namespace **persists across turns**, so ``x = 40`` on
one turn and ``x + 2`` on the next returns ``42`` — the cross-turn capability a stateless
:class:`~unirl.rollout.loop.tools.tool.Tool` cannot express.

Lifecycle (all off the shared event loop — ``execute_session``/``session_end`` run in
``ToolEnvironment``'s executor):

- ``session_start`` records the session; it does **no** I/O (no subprocess spawned yet).
- ``execute_session`` opens the subprocess **lazily** on first use (so an unused session never
  spawns one), pipes the code as one JSON line, and reads one framed JSON response.
- ``session_end`` kills the subprocess. Idempotent, never raises.

The subprocess is killed (not pooled) on ``session_end``: a live interpreter cannot be safely reset
between trajectories, and spawning a fresh one is cheap (tens of ms, paid lazily in the executor).
Bad code, exceptions, and timeouts are returned as **text** for the model to recover from, never
raised.

Note: uses ``select`` on the child's stdout pipe for the per-call timeout — Unix (Linux/macOS),
which is where UniRL rollouts run.
"""

from __future__ import annotations

import json
import select
import subprocess
import sys
import threading
from typing import Any, Dict, Optional

from unirl.rollout.loop.tools.tool import StatefulTool

# Runs in the child process. Protocol: read one JSON line ``{"code": ...}`` from stdin; exec it in a
# persistent namespace with stdout+stderr captured; if the last top-level statement is an expression,
# echo its ``repr`` (REPL-style); write one JSON line ``{"out": ...}`` back and flush; loop until
# stdin closes. User output is captured into a StringIO (never the real pipe), so a chatty cell can't
# deadlock the pipe — only the single framed response reaches the parent.
_WORKER_SRC = r"""
import ast, io, json, sys, contextlib, traceback

_ns = {"__name__": "__sandbox__"}


def _run(code):
    out = io.StringIO()
    try:
        block = ast.parse(code, "<cell>", "exec")
    except SyntaxError:
        return traceback.format_exc()
    last = block.body.pop() if (block.body and isinstance(block.body[-1], ast.Expr)) else None
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            exec(compile(block, "<cell>", "exec"), _ns)
            if last is not None:
                value = eval(compile(ast.Expression(last.value), "<cell>", "eval"), _ns)
                if value is not None:
                    print(repr(value), file=out)
    except BaseException:
        traceback.print_exc(file=out)
    return out.getvalue()


for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _code = json.loads(_line).get("code", "")
    except Exception:
        _code = ""
    sys.stdout.write(json.dumps({"out": _run(_code)}) + "\n")
    sys.stdout.flush()
"""


class _Session:
    """Per-trajectory sandbox state: the (lazily opened) subprocess + a guard lock."""

    __slots__ = ("proc", "lock", "context")

    def __init__(self, context: Dict[str, Any]) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.context = context


class SandboxTool(StatefulTool):
    """Execute Python in a per-session persistent interpreter (state survives across turns)."""

    name = "python"

    def __init__(self, *, timeout_s: float = 10.0, max_output_chars: int = 4000) -> None:
        self._timeout_s = float(timeout_s)
        self._max_output_chars = int(max_output_chars)
        # Session store shared across concurrent trajectories on the worker → lock-guarded.
        self._sessions: Dict[str, _Session] = {}
        self._store_lock = threading.Lock()

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Execute Python code in a persistent, stateful interpreter. Variables, imports, "
                    "and definitions from earlier calls remain available in later calls. The value of "
                    "a trailing expression is echoed (like a REPL); use print(...) for other output."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python source to execute."}},
                    "required": ["code"],
                },
            },
        }

    # ------------------------------------------------------------------
    # StatefulTool lifecycle
    # ------------------------------------------------------------------
    def session_start(self, session_id: str, context: Dict[str, Any]) -> None:
        """Record the session — no subprocess yet (opened lazily in :meth:`execute_session`)."""
        with self._store_lock:
            self._sessions[session_id] = _Session(context or {})

    def execute_session(self, session_id: str, arguments: Dict[str, Any]) -> str:
        code = arguments.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("python tool requires a non-empty 'code' string argument")
        with self._store_lock:
            session = self._sessions.get(session_id)
            if session is None:  # session_start skipped (not via ToolEnvironment) — tolerate it
                session = self._sessions.setdefault(session_id, _Session({}))
        with session.lock:
            if self._ensure_proc(session) is None:
                return "Error: sandbox failed to start"
            return self._exchange(session, code)

    def session_end(self, session_id: str) -> None:
        """Kill the subprocess and drop the session. Idempotent; never raises."""
        with self._store_lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            with session.lock:
                self._kill(session)

    # ------------------------------------------------------------------
    # Subprocess plumbing
    # ------------------------------------------------------------------
    def _ensure_proc(self, session: _Session) -> Optional[subprocess.Popen]:
        """Lazily spawn (or respawn a dead) worker subprocess. Caller holds ``session.lock``."""
        if session.proc is not None and session.proc.poll() is None:
            return session.proc
        try:
            session.proc = subprocess.Popen(
                [sys.executable, "-u", "-c", _WORKER_SRC],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception:  # noqa: BLE001 — surfaced to the model as an error string
            session.proc = None
        return session.proc

    def _exchange(self, session: _Session, code: str) -> str:
        """Send one code cell, read one framed response; enforce the timeout. Caller holds the lock."""
        proc = session.proc
        assert proc is not None and proc.stdin is not None and proc.stdout is not None
        try:
            proc.stdin.write(json.dumps({"code": code}) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._kill(session)
            return "Error: sandbox process is not writable (it may have crashed)"

        ready, _, _ = select.select([proc.stdout], [], [], self._timeout_s)
        if not ready:
            self._kill(session)
            return f"Error: sandbox timed out after {self._timeout_s:g}s"
        line = proc.stdout.readline()
        if not line:
            self._kill(session)
            return "Error: sandbox process exited unexpectedly"

        try:
            out = json.loads(line).get("out", "")
        except (json.JSONDecodeError, ValueError):
            out = line
        if len(out) > self._max_output_chars:
            out = out[: self._max_output_chars] + "\n...(truncated)"
        return out if out else "(no output)"

    @staticmethod
    def _kill(session: _Session) -> None:
        """Terminate the subprocess and close its pipes. Idempotent; swallows all errors."""
        proc = session.proc
        session.proc = None
        if proc is None:
            return
        for closer in (lambda: proc.stdin.close(),):  # stop writes first
            try:
                closer()
            except Exception:  # noqa: BLE001
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                proc.kill()
                try:
                    proc.wait(timeout=2)
                except Exception:  # noqa: BLE001
                    pass
        for stream in (proc.stdout, proc.stdin):
            try:
                stream.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["SandboxTool"]
