"""Opt-in NVIDIA MPS daemon lifecycle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_SUFFIX = f"{os.getuid()}-{os.getpid()}"
MPS_PIPE_DIRECTORY = Path(f"/tmp/unirl-mps-{_SUFFIX}")
_MPS_LOG_DIRECTORY = Path(f"/tmp/unirl-mps-log-{_SUFFIX}")
_CLIENT_LIMIT_ENV = (
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
    "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT",
)

_WATCHDOG = r"""
import os, shutil, subprocess, sys, time
owner, binary, pipe, log = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
while True:
    try:
        os.kill(owner, 0)
    except ProcessLookupError:
        break
    time.sleep(1)
env = {**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe, "CUDA_MPS_LOG_DIRECTORY": log}
try:
    subprocess.run([binary], input="quit\n", text=True, env=env, timeout=10,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
except subprocess.TimeoutExpired:
    pass
shutil.rmtree(pipe, ignore_errors=True)
shutil.rmtree(log, ignore_errors=True)
"""


@contextmanager
def mps_daemon(*, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return
    binary = shutil.which("nvidia-cuda-mps-control")
    if binary is None:
        raise RuntimeError("MPS requested but nvidia-cuda-mps-control is unavailable")
    for path in (MPS_PIPE_DIRECTORY, _MPS_LOG_DIRECTORY):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace existing MPS path: {path}")

    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _CLIENT_LIMIT_ENV
    }
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(MPS_PIPE_DIRECTORY)
    env["CUDA_MPS_LOG_DIRECTORY"] = str(_MPS_LOG_DIRECTORY)
    watchdog: subprocess.Popen | None = None
    daemon_started = False
    try:
        MPS_PIPE_DIRECTORY.mkdir(parents=True, mode=0o700)
        _MPS_LOG_DIRECTORY.mkdir(parents=True, mode=0o700)
        result = subprocess.run(
            [binary, "-d"],
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"failed to start MPS daemon: {detail}")
        daemon_started = True
        watchdog = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WATCHDOG,
                str(os.getpid()),
                binary,
                str(MPS_PIPE_DIRECTORY),
                str(_MPS_LOG_DIRECTORY),
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        yield
    finally:
        if watchdog is not None:
            watchdog.terminate()
            try:
                watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                watchdog.kill()
                watchdog.wait()
        if daemon_started:
            try:
                subprocess.run(
                    [binary],
                    input="quit\n",
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=10,
                )
            except subprocess.TimeoutExpired:
                pass
        shutil.rmtree(MPS_PIPE_DIRECTORY, ignore_errors=True)
        shutil.rmtree(_MPS_LOG_DIRECTORY, ignore_errors=True)
