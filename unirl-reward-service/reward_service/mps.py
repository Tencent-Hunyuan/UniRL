"""Opt-in NVIDIA MPS daemon lifecycle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

_SUFFIX = f"{os.getuid()}-{os.getpid()}"
MPS_PIPE_DIRECTORY = Path(f"/tmp/unirl-mps-{_SUFFIX}")
_MPS_LOG_DIRECTORY = Path(f"/tmp/unirl-mps-log-{_SUFFIX}")

_CLIENT_ENV = (
    "CUDA_MPS_PIPE_DIRECTORY",
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


class MpsRuntime:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.binary = shutil.which("nvidia-cuda-mps-control")
        self.pipe_directory = MPS_PIPE_DIRECTORY
        self.log_directory = _MPS_LOG_DIRECTORY
        self._started = False
        self._saved_environment: dict[str, str] = {}
        self._watchdog: subprocess.Popen | None = None

    def _run_control(
        self,
        *args: str,
        input_cmd: str | None = None,
        check: bool = True,
    ) -> str:
        env = {k: v for k, v in os.environ.items() if k not in _CLIENT_ENV}
        env["CUDA_MPS_PIPE_DIRECTORY"] = str(self.pipe_directory)
        env["CUDA_MPS_LOG_DIRECTORY"] = str(self.log_directory)
        result = subprocess.run(
            [self.binary, *args],
            input=input_cmd,
            text=True,
            capture_output=True,
            env=env,
            timeout=10,
        )
        if check and result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            label = input_cmd.strip() if input_cmd else " ".join(args)
            raise RuntimeError(f"MPS command {label!r} failed: {detail}")
        return result.stdout.strip()

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        if self.binary is None:
            raise RuntimeError("MPS requested but nvidia-cuda-mps-control is unavailable")
        for path in (self.pipe_directory, self.log_directory):
            if path.exists() or path.is_symlink():
                raise RuntimeError(f"refusing to replace existing MPS path: {path}")

        daemon_started = False
        try:
            self.pipe_directory.mkdir(parents=True, mode=0o700)
            self.log_directory.mkdir(parents=True, mode=0o700)
            self._run_control("-d")
            daemon_started = True
            self._run_control(input_cmd="get_server_list\n")
        except Exception:
            if daemon_started:
                self._run_control(input_cmd="quit\n", check=False)
            shutil.rmtree(self.pipe_directory, ignore_errors=True)
            shutil.rmtree(self.log_directory, ignore_errors=True)
            raise

        self._watchdog = subprocess.Popen(
            [
                sys.executable,
                "-c",
                _WATCHDOG,
                str(os.getpid()),
                self.binary,
                str(self.pipe_directory),
                str(self.log_directory),
            ],
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        for name in _CLIENT_ENV:
            value = os.environ.pop(name, None)
            if value is not None:
                self._saved_environment[name] = value
        self._started = True

    def status(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "enabled": False,
            "pipe_directory": str(self.pipe_directory),
            "shared_fault_domain": True,
        }
        if not self._started:
            return payload
        try:
            payload["server_pids"] = self._run_control(input_cmd="get_server_list\n").splitlines()
            payload["enabled"] = True
        except Exception as exc:
            payload["error"] = str(exc)
        return payload

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._watchdog is not None:
                self._watchdog.terminate()
                try:
                    self._watchdog.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._watchdog.kill()
                    self._watchdog.wait(timeout=5)
                self._watchdog = None
            self._run_control(input_cmd="quit\n", check=False)
        finally:
            os.environ.update(self._saved_environment)
            self._saved_environment.clear()
            shutil.rmtree(self.pipe_directory, ignore_errors=True)
            shutil.rmtree(self.log_directory, ignore_errors=True)
            self._started = False
