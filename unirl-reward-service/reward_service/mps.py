"""Opt-in NVIDIA MPS daemon lifecycle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from reward_service.config import MpsRuntimeCfg

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
subprocess.run([binary], input="quit\n", text=True, env=env,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
shutil.rmtree(pipe, ignore_errors=True)
shutil.rmtree(log, ignore_errors=True)
"""


class MpsRuntime:
    def __init__(self, cfg: MpsRuntimeCfg, *, binary: str | None = None) -> None:
        self.cfg = cfg
        self.binary = binary or shutil.which("nvidia-cuda-mps-control")
        suffix = f"{os.getuid()}-{os.getpid()}"
        self.pipe_directory = Path(
            cfg.pipe_directory or f"/tmp/unirl-mps-{suffix}"
        )
        self.log_directory = Path(
            cfg.log_directory or f"/tmp/unirl-mps-log-{suffix}"
        )
        self._started = False
        self._owned_directories: list[Path] = []
        self._saved_environment: dict[str, str] = {}
        self._watchdog: subprocess.Popen | None = None

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "CUDA_MPS_PIPE_DIRECTORY": str(self.pipe_directory),
            "CUDA_MPS_LOG_DIRECTORY": str(self.log_directory),
        }

    def _control(self, command: str, *, check: bool = True) -> str:
        if self.binary is None:
            raise RuntimeError("nvidia-cuda-mps-control is not available")
        result = subprocess.run(
            [self.binary],
            input=f"{command}\n",
            text=True,
            capture_output=True,
            env=self._environment(),
            timeout=10,
        )
        if check and result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"MPS command {command!r} failed: {detail}")
        return result.stdout.strip()

    def _make_directory(self, path: Path) -> None:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"refusing to replace existing MPS path: {path}")
        path.mkdir(parents=True, mode=0o700)
        self._owned_directories.append(path)

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        if self.binary is None:
            raise RuntimeError("MPS requested but nvidia-cuda-mps-control is unavailable")
        daemon_started = False
        try:
            if self.cfg.mode == "managed":
                self._make_directory(self.pipe_directory)
                self._make_directory(self.log_directory)
                result = subprocess.run(
                    [self.binary, "-d"],
                    text=True,
                    capture_output=True,
                    env=self._environment(),
                    timeout=10,
                )
                if result.returncode:
                    raise RuntimeError(
                        f"failed to start MPS daemon: {result.stderr.strip()}"
                    )
                daemon_started = True
            elif not self.pipe_directory.is_dir():
                raise RuntimeError(
                    f"external MPS pipe does not exist: {self.pipe_directory}"
                )
            self._control("get_server_list")
        except Exception:
            if daemon_started:
                self._control("quit", check=False)
            self._cleanup()
            raise
        if daemon_started:
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
            "mode": self.cfg.mode,
            "pipe_directory": str(self.pipe_directory),
            "shared_fault_domain": True,
        }
        if not self._started:
            return payload
        try:
            payload["server_pids"] = self._control("get_server_list").splitlines()
            payload["enabled"] = True
        except Exception as exc:
            payload["error"] = str(exc)
        return payload

    def _cleanup(self) -> None:
        for path in reversed(self._owned_directories):
            shutil.rmtree(path, ignore_errors=True)
        self._owned_directories.clear()

    def stop(self) -> None:
        if not self._started:
            return
        try:
            if self._watchdog is not None:
                self._watchdog.terminate()
                self._watchdog.wait(timeout=5)
                self._watchdog = None
            if self.cfg.mode == "managed":
                self._control("quit", check=False)
        finally:
            os.environ.update(self._saved_environment)
            self._saved_environment.clear()
            self._cleanup()
            self._started = False

    def __enter__(self) -> MpsRuntime:
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.stop()
