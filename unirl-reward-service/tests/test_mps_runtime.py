from __future__ import annotations

import os
from pathlib import Path

import pytest

from reward_service.config import MpsRuntimeCfg
from reward_service.mps import MpsRuntime


def _fake_control(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    calls = tmp_path / "calls"
    binary = tmp_path / "nvidia-cuda-mps-control"
    binary.write_text(
        """#!/bin/sh
if [ "${1:-}" = "-d" ]; then echo start >> "$CALLS"; exit 0; fi
while read -r command; do
  echo "$command" >> "$CALLS"
  [ "$command" = get_server_list ] && echo 4242
done
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    monkeypatch.setenv("CALLS", str(calls))
    return binary, calls


def test_managed_runtime_owns_daemon_and_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, calls = _fake_control(tmp_path, monkeypatch)
    pipe, log = tmp_path / "pipe", tmp_path / "log"
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/previous")
    runtime = MpsRuntime(
        MpsRuntimeCfg(
            mode="managed",
            pipe_directory=str(pipe),
            log_directory=str(log),
        ),
        binary=str(binary),
    )

    with runtime:
        assert "CUDA_MPS_PIPE_DIRECTORY" not in os.environ
        assert runtime.status()["server_pids"] == ["4242"]

    assert os.environ["CUDA_MPS_PIPE_DIRECTORY"] == "/previous"
    assert not pipe.exists() and not log.exists()
    assert calls.read_text().splitlines() == [
        "start",
        "get_server_list",
        "get_server_list",
        "quit",
    ]


def test_external_runtime_validates_but_does_not_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, calls = _fake_control(tmp_path, monkeypatch)
    pipe = tmp_path / "pipe"
    pipe.mkdir()
    with MpsRuntime(
        MpsRuntimeCfg(mode="external", pipe_directory=str(pipe)),
        binary=str(binary),
    ):
        pass
    assert pipe.exists()
    assert calls.read_text().splitlines() == ["get_server_list"]


def test_external_runtime_requires_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, _ = _fake_control(tmp_path, monkeypatch)
    runtime = MpsRuntime(
        MpsRuntimeCfg(mode="external", pipe_directory=str(tmp_path / "missing")),
        binary=str(binary),
    )
    with pytest.raises(RuntimeError, match="does not exist"):
        runtime.start()


def test_managed_runtime_refuses_existing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, _ = _fake_control(tmp_path, monkeypatch)
    pipe = tmp_path / "pipe"
    pipe.mkdir()
    runtime = MpsRuntime(
        MpsRuntimeCfg(
            mode="managed",
            pipe_directory=str(pipe),
            log_directory=str(tmp_path / "log"),
        ),
        binary=str(binary),
    )
    with pytest.raises(RuntimeError, match="existing MPS path"):
        runtime.start()
