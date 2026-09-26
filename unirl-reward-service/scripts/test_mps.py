from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

try:
    import ray  # noqa: F401
except ImportError:
    sys.modules["ray"] = SimpleNamespace(
        remote=lambda cls: cls,
        get_gpu_ids=lambda: [],
        get=Mock(),
        kill=Mock(),
    )

from reward_service.config import MpsClientCfg, MpsRuntimeCfg, RewardModelCfg, load_config
from reward_service.mps import MpsRuntime
from reward_service.workers.group import WorkerGroup, _actor_options


def _config(tmp_path: Path, **reward_overrides) -> Path:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    reward = {
        "name": "reward",
        "scorer": "pickscore",
        "runtime_env": str(requirements),
        "num_gpus": 0.5,
        "mps": {},
        "params": {},
        **reward_overrides,
    }
    path = tmp_path / "service.yaml"
    path.write_text(
        yaml.safe_dump({"mps": {"mode": "managed"}, "rewards": [reward]}),
        encoding="utf-8",
    )
    return path


def test_loads_typed_mps_limits(tmp_path: Path) -> None:
    cfg = load_config(
        _config(
            tmp_path,
            mps={
                "active_thread_percentage": 75,
                "device_memory_limit": "12GiB",
            },
        )
    )
    assert cfg.rewards[0].mps.active_thread_percentage == 75
    assert cfg.rewards[0].mps.device_memory_limit_env == "0=12288M"


@pytest.mark.parametrize("active", [0, 101, 50.0, True])
def test_rejects_invalid_active_thread_percentage(
    tmp_path: Path, active: object
) -> None:
    with pytest.raises(ValueError, match="integer in"):
        load_config(_config(tmp_path, mps={"active_thread_percentage": active}))


@pytest.mark.parametrize("num_gpus", [0, 1, float("nan"), float("inf"), True])
def test_requires_fractional_gpu(tmp_path: Path, num_gpus: object) -> None:
    with pytest.raises(ValueError, match="num_gpus"):
        load_config(_config(tmp_path, num_gpus=num_gpus))


def test_rejects_tensor_parallel_mps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="tensor_parallel_size>1"):
        load_config(_config(tmp_path, params={"tensor_parallel_size": 2}))


def test_rejects_unqualified_backend(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not MPS-qualified"):
        load_config(_config(tmp_path, scorer="unified_reward"))


def test_rejects_unsafe_float16_clip_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only qualified"):
        load_config(
            _config(
                tmp_path,
                scorer="clip",
                params={"dtype": "float16"},
                mps={"active_thread_percentage": 50},
            )
        )


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


def _cfg(tmp_path: Path, mps: MpsClientCfg | None) -> RewardModelCfg:
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    return RewardModelCfg(
        name="pickscore",
        scorer="pickscore",
        runtime_env=str(requirements),
        num_gpus=0.5,
        mps=mps,
    )


def test_actor_options_inject_mps_environment(tmp_path: Path) -> None:
    options = _actor_options(
        _cfg(tmp_path, MpsClientCfg(75, 12288)),
        mps_pipe_directory="/tmp/unirl-mps-test",
    )
    assert options["runtime_env"]["env_vars"] == {
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/unirl-mps-test",
        "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE": "75",
        "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": "0=12288M",
    }


def test_non_mps_actor_options_are_unchanged(tmp_path: Path) -> None:
    assert _actor_options(_cfg(tmp_path, None))["runtime_env"] == {
        "pip": {"packages": [], "pip_check": False}
    }


def test_group_drains_before_kill(monkeypatch) -> None:
    events = []
    actor = SimpleNamespace(
        shutdown=SimpleNamespace(remote=lambda: events.append("drain") or object()),
    )
    monkeypatch.setattr(
        "reward_service.workers.group.ray.get",
        lambda refs, timeout: events.append("wait"),
    )
    monkeypatch.setattr(
        "reward_service.workers.group.ray.kill",
        lambda actor, no_restart: events.append("kill"),
    )
    group = WorkerGroup.__new__(WorkerGroup)
    group.cfg = SimpleNamespace(name="test")
    group.actors = [actor]

    group.shutdown()

    assert events == ["drain", "wait", "kill"]
