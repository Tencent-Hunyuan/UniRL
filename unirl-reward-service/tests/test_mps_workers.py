from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import ray  # noqa: F401
except ImportError:
    sys.modules["ray"] = SimpleNamespace(
        remote=lambda cls: cls,
        get_gpu_ids=lambda: [],
        get=Mock(),
        kill=Mock(),
    )

from reward_service.config import MpsClientCfg, RewardModelCfg
from reward_service.workers.group import WorkerGroup, _actor_options


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
