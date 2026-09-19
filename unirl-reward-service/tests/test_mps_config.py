from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reward_service.config import load_config


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
