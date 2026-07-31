from pathlib import Path

import pytest
import yaml

from unirl.rollout.engine.sglang.backends.base import (
    _REQUIRED_SERVER_ARGS_METADATA_KEY,
    _filter_server_args_or_raise,
    _normalize_cuda_visible_devices,
)


def test_filter_server_args_drops_escape_hatches_but_keeps_required_fields() -> None:
    intent = {
        "model_path": "model",
        "enable_weights_cpu_backup": True,
        "non_server_escape_hatch": "ignored",
        _REQUIRED_SERVER_ARGS_METADATA_KEY: ["enable_weights_cpu_backup"],
    }

    assert _filter_server_args_or_raise(
        intent,
        allowed={"model_path", "enable_weights_cpu_backup"},
        backend_name="HTTP",
    ) == {
        "model_path": "model",
        "enable_weights_cpu_backup": True,
    }


def test_filter_server_args_fails_closed_when_required_field_is_missing() -> None:
    intent = {
        "enable_weights_cpu_backup": True,
        _REQUIRED_SERVER_ARGS_METADATA_KEY: ["enable_weights_cpu_backup"],
    }

    with pytest.raises(RuntimeError, match="enable_weights_cpu_backup"):
        _filter_server_args_or_raise(
            intent,
            allowed=set(),
            backend_name="HTTP",
        )


def test_normalize_cuda_visible_devices_preserves_opaque_tokens() -> None:
    assert _normalize_cuda_visible_devices(["0", "GPU-abc"], tp_size=2) == ["0", "GPU-abc"]


@pytest.mark.parametrize("tokens", [["0"], ["0", ""], ["0,1", "2"]])
def test_normalize_cuda_visible_devices_rejects_ambiguous_layout(tokens: list[str]) -> None:
    with pytest.raises(ValueError):
        _normalize_cuda_visible_devices(tokens, tp_size=2)


def test_full_weight_sync_recipe_enables_expert_parallelism_with_ep_size() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    recipe = yaml.safe_load(
        (repo_root / "examples/ar/qwen3_cppo_30b_a3b_base_dapo_sglang_full_weight_sync.yaml").read_text()
    )
    rollout_config = recipe["rollout"]["config"]

    assert rollout_config["ep_size"] == rollout_config["tp_size"] == 8
    assert "enable_expert_parallel" not in rollout_config
