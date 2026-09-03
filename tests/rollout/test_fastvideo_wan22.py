from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).parents[2] / "unirl" / "rollout" / "engine"
for _name, _path in (
    ("unirl.config", Path(__file__).parents[2] / "unirl" / "config"),
    ("unirl.distributed.tensor", Path(__file__).parents[2] / "unirl" / "distributed" / "tensor"),
    ("unirl.rollout.engine", _ROOT),
    ("unirl.rollout.engine.fastvideo", _ROOT / "fastvideo"),
    ("unirl.rollout.engine.fastvideo.adapters", _ROOT / "fastvideo" / "adapters"),
    ("unirl.rollout.engine.fastvideo._patches", _ROOT / "fastvideo" / "_patches"),
):
    if _name not in sys.modules:
        _package = types.ModuleType(_name)
        _package.__path__ = [str(_path)]
        sys.modules[_name] = _package

from unirl.rollout.engine.fastvideo.adapters.wan22 import Wan22FastVideoAdapter  # noqa: E402


def _adapter(tmp_path, *, model_index=None) -> Wan22FastVideoAdapter:
    if model_index is not None:
        (tmp_path / "model_index.json").write_text(json.dumps(model_index))
    config = SimpleNamespace(native_logprob=True)
    model_config = SimpleNamespace(
        pretrained_model_ckpt_path=str(tmp_path),
        shift=5.0,
        boundary_ratio=0.875,
        guidance_scale_2=3.0,
        num_train_timesteps=1000,
    )
    strategy = SimpleNamespace(canonical_name="dance")
    return Wan22FastVideoAdapter(config, model_config, strategy=strategy)


def test_wan22_adapter_requires_both_experts(tmp_path) -> None:
    with pytest.raises(ValueError, match="transformer_2"):
        _adapter(tmp_path, model_index={"transformer": ["diffusers", "WanTransformer3DModel"]})


def test_wan22_adapter_aligns_dual_expert_boundary(tmp_path) -> None:
    adapter = _adapter(
        tmp_path,
        model_index={
            "transformer": ["diffusers", "WanTransformer3DModel"],
            "transformer_2": ["diffusers", "WanTransformer3DModel"],
        },
    )
    pipeline_config = SimpleNamespace(
        flow_shift=12.0,
        boundary_ratio=None,
        dit_config=SimpleNamespace(boundary_ratio=None),
    )
    fastvideo_args = SimpleNamespace(pipeline_config=pipeline_config)

    adapter.align_runtime_args(fastvideo_args)

    assert pipeline_config.flow_shift == 5.0
    assert pipeline_config.boundary_ratio == 0.875
    assert pipeline_config.dit_config.boundary_ratio == 0.875
    assert fastvideo_args._unirl_custom_sigmas_dtype == "float32"
