from __future__ import annotations

import inspect
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class, instantiate

from unirl.rollout.engine.native_sd3.config import NativeSD3EngineConfig
from unirl.trainer.diffusion import build_eval_sampling
from unirl.types.sampling import DiffusionSamplingParams


def test_solrl_h20_recipe_composes_with_expected_geometry() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(config_name="diffusion/sd3/sd3_solrl_fp8_h20")

    assert cfg.num_devices == 32
    assert cfg.batch_size == 48
    assert cfg.scout_sampling.samples_per_prompt == 128
    assert cfg.sampling.samples_per_prompt == 16
    assert cfg.bundle.config.trajectory_precision == "bf16"
    assert cfg.contrastive_rollout.top_k == 8
    assert cfg.contrastive_rollout.bottom_k == 8
    assert cfg.eval_sampling.num_inference_steps == 40
    assert cfg.algorithm.ref_deviation_coef == 1.0e-4
    assert "fp8_dim_multiple" not in cfg.rollout.config
    assert "ema_decay" not in cfg.backend.ema_lora_cfg
    assert isinstance(instantiate(cfg.rollout.config), NativeSD3EngineConfig)
    assert isinstance(instantiate(cfg.sampling), DiffusionSamplingParams)
    assert isinstance(instantiate(cfg.scout_sampling), DiffusionSamplingParams)
    algorithm_cls = get_class(cfg.algorithm._target_)
    assert set(cfg.algorithm) - {"_target_"} <= set(inspect.signature(algorithm_cls).parameters)


def test_paper_naive_override_arm_composes() -> None:
    examples = Path(__file__).resolve().parents[2] / "examples"
    overrides = [
        "contrastive_rollout.mode=naive",
        "contrastive_rollout.top_k=12",
        "contrastive_rollout.bottom_k=12",
        "sampling.samples_per_prompt=24",
        "scout_sampling.samples_per_prompt=96",
        "rollout.config.fp8_enabled=false",
    ]
    with initialize_config_dir(config_dir=str(examples), version_base=None):
        cfg = compose(
            config_name="diffusion/sd3/sd3_solrl_fp8_h20",
            overrides=overrides,
        )
    assert cfg.contrastive_rollout.mode == "naive"
    assert cfg.sampling.samples_per_prompt == 24
    assert cfg.scout_sampling.samples_per_prompt == 96


def test_eval_fanout_override_keeps_deprecated_alias_in_sync() -> None:
    sampling = {
        "diffusion": DiffusionSamplingParams(
            samples_per_prompt=16,
            num_samples_per_prompt=16,
        )
    }
    evaluated = build_eval_sampling(sampling, samples_per_prompt=1)
    assert evaluated["diffusion"].samples_per_prompt == 1
    assert evaluated["diffusion"].num_samples_per_prompt == 1

    alias_override = build_eval_sampling(
        sampling,
        samples_per_prompt=1,
        overrides={"num_samples_per_prompt": 2},
    )
    assert alias_override["diffusion"].samples_per_prompt == 2
    assert alias_override["diffusion"].num_samples_per_prompt == 2

    try:
        build_eval_sampling(
            sampling,
            overrides={"samples_per_prompt": 2, "num_samples_per_prompt": 3},
        )
    except ValueError as exc:
        assert "conflicting samples_per_prompt" in str(exc)
    else:
        raise AssertionError("expected conflicting fanout aliases to fail")

    with pytest.raises(ValueError, match="conflicting samples_per_prompt"):
        DiffusionSamplingParams(samples_per_prompt=2, num_samples_per_prompt=3)
