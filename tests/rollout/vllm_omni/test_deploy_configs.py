from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
DEPLOY_DIR = ROOT / "unirl/rollout/engine/vllm_omni/deploy_configs"


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def test_all_deploy_configs_use_stable_schema() -> None:
    paths = sorted(DEPLOY_DIR.glob("*.yaml"))
    assert len(paths) == 12

    for path in paths:
        config = yaml.safe_load(path.read_text())
        assert "stage_args" not in config, path
        assert isinstance(config.get("pipeline"), str), path
        assert isinstance(config.get("stages"), list), path
        assert config["stages"], path


def test_all_vllm_omni_recipes_use_deploy_config_vocabulary() -> None:
    examples = ROOT / "examples"
    paths = {
        *examples.glob("**/*vllmomni*.yaml"),
        *examples.glob("**/*vllm_omni*.yaml"),
        examples / "diffusion/bagel/bagel_it2i_managed_editscore.yaml",
    }
    assert paths

    for path in sorted(paths):
        config = yaml.safe_load(path.read_text())
        for mapping in _walk(config):
            assert "stage_yaml_override" not in mapping, path
            assert "stage_yaml" not in mapping, path
            override = mapping.get("deploy_config_override")
            if override is not None:
                assert (DEPLOY_DIR / override).is_file(), (path, override)


def test_every_adapter_deploy_config_exists() -> None:
    from unirl.rollout.engine.vllm_omni.adapters import registered_adapters
    from unirl.rollout.engine.vllm_omni.adapters.base import get_adapter

    for name in registered_adapters():
        filename = get_adapter(name).deploy_config
        assert filename
        assert (DEPLOY_DIR / filename).is_file(), (name, filename)


def test_stable_factory_resolves_every_deploy_config() -> None:
    pytest.importorskip("vllm_omni")
    from vllm_omni.config.config_factory import StageConfigFactory
    from vllm_omni.config.pipeline_registry import resolve_pipeline_config
    from vllm_omni.config.stage_config import load_deploy_config

    from unirl.rollout.engine.vllm_omni.pipeline_configs import register_unirl_pipeline_configs

    register_unirl_pipeline_configs()
    for path in sorted(DEPLOY_DIR.glob("*.yaml")):
        deploy = load_deploy_config(path)
        pipeline = resolve_pipeline_config(deploy.pipeline)
        assert pipeline is not None, path
        stages, _ = StageConfigFactory._create_legacy_from_registry(
            pipeline,
            {},
            str(path),
            deploy,
        )
        assert stages, path
        assert [stage.stage_id for stage in stages] == list(range(len(stages))), path


def test_hi3_text_topologies_preserve_legacy_output_contract() -> None:
    pytest.importorskip("vllm_omni")
    from unirl.rollout.engine.vllm_omni.pipeline_configs import (
        UNIRL_HI3_AR_MULTIMODAL_TEXT,
        UNIRL_HI3_AR_RECAPTION,
        UNIRL_HI3_AR_TEXT,
        UNIRL_QWEN3_OMNI_THINKER,
    )

    text_stage = UNIRL_HI3_AR_TEXT.stages[0]
    multimodal_stage = UNIRL_HI3_AR_MULTIMODAL_TEXT.stages[0]
    for stage in (text_stage, multimodal_stage):
        assert stage.final_output_type == "text"
        assert stage.engine_output_type == "text"
        assert stage.owns_tokenizer is True
    assert text_stage.requires_multimodal_data is False
    assert multimodal_stage.requires_multimodal_data is True

    recaption_stage = UNIRL_HI3_AR_RECAPTION.stages[0]
    assert recaption_stage.final_output_type == "text"
    assert recaption_stage.engine_output_type == "latent"
    assert recaption_stage.owns_tokenizer is True
    assert recaption_stage.requires_multimodal_data is False

    qwen_stage = UNIRL_QWEN3_OMNI_THINKER.stages[0]
    assert qwen_stage.model_arch == "Qwen3OmniMoeThinkerForConditionalGeneration"
    assert qwen_stage.engine_output_type == "text"


def test_hi3_t2i_does_not_forward_at_raw_prefill_output() -> None:
    config = yaml.safe_load((DEPLOY_DIR / "hunyuan_image3_t2i_rl.yaml").read_text())
    stage_zero = config["stages"][0]
    assert stage_zero["omni_kv_config"] == {"need_send_cache": True}


@pytest.mark.parametrize("name", ["hunyuan_image3_ar_rl.yaml", "hunyuan_image3_i2t_rl.yaml"])
def test_hi3_text_lora_adapters_install_weight_sync_extension(name: str) -> None:
    config = yaml.safe_load((DEPLOY_DIR / name).read_text())
    stage_zero = config["stages"][0]
    assert stage_zero["enable_lora"] is True
    assert stage_zero["worker_extension_cls"].endswith(".HI3ARWeightSyncExtension")
