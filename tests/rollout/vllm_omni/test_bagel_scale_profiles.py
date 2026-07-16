from __future__ import annotations

import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

from unirl.trainer.base import build_sampling_dict
from unirl.trainer.unified_model import UnifiedModelTrainer

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_DIR = REPO_ROOT / "examples"
LAUNCHER = REPO_ROOT / "scripts" / "launch_bagel_vllmomni_t2ti.sh"


def _compose(config_name: str):
    with initialize_config_dir(config_dir=str(EXAMPLES_DIR), version_base=None):
        return compose(config_name=f"unified_model/{config_name}")


def _resolve_launcher_profile(profile: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; resolve_bagel_profile_config "$2"',
            "profile-test",
            str(LAUNCHER),
            profile,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_strict_t2ti_contract(cfg, *, expected_pairs: int) -> None:
    assert cfg.weight_sync_interval == 1
    assert cfg.rollout.config.modality == "bagel_t2ti"
    assert cfg.pipeline.t2ti_replay_chunk_mode == "exact"
    assert cfg.pipeline.t2ti_replay_execution_order == "layer_major"
    assert cfg.algorithm.image.context_gradient_mode == "stage_boundary"
    assert cfg.algorithm.image.lazy_first_update_anchor is True
    assert cfg.algorithm.image.reuse_ratio_context_for_mse is False
    assert list(cfg.sync.stage_ids) == [0, 1]
    assert cfg.sampling.diffusion.samples_per_prompt == 1
    assert cfg.batch_size * cfg.sampling.ar.samples_per_prompt == expected_pairs
    UnifiedModelTrainer._validate_bagel_t2ti_contract(build_sampling_dict(cfg.sampling), cfg.sync)


def test_production_profile_matches_unigrpo_scale() -> None:
    cfg = _compose("bagel_vllmomni_t2ti")

    _assert_strict_t2ti_contract(cfg, expected_pairs=32 * 24)
    assert (cfg.num_devices, cfg.devices_per_node, cfg.batch_size) == (32, 8, 32)
    assert cfg.enable_fsdp_offload is False
    assert cfg.backend.fsdp_cfg.cpu_offload is False
    assert cfg.sampling.ar.samples_per_prompt == 24
    assert cfg.sampling.diffusion.samples_per_prompt == 1
    assert cfg.sampling.ar.max_new_tokens == 1024
    assert cfg.sampling.diffusion.num_inference_steps == 25
    assert cfg.sampling.diffusion.scheduler.num_timesteps == 25
    assert cfg.sampling.diffusion.scheduler.num_sde_steps == 3
    assert cfg.reward.backend.base_device == "cuda"
    assert cfg.reward.backend.config.device == "auto"
    assert cfg.reward.backend.config.batch_size == 8
    assert cfg.stack.num_updates_per_batch == 2


def test_single_gpu_smoke_profile_composes_reduced_overrides() -> None:
    production = _compose("bagel_vllmomni_t2ti")
    smoke = _compose("bagel_vllmomni_t2ti_smoke")

    _assert_strict_t2ti_contract(smoke, expected_pairs=4)
    assert (smoke.num_devices, smoke.devices_per_node, smoke.batch_size) == (1, 1, 1)
    assert smoke.enable_fsdp_offload is True
    assert smoke.backend.fsdp_cfg.cpu_offload is True
    assert smoke.sampling.ar.samples_per_prompt == 4
    assert smoke.sampling.diffusion.samples_per_prompt == 1
    assert smoke.sampling.ar.max_new_tokens == 512
    assert smoke.sampling.diffusion.num_inference_steps == 14
    assert smoke.sampling.diffusion.scheduler.num_timesteps == 14
    assert smoke.sampling.diffusion.scheduler.num_sde_steps == 2
    assert smoke.reward.backend.base_device == "cpu"
    assert smoke.reward.backend.config.device == "cpu"
    assert smoke.reward.backend.config.batch_size == 2
    assert smoke.stack.num_updates_per_batch == 1

    # Smoke changes capacity, not the native two-stage or strict-sync contract.
    assert smoke.rollout == production.rollout
    assert smoke.sync == production.sync
    assert smoke.pipeline == production.pipeline
    assert smoke.sampling.diffusion.guidance_scale == 1.0
    assert smoke.sampling.diffusion.cfg_text_scale == 1.0
    assert smoke.sampling.diffusion.cfg_img_scale == 1.0


def test_launcher_resolves_explicit_scale_profiles() -> None:
    production = _resolve_launcher_profile("production")
    smoke = _resolve_launcher_profile("smoke")
    invalid = _resolve_launcher_profile("tiny")

    assert production.returncode == 0
    assert production.stdout.strip() == "unified_model/bagel_vllmomni_t2ti"
    assert smoke.returncode == 0
    assert smoke.stdout.strip() == "unified_model/bagel_vllmomni_t2ti_smoke"
    assert invalid.returncode == 2
    assert "expected production or smoke" in invalid.stderr
