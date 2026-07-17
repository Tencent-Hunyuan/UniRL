from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir

import unirl.train_unified_model as train_entrypoint
from unirl.distributed.group.device_pool import _cuda_allocator_env_vars
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
    assert cfg.pipeline.t2ti_flow_many_enabled is True
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
    assert cfg.cuda_allocator_conf == (
        "expandable_segments:True,garbage_collection_threshold:0.95,per_process_memory_fraction:0.90"
    )
    assert cfg.park_optimizer_state_during_rollout is True
    assert cfg.park_optimizer_state_during_train is True
    assert cfg.backend.fsdp_cfg.cpu_offload is False
    assert cfg.algorithm.image.stage_prepared_replay_to_cpu is True
    assert cfg.sampling.ar.samples_per_prompt == 24
    assert cfg.sampling.diffusion.samples_per_prompt == 1
    assert cfg.sampling.ar.max_new_tokens == 1024
    assert cfg.sampling.diffusion.num_inference_steps == 25
    assert cfg.sampling.diffusion.scheduler.num_timesteps == 25
    assert cfg.sampling.diffusion.scheduler.num_sde_steps == 3
    assert cfg.reward.backend.base_device == "cuda"
    assert cfg.reward.backend.config.device == "auto"
    assert cfg.reward.backend.config.batch_size == 8
    assert cfg.reward.park_backend_between_calls is True
    assert cfg.stack.num_updates_per_batch == 2
    assert cfg.stack.empty_cache_after_image_micro is True
    assert cfg.stack.image_micro_empty_cache_interval == 1
    assert cfg.stack.image_micro_empty_cache_min_free_gb == 0.0
    assert cfg.stack.empty_cache_after_optimizer is True
    assert cfg.stack.cuda_peak_telemetry is True


def test_single_gpu_smoke_profile_composes_reduced_overrides() -> None:
    production = _compose("bagel_vllmomni_t2ti")
    smoke = _compose("bagel_vllmomni_t2ti_smoke")

    _assert_strict_t2ti_contract(smoke, expected_pairs=4)
    assert (smoke.num_devices, smoke.devices_per_node, smoke.batch_size) == (1, 1, 1)
    assert smoke.enable_fsdp_offload is True
    assert smoke.park_optimizer_state_during_rollout is False
    assert smoke.park_optimizer_state_during_train is False
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
    assert smoke.reward.park_backend_between_calls is False
    assert smoke.stack.num_updates_per_batch == 1
    assert smoke.stack.empty_cache_after_image_micro is False
    assert smoke.stack.image_micro_empty_cache_interval == 1
    assert smoke.stack.image_micro_empty_cache_min_free_gb == 0.0
    assert smoke.stack.empty_cache_after_optimizer is False
    assert smoke.stack.cuda_peak_telemetry is False
    assert smoke.algorithm.image.stage_prepared_replay_to_cpu is False

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


def test_unified_model_entrypoint_wires_optimizer_parking(monkeypatch) -> None:
    cfg = _compose("bagel_vllmomni_t2ti")
    captured = {}
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)

    class _Trainer:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def train(self, **_kwargs) -> None:
            return None

    monkeypatch.setattr(train_entrypoint, "UnifiedModelTrainer", _Trainer)
    train_entrypoint.main.__wrapped__(cfg)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == cfg.cuda_allocator_conf
    assert captured["park_optimizer_state_during_rollout"] is True
    assert captured["park_optimizer_state_during_train"] is True


def test_unified_model_entrypoint_preserves_allocator_override(monkeypatch) -> None:
    cfg = _compose("bagel_vllmomni_t2ti")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    train_entrypoint._configure_cuda_allocator(cfg)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:64"


def test_device_pool_propagates_allocator_policy_to_ray_workers(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "per_process_memory_fraction:0.90")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.95")

    assert _cuda_allocator_env_vars() == {
        "PYTORCH_CUDA_ALLOC_CONF": "per_process_memory_fraction:0.90",
        "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.95",
    }
