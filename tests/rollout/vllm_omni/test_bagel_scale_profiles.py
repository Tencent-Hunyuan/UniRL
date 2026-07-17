from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import unirl.train_unified_model as train_entrypoint
from unirl.distributed.group.device_pool import DevicePool, _cuda_allocator_env_vars
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


def _run_launcher_allocator_helper(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; configure_bagel_cuda_allocator_default; '
                "printf 'cuda=%s\\ngeneric=%s\\n' "
                '"${PYTORCH_CUDA_ALLOC_CONF-<unset>}" "${PYTORCH_ALLOC_CONF-<unset>}"'
            ),
            "allocator-test",
            str(LAUNCHER),
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _run_launcher_proxy_helper(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; configure_bagel_proxy; '
                "printf 'http=%s\\nhttps=%s\\nHTTP=%s\\nHTTPS=%s\\nno=%s\\nNO=%s\\n' "
                '"${http_proxy-<unset>}" "${https_proxy-<unset>}" '
                '"${HTTP_PROXY-<unset>}" "${HTTPS_PROXY-<unset>}" '
                '"${no_proxy-<unset>}" "${NO_PROXY-<unset>}"'
            ),
            "proxy-test",
            str(LAUNCHER),
        ],
        check=False,
        capture_output=True,
        env=env,
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


def test_launcher_does_not_install_a_site_specific_proxy_by_default() -> None:
    env = dict(os.environ)
    for name in ("STAR_PROXY_URL", "STAR_NO_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(name, None)
    env.pop("no_proxy", None)
    env.pop("NO_PROXY", None)

    result = _run_launcher_proxy_helper(env)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "http=<unset>\nhttps=<unset>\nHTTP=<unset>\nHTTPS=<unset>\nno=<unset>\nNO=<unset>\n"
    )


def test_launcher_supports_explicit_proxy_injection() -> None:
    env = dict(os.environ)
    env["STAR_PROXY_URL"] = "http://proxy.example:3128"
    env["STAR_NO_PROXY"] = "localhost,.example.com"

    result = _run_launcher_proxy_helper(env)

    assert result.returncode == 0, result.stderr
    assert result.stdout == (
        "http=http://proxy.example:3128\n"
        "https=http://proxy.example:3128\n"
        "HTTP=http://proxy.example:3128\n"
        "HTTPS=http://proxy.example:3128\n"
        "no=localhost,.example.com\n"
        "NO=localhost,.example.com\n"
    )


def test_unified_model_entrypoint_wires_optimizer_parking(monkeypatch) -> None:
    cfg = _compose("bagel_vllmomni_t2ti")
    captured = {}
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)

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
    override = "max_split_size_mb:64,per_process_memory_fraction:0.85"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", override)

    train_entrypoint._configure_cuda_allocator(cfg)

    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == override


def test_unified_model_entrypoint_preserves_valid_generic_alias(monkeypatch) -> None:
    cfg = _compose("bagel_vllmomni_t2ti")
    override = "garbage_collection_threshold:0.8,per_process_memory_fraction:0.80"
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", override)

    train_entrypoint._configure_cuda_allocator(cfg)

    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ
    assert os.environ["PYTORCH_ALLOC_CONF"] == override


@pytest.mark.parametrize(
    "override",
    [
        "max_split_size_mb:64",
        "per_process_memory_fraction:0.91",
        "per_process_memory_fraction:0",
        "per_process_memory_fraction:nan",
        "per_process_memory_fraction:inf",
        "per_process_memory_fraction:not-a-number",
    ],
)
def test_production_flow_many_rejects_unsafe_cuda_allocator_override(monkeypatch, override: str) -> None:
    cfg = _compose("bagel_vllmomni_t2ti")
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", override)
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "per_process_memory_fraction:0.80")

    with pytest.raises(ValueError, match="finite per_process_memory_fraction") as exc_info:
        train_entrypoint._configure_cuda_allocator(cfg)

    assert "PYTORCH_CUDA_ALLOC_CONF" in str(exc_info.value)
    assert "takes precedence over PYTORCH_ALLOC_CONF" in str(exc_info.value)


def test_non_production_allocator_paths_do_not_require_cap(monkeypatch) -> None:
    smoke = _compose("bagel_vllmomni_t2ti_smoke")
    flow_many_off = _compose("bagel_vllmomni_t2ti")
    flow_many_off.pipeline.t2ti_flow_many_enabled = False
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")

    train_entrypoint._configure_cuda_allocator(smoke)
    train_entrypoint._configure_cuda_allocator(flow_many_off)


def test_launcher_allocator_default_does_not_shadow_generic_alias() -> None:
    env = dict(os.environ)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    override = "per_process_memory_fraction:0.80"
    env["PYTORCH_ALLOC_CONF"] = override

    result = _run_launcher_allocator_helper(env)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"cuda=<unset>\ngeneric={override}\n"


def test_device_pool_propagates_allocator_policy_to_ray_workers(monkeypatch) -> None:
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "per_process_memory_fraction:0.90")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "garbage_collection_threshold:0.95")

    assert _cuda_allocator_env_vars() == {
        "PYTORCH_CUDA_ALLOC_CONF": "per_process_memory_fraction:0.90",
        "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.95",
    }


def test_device_pool_lazy_slot_receives_allocator_construction_snapshot(monkeypatch) -> None:
    original = "per_process_memory_fraction:0.90"
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", original)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    pool = DevicePool(
        num_devices=1,
        devices_per_node=1,
        workers_per_device=2,
        transport_kind="gpu_store",
    )
    pool._device_to_workers[0] = [object()]
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "per_process_memory_fraction:0.50")
    captured = {}

    def _spawn_worker(device_id, slot, env_vars=None):
        captured.update(device_id=device_id, slot=slot, env_vars=env_vars)
        return "lazy-worker"

    monkeypatch.setattr(pool, "_spawn_worker", _spawn_worker)

    assert pool._get_or_create_worker(0, 1) == "lazy-worker"
    assert captured == {
        "device_id": 0,
        "slot": 1,
        "env_vars": {"PYTORCH_CUDA_ALLOC_CONF": original},
    }
