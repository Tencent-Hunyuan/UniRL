from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.base import RolloutCapabilities
from unirl.trainer.contrastive import ContrastiveRolloutConfig, select_top_bottom_indices
from unirl.trainer.diffusion import DiffusionTrainer, _validate_rollout_sampling_capabilities
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams


def _scored_part(rewards: list[float], *, groups: int, group_size: int) -> Part:
    assert len(rewards) == groups * group_size
    return Part(
        sample_ids=[f"p{group}/candidate/{row}" for group in range(groups) for row in range(group_size)],
        rewards=torch.tensor(rewards, dtype=torch.float32),
    )


def test_top_bottom_selection_is_group_contiguous_and_stable() -> None:
    part = _scored_part(
        [0.1, 0.9, 0.3, 0.8, 0.2, 0.4, 0.4, 0.0],
        groups=2,
        group_size=4,
    )
    indices = select_top_bottom_indices(part, top_k=1, bottom_k=1)
    assert indices.tolist() == [1, 0, 5, 7]

    selected = part.select(indices)
    assert selected.group_ids == [
        "p0/candidate",
        "p0/candidate",
        "p1/candidate",
        "p1/candidate",
    ]


def test_selection_rejects_overlapping_top_and_bottom() -> None:
    part = _scored_part([0.0, 1.0], groups=1, group_size=2)
    try:
        select_top_bottom_indices(part, top_k=2, bottom_k=1)
    except ValueError as exc:
        assert "exceeds scout group size" in str(exc)
    else:
        raise AssertionError("expected selection overlap to fail")


def test_tied_rewards_never_duplicate_top_and_bottom_rows() -> None:
    part = _scored_part([1.0, 1.0, 1.0, 1.0], groups=1, group_size=4)
    indices = select_top_bottom_indices(part, top_k=1, bottom_k=1)
    assert indices.tolist() == [0, 1]


def test_contrastive_counts_require_integers() -> None:
    try:
        ContrastiveRolloutConfig(top_k=1.5)
    except TypeError as exc:
        assert "top_k must be an integer" in str(exc)
    else:
        raise AssertionError("expected fractional top_k to fail")
    with pytest.raises(TypeError, match="trace_dir"):
        ContrastiveRolloutConfig(trace_dir=True)
    with pytest.raises(ValueError, match="policy_snapshot_id"):
        ContrastiveRolloutConfig(trace_dir="/tmp/traces")


def test_unsupported_rollout_execution_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="rollout_precision='fp8' is unsupported"):
        _validate_rollout_sampling_capabilities(
            RolloutCapabilities(),
            scout_sampling=DiffusionSamplingParams(rollout_precision="fp8"),
        )
    with pytest.raises(ValueError, match="image-resize capability"):
        _validate_rollout_sampling_capabilities(
            RolloutCapabilities(),
            scout_sampling=DiffusionSamplingParams(reward_image_size=224),
        )


def test_regen_shell_preserves_exact_selected_initial_noise() -> None:
    root = Part.input(
        ["prompt"],
        primitives={"text": Texts(texts=["a prompt"])},
    )
    scout_params = DiffusionSamplingParams(
        samples_per_prompt=4,
        num_samples_per_prompt=4,
        num_inference_steps=6,
        seed=123,
        init_noise_latent_shape=[2, 3, 3],
        sde_indices=[],
    )
    scout = Sample.request(root).fork(4, sampling_params=scout_params)
    full_xt = NoiseRecipe.from_sample(scout).resolve()
    assert full_xt is not None

    selected = scout.parts[-1].select(torch.tensor([3, 0]))
    regen_params = DiffusionSamplingParams(
        samples_per_prompt=2,
        num_samples_per_prompt=2,
        num_inference_steps=10,
        seed=123,
        init_noise_latent_shape=[2, 3, 3],
        sde_indices=[],
    )
    shell = DiffusionTrainer._selected_regen_shell(selected, regen_params)
    regen = scout.replace_frontier(shell)
    regen_xt = NoiseRecipe.from_sample(regen).resolve()
    assert regen_xt is not None
    assert torch.equal(regen_xt, full_xt.index_select(0, torch.tensor([3, 0])))
    assert shell.segment is None
    assert shell.primitives == {}
    assert shell.sampling_params.sigmas is None


def test_naive_sampling_inherits_main_generation_policy() -> None:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer._contrastive = ContrastiveRolloutConfig(mode="naive", top_k=1, bottom_k=1)
    trainer.sampling_params = {
        "diffusion": DiffusionSamplingParams(
            samples_per_prompt=2,
            num_inference_steps=10,
            guidance_scale=1.5,
            seed=123,
            rollout_precision="bf16",
        )
    }
    trainer._scout_sampling_params = {
        "diffusion": DiffusionSamplingParams(
            samples_per_prompt=8,
            num_inference_steps=6,
            guidance_scale=7.0,
            seed=999,
            rollout_precision="fp8",
            reward_image_size=224,
        )
    }

    trainer._normalize_contrastive_sampling()
    effective = trainer._scout_sampling_params["diffusion"]
    assert effective.samples_per_prompt == 8
    assert effective.num_inference_steps == 10
    assert effective.guidance_scale == 1.5
    assert effective.seed == 123
    assert effective.rollout_precision == "bf16"
    assert effective.reward_image_size == 224


def test_contrastive_scoring_rejects_non_finite_rewards() -> None:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer.reward = type("_Reward", (), {"score_and_attach": lambda self, sample: sample})()
    trainer._reward_phase = nullcontext
    root = Part.input(["prompt"], primitives={"text": Texts(texts=["a prompt"])})
    sample = Sample.request(root).fork(1, sampling_params=DiffusionSamplingParams())
    sample.parts[-1].rewards = torch.tensor([float("nan")])

    try:
        trainer._score_generated_sample(sample)
    except ValueError as exc:
        assert "non-finite values" in str(exc)
    else:
        raise AssertionError("expected non-finite reward to fail")


def test_trace_writer_records_policy_and_schema_identity(tmp_path: Path) -> None:
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer._contrastive = ContrastiveRolloutConfig(
        top_k=1,
        bottom_k=1,
        trace_dir=str(tmp_path),
        policy_snapshot_id="immutable-checkpoint-v1",
    )
    trainer._scout_sampling_params = {
        "diffusion": DiffusionSamplingParams(num_inference_steps=6, rollout_precision="fp8")
    }
    trainer._trace_model_fingerprint = "a" * 64
    trainer._trace_pipeline_fingerprint = "f" * 64
    trainer._trace_reward_fingerprint = "d" * 64
    trainer._write_contrastive_trace(
        0,
        policy_version=0,
        groups=[
            {
                "group_id": "prompt",
                "prompt_sha256": "b" * 64,
                "noise_sha256": "c" * 64,
                "candidates": [],
            }
        ],
    )
    payload = json.loads((tmp_path / "rollout_000000.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["policy_version"] == 0
    assert payload["policy_snapshot_id"] == "immutable-checkpoint-v1"
    assert payload["model_fingerprint"] == "a" * 64
    assert payload["pipeline_fingerprint"] == "f" * 64
    assert payload["reward_fingerprint"] == "d" * 64
    assert len(payload["comparison_fingerprint"]) == 64


def test_failed_weight_sync_skips_rollout_rpc_until_bounded_cleanup() -> None:
    sleeps = []
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer._layout = "separate"
    trainer._enable_fsdp_offload = False
    trainer._rollout_is_trainside = False
    trainer._uses_ema = False
    trainer._rollout_rpc_blocked = False
    trainer.rollout = SimpleNamespace(
        wake_up=lambda: None,
        sleep=lambda: sleeps.append(True),
    )
    trainer.weight_sync = SimpleNamespace(sync=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")))

    with pytest.raises(RuntimeError, match="sync failed"):
        trainer._generate_with_residency(Sample(), sync_weights=True, sleep_rollout=True)
    assert sleeps == []
    assert trainer._rollout_rpc_blocked


def test_empty_eval_sync_failure_skips_rollout_sleep() -> None:
    sleeps = []
    trainer = DiffusionTrainer.__new__(DiffusionTrainer)
    trainer._rollout_rpc_blocked = False
    trainer.rollout = SimpleNamespace(
        wake_up=lambda: None,
        sleep=lambda: sleeps.append(True),
    )
    trainer.weight_sync = SimpleNamespace(sync=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")))
    with pytest.raises(RuntimeError, match="sync failed"):
        trainer._prepare_empty_evaluation(sync_weights=True, sleep_rollout=True)
    assert sleeps == []
    assert trainer._rollout_rpc_blocked
