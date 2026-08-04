"""CPU unit tests for the FlashGRPO trainer orchestration (DiffusionTrainer).

Covers the three per-sample helpers that fan a rollout into one generate per
SDE-step group and stamp each sample's own step:

* ``_flash_candidate_pool`` — reads the scheduler's eligible-step pool (loud on
  a missing / pool-less / empty scheduler);
* ``_build_flash_generate_jobs`` — draws one i.i.d. step per prompt (seeded on
  ``rollout_id``) and groups prompts that share a step into one request;
* ``_stamp_sde_index_per_sample`` — stamps each track's segment with its group's
  single SDE index.

The helpers run on a bare ``DiffusionTrainer`` shell (``__new__`` + the few
attributes each reads) so no engine / backend / reward is needed; the heavy
``_build_req`` is replaced by a recorder to assert the grouping alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unirl.trainer.diffusion import DiffusionTrainer
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.segments.latent import make_video_segment
from unirl.utils.scheduler_utils import AllSDEScheduler


def _trainer_with_scheduler(scheduler, num_inference_steps: int = 20) -> DiffusionTrainer:
    """A bare trainer whose sampling_params expose ``scheduler`` + step count under 'diffusion'.

    ``num_inference_steps`` is the rollout length T the uniform-K guard checks the
    candidate pool against (a pool reaching step T-1 stores mismatched-K latents).
    """
    trainer = object.__new__(DiffusionTrainer)
    trainer.sampling_params = {
        "diffusion": SimpleNamespace(scheduler=scheduler, num_inference_steps=num_inference_steps)
    }
    return trainer


def _inputs(n_prompts: int) -> RolloutInputs:
    return RolloutInputs(
        sample_ids=[f"p{i}" for i in range(n_prompts)],
        group_ids=[f"g{i}" for i in range(n_prompts)],
    )


def _recorder(calls: list):
    """Stand-in for _build_req recording (group sample_ids, override, rollout_id)."""

    def fake_build_req(group_inputs, rollout_id, *, sde_index_override):
        calls.append((sorted(group_inputs.sample_ids), int(sde_index_override), int(rollout_id)))
        return f"req@{sde_index_override}"

    return fake_build_req


# ---- _flash_candidate_pool ------------------------------------------------


def test_candidate_pool_reads_scheduler() -> None:
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=1)
    assert _trainer_with_scheduler(scheduler)._flash_candidate_pool() == list(range(0, 10))


def test_candidate_pool_missing_scheduler_raises() -> None:
    with pytest.raises(ValueError, match="sde_candidate_pool"):
        _trainer_with_scheduler(None)._flash_candidate_pool()


def test_candidate_pool_scheduler_without_accessor_raises() -> None:
    with pytest.raises(ValueError, match="sde_candidate_pool"):
        _trainer_with_scheduler(SimpleNamespace())._flash_candidate_pool()


def test_candidate_pool_empty_raises() -> None:
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.0])  # empty range
    with pytest.raises(ValueError, match="empty"):
        _trainer_with_scheduler(scheduler)._flash_candidate_pool()


def test_candidate_pool_rejects_last_step_mixed_k() -> None:
    """A pool reaching the final denoising step (T-1) mixes K=2 and K=3 rollout
    groups (uncatable), so the pool is rejected at build rather than mid-training."""
    # fraction [0.5, 1.0] over T=20 → pool [10, 20); step 19 == num_inference_steps-1.
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.5, 1.0], num_sde_steps=1)
    with pytest.raises(ValueError, match="mixed-K"):
        _trainer_with_scheduler(scheduler, num_inference_steps=20)._flash_candidate_pool()


# ---- _check_flash_rectification_pool --------------------------------------


def test_rectification_pool_match_passes() -> None:
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=1)
    trainer = _trainer_with_scheduler(scheduler)
    trainer._check_flash_rectification_pool({"rectification_indices": list(range(10))})  # must not raise


def test_rectification_pool_mismatch_raises() -> None:
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=1)
    trainer = _trainer_with_scheduler(scheduler)
    with pytest.raises(ValueError, match="must equal the SDE candidate pool"):
        trainer._check_flash_rectification_pool({"rectification_indices": [0, 1, 2]})


def test_rectification_pool_none_raises() -> None:
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=1)
    trainer = _trainer_with_scheduler(scheduler)
    with pytest.raises(ValueError, match="rectification_indices"):
        trainer._check_flash_rectification_pool({})


# ---- _build_flash_generate_jobs -------------------------------------------


def test_jobs_partition_prompts_across_step_groups() -> None:
    """Every prompt lands in exactly one job; jobs are one-per-drawn-step ascending."""
    pool = list(range(10))
    trainer = _trainer_with_scheduler(AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5]))
    calls: list = []
    trainer._build_req = _recorder(calls)
    inputs = _inputs(24)
    rollout_id = 5

    jobs = trainer._build_flash_generate_jobs(inputs, rollout_id)

    # Reproduce the exact per-prompt draw the implementation uses.
    drawn = np.random.default_rng(rollout_id).choice(pool, size=24, replace=True)
    expected_by_step: dict = {}
    for pos, step in enumerate(drawn):
        expected_by_step.setdefault(int(step), []).append(pos)

    assert [step for step, _ in jobs] == sorted(expected_by_step)
    assert [req for _, req in jobs] == [f"req@{step}" for step in sorted(expected_by_step)]

    # Each group holds exactly its step's prompts, stamped with that override.
    for (step, _), (group_ids, override, rid) in zip(jobs, calls):
        assert override == step
        assert rid == rollout_id
        assert group_ids == sorted(f"p{pos}" for pos in expected_by_step[step])

    # Union across groups is the full prompt set, no duplicates.
    covered = sorted(sid for group_ids, _, _ in calls for sid in group_ids)
    assert covered == sorted(inputs.sample_ids)


def test_jobs_are_seed_deterministic() -> None:
    """Same rollout_id → identical assignment (resume-determinism)."""
    trainer = _trainer_with_scheduler(AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5]))
    first: list = []
    trainer._build_req = _recorder(first)
    trainer._build_flash_generate_jobs(_inputs(16), rollout_id=7)
    second: list = []
    trainer._build_req = _recorder(second)
    trainer._build_flash_generate_jobs(_inputs(16), rollout_id=7)
    assert first == second


def test_jobs_all_steps_in_pool() -> None:
    pool = set(range(10))
    trainer = _trainer_with_scheduler(AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5]))
    trainer._build_req = _recorder([])
    jobs = trainer._build_flash_generate_jobs(_inputs(24), rollout_id=3)
    assert all(step in pool for step, _ in jobs)


# ---- _stamp_sde_index_per_sample ------------------------------------------


def test_stamp_sets_full_index_tensor() -> None:
    trainer = object.__new__(DiffusionTrainer)
    seg = make_video_segment(latents=torch.zeros(3, 2, 3), sde_logp=torch.zeros(3, 1))
    resp = RolloutResp(tracks={"video": RolloutTrack(sample_ids=["a", "b", "c"], segment=seg)})

    trainer._stamp_sde_index_per_sample(resp, sde_index=7)

    stamped = resp.tracks["video"].segment.sde_index_per_sample
    assert stamped.tolist() == [7, 7, 7]
    assert stamped.dtype == torch.long
    assert stamped.shape == (3,)


def test_stamp_skips_segmentless_track() -> None:
    trainer = object.__new__(DiffusionTrainer)
    resp = RolloutResp(tracks={"empty": RolloutTrack(sample_ids=[], segment=None)})
    trainer._stamp_sde_index_per_sample(resp, sde_index=4)  # must not raise
    assert resp.tracks["empty"].segment is None
