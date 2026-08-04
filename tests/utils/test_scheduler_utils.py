"""CPU unit tests for AllSDEScheduler.sde_candidate_pool (FlashGRPO per-sample pool)."""

from __future__ import annotations

from unirl.utils.scheduler_utils import AllSDEScheduler


def test_candidate_pool_matches_wan_recipe_fraction_range() -> None:
    """WAN recipe (20 steps, fraction [0.0, 0.5]) → first 10 steps [0, 10)."""
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=1)
    assert scheduler.sde_candidate_pool() == list(range(0, 10))


def test_candidate_pool_independent_of_num_sde_steps() -> None:
    """The candidate pool is how many steps are ELIGIBLE, not how many are drawn."""
    one = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=1)
    three = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.0, 0.5], num_sde_steps=3)
    assert one.sde_candidate_pool() == three.sde_candidate_pool() == list(range(0, 10))


def test_candidate_pool_full_range_for_default_fraction() -> None:
    """Default fraction 1.0 → the whole [0, num_timesteps) schedule."""
    scheduler = AllSDEScheduler(num_timesteps=8)
    assert scheduler.sde_candidate_pool() == list(range(0, 8))


def test_candidate_pool_offset_fraction_window() -> None:
    """A non-zero start offsets the pool: fraction [0.25, 0.75] of 20 → [5, 15)."""
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.25, 0.75])
    assert scheduler.sde_candidate_pool() == list(range(5, 15))


def test_get_sde_indices_always_within_candidate_pool() -> None:
    """Every step get_sde_indices can draw is a member of the candidate pool."""
    scheduler = AllSDEScheduler(num_timesteps=20, timestep_fraction=[0.25, 0.75], num_sde_steps=2)
    pool = set(scheduler.sde_candidate_pool())
    for step in range(6):
        assert scheduler.get_sde_indices(step).issubset(pool)
