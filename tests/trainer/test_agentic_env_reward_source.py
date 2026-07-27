"""Every env-reward trainer must read the environment's return the SAME way.

The barrier, colocate-partial, and fully-async env trainers differ only in their reward
SOURCE, and that source is one shared mixin. This file pins both halves of that claim:
the sentinel a failed trajectory receives (NaN, so GRPO excludes it), and the structural
fact that all three paths resolve to a single implementation.

Regression context: ``7ad5b349`` established NaN-not-0.0 for infrastructure faults —
"scoring an infrastructure fault as a genuine miss manufactures a gradient for every
sibling in the group" — but only reached the barrier trainer. The partial and async env
variants each carried their own copy of the method and kept ``0.0`` for months. Both the
parametrized sentinel test and the shared-identity test below fail on that state.
"""

from __future__ import annotations

import pytest
import torch

from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine
from unirl.trainer.agentic_env import AgenticEnvTrainer
from unirl.trainer.agentic_env_async import AsyncAgenticEnvTrainer
from unirl.trainer.agentic_partial import AgenticEnvPartialTrainer
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

ENV_TRAINERS = [AgenticEnvTrainer, AsyncAgenticEnvTrainer, AgenticEnvPartialTrainer]
ROOT = "r0:prompt:0:sample:0"


def _rewarded(reward: float, root: str = ROOT) -> Sample:
    """A completed trajectory carrying an env return, attached exactly as the engine does."""
    sample = Sample.request(Part.input([root], primitives={"text": Texts(texts=["go"])}))
    sample = sample.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
    sample = sample.with_filled_frontier(primitives={"text": Texts(texts=["open door"])})
    return AgenticRolloutEngine._attach_env_reward(sample, reward)


def _gen_less(root: str = ROOT) -> Sample:
    """A trajectory whose fault preceded the first turn: no gen Parts, so no reward."""
    return Sample.request(Part.input([root], primitives={"text": Texts(texts=["go"])}))


def _source(trainer_cls) -> object:
    """The reward step, on a trainer built without running __init__ (no GPUs, no Ray)."""
    return object.__new__(trainer_cls)._rewards_and_groups


@pytest.mark.parametrize("trainer_cls", ENV_TRAINERS, ids=lambda c: c.__name__)
def test_a_gen_less_trajectory_scores_nan_not_zero(trainer_cls) -> None:
    rewards, _ = _source(trainer_cls)(None, [_rewarded(1.0), _gen_less()], 0)

    assert rewards[0].item() == 1.0
    # 0.0 here would enter GRPO as a genuine miss and bias every sibling.
    assert torch.isnan(rewards[1])


@pytest.mark.parametrize("trainer_cls", ENV_TRAINERS, ids=lambda c: c.__name__)
def test_a_mid_trajectory_fault_keeps_the_engines_nan(trainer_cls) -> None:
    """The engine already attaches NaN when a fault hits after turn one; don't launder it."""
    rewards, _ = _source(trainer_cls)(None, [_rewarded(float("nan"))], 0)

    assert torch.isnan(rewards[0])


@pytest.mark.parametrize("trainer_cls", ENV_TRAINERS, ids=lambda c: c.__name__)
def test_a_healthy_env_return_is_read_verbatim(trainer_cls) -> None:
    rewards, group_ids = _source(trainer_cls)(None, [_rewarded(1.0), _rewarded(0.0)], 0)

    # 0.0 from the ENVIRONMENT is a real task failure and must survive as a real reward —
    # only a missing reward is a fault. This is the distinction the old code collapsed.
    assert rewards.tolist() == [1.0, 0.0]
    assert group_ids == [ROOT, ROOT]  # grouped by root id, so GRPO siblings group together


@pytest.mark.parametrize("trainer_cls", ENV_TRAINERS, ids=lambda c: c.__name__)
def test_group_ids_separate_distinct_roots(trainer_cls) -> None:
    other = "r0:prompt:1:sample:0"
    _, group_ids = _source(trainer_cls)(None, [_rewarded(1.0), _rewarded(1.0, other)], 0)

    assert group_ids == [ROOT, other]


def test_every_env_path_resolves_to_one_shared_reward_source() -> None:
    """The structural guard: a sentinel change cannot land on some paths and miss others.

    This is the test whose absence let 7ad5b349 go unpropagated. If a variant ever
    re-declares its own ``_rewards_and_groups``, this fails immediately.
    """
    sources = {c: c._rewards_and_groups for c in ENV_TRAINERS}

    assert len(set(sources.values())) == 1, f"env reward source has diverged: {sources}"
    assert AgenticEnvTrainer._rewards_and_groups.__qualname__.startswith("_EnvRewardSource.")


def test_a_failed_env_sibling_does_not_manufacture_gradient_for_its_group() -> None:
    """Why the sentinel matters: NaN is neutral, 0.0 is a signal the model never earned."""
    trainer = object.__new__(AgenticEnvTrainer)
    trainer.adv_normalization_scope = "group"
    trainer.normalize_adv_by_std = True
    group_ids = [ROOT] * 4

    rewards, _ = trainer._rewards_and_groups(None, [_rewarded(1.0), _rewarded(1.0), _rewarded(1.0), _gen_less()], 0)
    marked = trainer._group_advantages(rewards, group_ids)
    scored_as_zero = trainer._group_advantages(torch.tensor([1.0, 1.0, 1.0, 0.0]), group_ids)

    # Three identical rewards carry no signal, and the failure is neutral.
    assert torch.equal(marked, torch.zeros(4))
    # Had the failure scored 0.0, every sibling would get a gradient instead.
    assert not torch.equal(scored_as_zero, torch.zeros(4))
