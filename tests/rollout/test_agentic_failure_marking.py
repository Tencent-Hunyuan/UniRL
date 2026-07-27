"""An infrastructure fault must be marked failed, not scored as a legitimate miss.

``_run_one`` isolates faults so one bad trajectory cannot sink the drain, but it must not
hand the trainer a trajectory that looks like a model which simply answered badly: a
backend outage, tool timeout, or context overflow would then enter GRPO as a real low
reward and bias every sibling in its group.
"""

from __future__ import annotations

import torch

from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine
from unirl.trainer.agentic import AgenticTrainer, _is_failed
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams


class _Env:
    def reset(self, task: Sample) -> Sample:
        return task

    def step(self, sample: Sample):
        return None, False, {}  # (observation, done, info) -> keep going


class _FlakyInner:
    """Inner single-turn engine that dies on the Nth decode."""

    def __init__(self, fail_on_call: int) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call

    def generate(self, sample: Sample) -> Sample:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("SGLang backend connection refused (HTTP 502)")
        rows = sample.parts[-1].batch_size
        return sample.with_filled_frontier(primitives={"text": Texts(texts=["<answer>4</answer>"] * rows)})


def _bare_engine(*, fail_on_call: int) -> AgenticRolloutEngine:
    engine = object.__new__(AgenticRolloutEngine)
    engine._env = _Env()
    engine._inner = _FlakyInner(fail_on_call)
    engine._max_turns = 4
    engine._stopping = False
    engine._sp = ARSamplingParams(samples_per_prompt=1)
    return engine


def _task() -> Sample:
    return Sample.request(Part.input(["r0:prompt:0:sample:0"], primitives={"text": Texts(texts=["2+2?"])}))


def test_a_fault_before_the_first_token_yields_a_gen_less_failed_trajectory() -> None:
    sample, done = _bare_engine(fail_on_call=1)._run_one(_task())

    assert done is True  # still terminal: the drain must not stall
    assert sample.gen_parts() == []
    assert _is_failed(sample)


def test_a_mid_trajectory_fault_marks_the_terminal_part_with_nan() -> None:
    sample, done = _bare_engine(fail_on_call=3)._run_one(_task())

    assert done is True
    assert len(sample.gen_parts()) == 2  # the turns completed before the fault
    assert torch.isnan(sample.gen_parts()[-1].rewards).all()
    assert _is_failed(sample)


def test_a_healthy_trajectory_is_not_marked_failed() -> None:
    engine = _bare_engine(fail_on_call=0)  # never fails; stops at max_turns
    sample, done = engine._run_one(_task())

    assert done is True
    assert not _is_failed(sample)


def test_a_failed_sibling_is_excluded_from_its_group_statistics() -> None:
    trainer = object.__new__(AgenticTrainer)
    trainer.adv_normalization_scope = "group"
    trainer.normalize_adv_by_std = True
    group_ids = ["r0:prompt:0:sample:0"] * 4

    marked = trainer._group_advantages(torch.tensor([1.0, 1.0, 1.0, float("nan")]), group_ids)
    scored_as_zero = trainer._group_advantages(torch.tensor([1.0, 1.0, 1.0, 0.0]), group_ids)

    # Excluded: three identical rewards carry no signal, and the failure is neutral.
    assert torch.equal(marked, torch.zeros(4))
    # Scored as a real miss it would instead manufacture a gradient for every sibling.
    assert not torch.equal(scored_as_zero, torch.zeros(4))
