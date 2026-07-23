"""``forward_batch_size`` must refuse a multi-stage lineage instead of dropping a stage.

Chunking assumes only ``parts[-1]`` is generated: it captures ``parts[:-1]`` once and keeps
only ``chunk.parts[-1]``. Given ``[input, ar, diffusion]`` a composed pipeline fills BOTH gen
Parts per chunk, so the interior stage is regenerated and then discarded, and the Sample comes
back carrying the original empty shell.
"""

from __future__ import annotations

import threading

import pytest

from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams


class _TwoStagePipeline:
    """PE-shaped: one ``generate`` fills the AR Part *and* the diffusion Part."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, sample: Sample) -> Sample:
        self.calls += 1
        parts = [
            part.fill(primitives={"text": Texts(texts=[f"out-{row}" for row in range(part.batch_size)])})
            if part.is_gen
            else part
            for part in sample.parts
        ]
        return sample.with_parts(parts)


def _bare_engine(*, forward_batch_size: int | None) -> TrainsideRolloutEngine:
    engine = object.__new__(TrainsideRolloutEngine)
    engine.pipeline = _TwoStagePipeline()
    engine._models = []
    engine.schedule_policy = None
    engine.forward_batch_size = forward_batch_size
    engine._generate_lock = threading.Lock()
    engine._shutdown_requested = False
    return engine


def _multi_stage(*, branch: int = 4) -> Sample:
    root = Part.input(["r0:prompt:0:sample:0"], primitives={"text": Texts(texts=["a cat"])})
    return (
        Sample.request(root)
        .fork(branch, sampling_params=ARSamplingParams(samples_per_prompt=branch))
        .fork(1, sampling_params=DiffusionSamplingParams(samples_per_prompt=1))
    )


def _single_stage(*, branch: int = 4) -> Sample:
    root = Part.input(["r0:prompt:0:sample:0"], primitives={"text": Texts(texts=["a cat"])})
    return Sample.request(root).fork(branch, sampling_params=DiffusionSamplingParams(samples_per_prompt=branch))


def test_forward_batch_size_rejects_a_multi_stage_sample() -> None:
    with pytest.raises(ValueError, match="forward_batch_size"):
        _bare_engine(forward_batch_size=2)._generate_core(_multi_stage())


def test_the_rejection_does_not_depend_on_this_batch_actually_chunking() -> None:
    # branch=2 fits under fbs=4, so the old code would have taken the whole-sample path
    # and silently worked — until the fan-out grew. The guard must fire regardless.
    with pytest.raises(ValueError, match="forward_batch_size"):
        _bare_engine(forward_batch_size=4)._generate_core(_multi_stage(branch=2))


def test_a_multi_stage_sample_without_chunking_still_fills_every_gen_part() -> None:
    out = _bare_engine(forward_batch_size=None)._generate_core(_multi_stage())

    assert out.parts[-2].primitives["text"].texts == ["out-0", "out-1", "out-2", "out-3"]
    assert out.parts[-1].primitives["text"].texts == ["out-0", "out-1", "out-2", "out-3"]


def test_single_stage_chunking_is_unchanged() -> None:
    engine = _bare_engine(forward_batch_size=2)
    # A 3-Part multimodal request is still single-stage (V2V chains a condition input
    # Part), so the guard must key on gen Parts, not on len(parts).
    out = engine._generate_core(_single_stage())

    assert engine.pipeline.calls == 2
    assert out.parts[-1].batch_size == 4
