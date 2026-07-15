"""Control-plane + provenance tests for the sync engine under threads.

The control verbs (``abort``/``pause``/``resume``) are sync methods reached via
the raw ``Worker.call`` RPC on a *different* thread than an in-flight
``generate`` (the ``worker_max_concurrency>1`` overlap) — they must interleave
with it, and ``abort`` must release parked requests so partials return.
Weight-version provenance is stamped onto the frontier gen ``Part`` by
``_stamp_weight_version`` (bumped per weight sync). Continuous batching is the
load-bearing property the sync rewrite must not lose: two concurrent callers'
requests stay in flight together, and a fast caller returns while a slow one
is still generating. All CPU-only against a fake backend.
"""

import threading

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from tests.rollout.engine._fakes import FakeEngine, build_request_batch  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

# --------------------------------------------------------------------------- #
# abort / pause / resume
# --------------------------------------------------------------------------- #


def test_controls_interleave_with_inflight_generate_and_abort_releases():
    """The worker_max_concurrency>1 overlap: while one (worker) thread is parked
    inside ``generate``, pause/resume/abort from a second thread reach the
    backend — flags flip, and ``abort`` releases the parked request so the
    generate completes (the best-effort-cancel-returns-partials path)."""
    engine = FakeEngine(concurrency=8)
    engine._backend.block_until_released = True
    batch = build_request_batch(P=1, n=2)

    result: dict = {}

    def drive():
        result["out"] = engine.generate(batch)  # parks in the backend hold-gate

    worker = threading.Thread(target=drive)
    worker.start()
    try:
        # Wait until a request body is actually in flight.
        assert engine._backend.entered.wait(timeout=5.0)

        engine.pause()
        assert engine._backend.paused is True
        engine.resume()
        assert engine._backend.paused is False

        # abort() flips the flag AND releases the parked request.
        engine.abort()
        assert engine._backend.aborted is True
    finally:
        worker.join(timeout=5.0)

    assert not worker.is_alive()  # the generate completed after the abort released it
    out = result["out"]
    assert "text" in out.parts[-1].primitives
    assert len(out.parts[-1].primitives["text"].texts) == 2

    engine.shutdown()


def test_controls_reach_backend_when_idle():
    """Sync controls always reach the backend — with nothing in flight they are
    harmless (the server-side no-op), never an error. (Contract change from the
    session-scoped loop, where idle controls were dropped engine-side.)"""
    engine = FakeEngine(concurrency=8)

    engine.pause()
    assert engine._backend.paused is True
    engine.resume()
    assert engine._backend.paused is False
    assert engine.abort() == []
    assert engine._backend.aborted is True

    engine.shutdown()


# --------------------------------------------------------------------------- #
# continuous batching — concurrent callers overlap; fast finishes first
# --------------------------------------------------------------------------- #


def test_concurrent_callers_overlap_and_finish_by_speed():
    """The continuous-batching regression the thread rewrite must not lose: two
    ``generate`` callers on two threads are in flight TOGETHER (peak == 2), and
    the fast one returns while the slow one is still generating (completion
    order ≠ submission order)."""
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=2, n=1)
    slow_group, fast_group = batch.split()
    prompts = list(batch.parts[0].primitives["text"].texts)
    engine._backend.delay_for = {prompts[0]: 0.25, prompts[1]: 0.0}  # slow first prompt

    finished: list = []
    lock = threading.Lock()

    def run(group, tag):
        engine.generate(group)
        with lock:
            finished.append(tag)

    slow = threading.Thread(target=run, args=(slow_group, "slow"))
    fast = threading.Thread(target=run, args=(fast_group, "fast"))
    slow.start()
    # Ensure the slow request is in flight before submitting the fast one, so
    # "fast finished first" can only come from genuine overlap.
    assert engine._backend.entered.wait(timeout=5.0)
    fast.start()
    fast.join(timeout=5.0)
    with lock:
        first_done = list(finished)
    slow.join(timeout=5.0)

    assert first_done == ["fast"]  # fast returned while slow was still in flight
    assert finished == ["fast", "slow"]
    assert engine._backend.peak == 2  # both were in flight together

    engine.shutdown()


# --------------------------------------------------------------------------- #
# weight-version provenance
# --------------------------------------------------------------------------- #


def test_part_weight_version_defaults_to_none():
    """A freshly built input Part and a forked gen shell carry no weight version."""
    head = Part.input(["p0"], primitives={"text": Texts(texts=["hello"])})
    assert head.weight_version is None
    shell = head.fork(2, sampling_params=ARSamplingParams(samples_per_prompt=2))
    assert shell.weight_version is None


def test_stamp_marks_frontier_and_survives_split_then_concat():
    """_stamp_weight_version stamps the engine's version onto the frontier gen Part
    only, and that stamp survives a split()/concat() round-trip."""
    engine = FakeEngine(concurrency=8)
    engine._weight_version = 5
    batch = build_request_batch(P=2, n=2)

    stamped = engine._stamp_weight_version(batch)
    assert stamped.parts[-1].weight_version == 5
    assert stamped.parts[0].weight_version is None  # the input Part is untouched

    round_tripped = Sample.concat(stamped.split())
    assert round_tripped.parts[-1].weight_version == 5
    assert round_tripped == stamped

    engine.shutdown()


def test_weight_version_bump_stamps_later_gens():
    """A weight sync bumps _weight_version (the real engine does this in its
    update_weights_* verbs); generations after the bump carry the new version, so
    the two cohorts keep distinct provenance."""
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=2, n=2)

    before = engine.generate(batch)
    assert before.parts[-1].weight_version == 0  # default starting version

    engine._weight_version += 1  # a weight update bumps the counter
    after = engine.generate(batch)
    assert after.parts[-1].weight_version == 1
    assert before.parts[-1].weight_version != after.parts[-1].weight_version

    engine.shutdown()
