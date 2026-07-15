"""Contract tests for the sync generation path under concurrent callers.

``generate`` is the ONE generation interface (LIN-522 → sync-only contract): it
must fill the whole Sample in group-by-parent order, hand the backend the
per-prompt wire in batch order, and hold one SHARED concurrency bound across
groups AND across concurrent caller threads (the agentic drain calls it from
one thread per trajectory). All CPU-only against a fake backend.
"""

import threading

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from tests.rollout.engine._fakes import (  # noqa: E402
    FakeEngine,
    build_request_batch,
    raw_text_for,
)
from unirl.types.sample import Sample  # noqa: E402


def test_generate_fills_group_by_parent_order():
    """The sync path fills every P*n row in group-by-parent order."""
    P, n = 3, 2
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=P, n=n)

    out = engine.generate(sample=batch)

    # Frontier gen Part filled for all P*n rows.
    gen = out.parts[-1]
    assert "text" in gen.primitives
    assert len(gen.primitives["text"].texts) == P * n

    # Explicit group-by-parent expected order: prompt-major, sibling-contiguous.
    prompts = list(batch.parts[0].primitives["text"].texts)
    expected = [raw_text_for(p, k) for p in prompts for k in range(n)]
    assert gen.primitives["text"].texts == expected

    engine.shutdown()


def test_backend_sees_per_prompt_wire_in_batch_order():
    """The backend receives one payload per prompt, in the whole batch's
    per-prompt wire order."""
    P, n = 3, 2
    engine = FakeEngine(concurrency=8)
    batch = build_request_batch(P=P, n=n)

    engine.generate(batch)

    prompts = list(batch.parts[0].primitives["text"].texts)
    assert [c["text"] for c in engine._backend.calls] == prompts
    assert len(engine._backend.calls) == P  # one payload per group/prompt
    assert all(c["sampling_params"]["n"] == n for c in engine._backend.calls)

    engine.shutdown()


def test_shared_semaphore_bounds_concurrency_across_groups():
    """All groups of one generate share a single semaphore, so peak in-flight is
    bounded by the configured concurrency C — not P (or P×C) for P groups."""
    P, n, C = 4, 2, 2
    engine = FakeEngine(concurrency=C)
    batch = build_request_batch(P=P, n=n)

    engine.generate(batch)

    peak = engine._backend.peak
    assert peak <= C  # the shared bound holds across all P groups
    assert peak > 1  # but generation genuinely overlapped (not serialized)

    engine.shutdown()


def test_shared_semaphore_bounds_concurrent_caller_threads():
    """The agentic-drain shape: N threads each call ``generate`` for their own
    single-prompt group. The backend's ONE semaphore bounds the union — and the
    admitted callers are in flight TOGETHER (deterministic via the hold-gate)."""
    C, N = 2, 5
    engine = FakeEngine(concurrency=C)
    engine._backend.block_until_released = True
    groups = build_request_batch(P=N, n=1).split()

    results: list = [None] * N
    threads = [
        threading.Thread(target=lambda i=i: results.__setitem__(i, engine.generate(groups[i]))) for i in range(N)
    ]
    for t in threads:
        t.start()
    deadline = threading.Event()
    for _ in range(1000):
        if engine._backend.inflight == C:
            break
        deadline.wait(0.005)
    assert engine._backend.inflight == C  # exactly the bound, all parked together

    engine._backend.release.set()  # stays set: the queued callers flow through
    for t in threads:
        t.join(timeout=5)

    assert engine._backend.peak == C
    assert all(r is not None and bool(r.parts[-1].primitives) for r in results)
    engine.shutdown()


def test_split_concat_round_trip_identity():
    """Sample.concat(sample.split()) reconstructs the batch exactly — the
    invariant the DP_SCATTER façade and per-group fan-out both rely on."""
    batch = build_request_batch(P=3, n=2)
    assert Sample.concat(batch.split()) == batch
