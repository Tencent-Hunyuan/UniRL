from __future__ import annotations

import pytest

from unirl.rollout.engine.vllm_omni.patches.bagel_kv_replay import finalize_bagel_kv_replay_trace


def test_finalizer_preserves_exact_committed_trace() -> None:
    trace = finalize_bagel_kv_replay_trace(
        chunk_records=[(0, 2), (2, 3)],
        captured_input_ids=[99, 10, 11, 99, 20],
        kv_length=5,
        request_id="exact",
    )

    assert trace == {
        "cache_input_ids": [99, 10, 11, 99, 20],
        "chunk_offsets": [0, 2, 5],
        "excluded_tail_input_ids": [],
        "kv_length": 5,
        "ropes": [5],
    }


def test_finalizer_excludes_only_terminal_async_singleton() -> None:
    trace = finalize_bagel_kv_replay_trace(
        chunk_records=[(0, 2), (2, 3), (5, 1)],
        captured_input_ids=[99, 10, 11, 99, 20, 21],
        kv_length=5,
        request_id="early-stop",
    )

    assert trace["cache_input_ids"] == [99, 10, 11, 99, 20]
    assert trace["chunk_offsets"] == [0, 2, 5]
    assert trace["excluded_tail_input_ids"] == [21]
    assert trace["ropes"] == [5]


@pytest.mark.parametrize(
    ("chunk_records", "captured_input_ids", "kv_length", "match"),
    [
        ([(0, 2)], [10, 11], 3, "captured=2, transferred=3"),
        ([(0, 3), (3, 1), (4, 1)], [10, 11, 12, 13, 14], 3, "captured=5, transferred=3"),
        ([(0, 3), (3, 2)], [10, 11, 12, 13, 14], 4, "captured=5, transferred=4"),
        ([(0, 2), (3, 1)], [10, 11, 12], 3, "non-contiguous"),
        ([(0, 2), (1, 1)], [10, 11, 12], 3, "non-contiguous"),
    ],
)
def test_finalizer_rejects_unproven_or_noncontiguous_boundaries(
    chunk_records: list[tuple[int, int]],
    captured_input_ids: list[int],
    kv_length: int,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        finalize_bagel_kv_replay_trace(
            chunk_records=chunk_records,
            captured_input_ids=captured_input_ids,
            kv_length=kv_length,
            request_id="bad",
        )
