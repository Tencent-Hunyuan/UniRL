"""Strict finalization for BAGEL's native Stage-0 KV replay trace."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def finalize_bagel_kv_replay_trace(
    *,
    chunk_records: Sequence[tuple[int, int]],
    captured_input_ids: Sequence[int],
    kv_length: int,
    request_id: str,
) -> dict[str, Any]:
    """Align captured scheduler inputs with the authoritative transferred KV."""
    records = [(int(start), int(length)) for start, length in chunk_records]
    token_ids = [int(token) for token in captured_input_ids]
    kv_length = int(kv_length)
    if not records:
        raise RuntimeError(f"BAGEL KV replay trace for request {request_id!r} is empty.")
    if kv_length <= 0:
        raise RuntimeError(
            f"BAGEL KV replay trace for request {request_id!r} has invalid transferred length {kv_length}."
        )

    expected_start = 0
    for start, length in records:
        if length <= 0 or start != expected_start:
            raise RuntimeError(
                "BAGEL KV replay trace is non-contiguous; prefix caching must be disabled: "
                f"request={request_id!r}, chunk_start={start}, expected_start={expected_start}, "
                f"chunk_length={length}."
            )
        expected_start += length
    if expected_start != len(token_ids):
        raise RuntimeError(
            "BAGEL KV replay trace token count does not match its chunk geometry: "
            f"request={request_id!r}, tokens={len(token_ids)}, chunks={expected_start}."
        )

    excluded_tail_input_ids: list[int] = []
    retained_records = records
    if len(token_ids) == kv_length + 1:
        final_start, final_length = records[-1]
        if final_start == kv_length and final_length == 1:
            excluded_tail_input_ids = token_ids[-1:]
            token_ids = token_ids[:-1]
            retained_records = records[:-1]

    if len(token_ids) != kv_length:
        raise RuntimeError(
            "BAGEL KV replay trace length does not match the transferred cache length; "
            "prefix caching must be disabled: "
            f"request={request_id!r}, captured={expected_start}, transferred={kv_length}."
        )
    if not retained_records:
        raise RuntimeError(f"BAGEL KV replay trace for request {request_id!r} has no transferred chunks.")

    chunk_offsets = [0]
    for _, length in retained_records:
        chunk_offsets.append(chunk_offsets[-1] + length)
    if chunk_offsets[-1] != kv_length:
        raise RuntimeError(
            "BAGEL KV replay trace retained chunk geometry does not match the transferred cache length: "
            f"request={request_id!r}, chunks={chunk_offsets[-1]}, transferred={kv_length}."
        )

    return {
        "cache_input_ids": token_ids,
        "chunk_offsets": chunk_offsets,
        "excluded_tail_input_ids": excluded_tail_input_ids,
        "kv_length": kv_length,
        "ropes": [kv_length],
    }


__all__ = ["finalize_bagel_kv_replay_trace"]
