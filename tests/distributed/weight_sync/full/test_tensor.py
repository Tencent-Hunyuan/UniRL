from types import SimpleNamespace

import pytest
import torch.distributed as dist

from unirl.distributed.weight_sync.full.tensor import TensorWeightSync


def test_rank_local_serialization_error_is_collected_before_payload_gather(monkeypatch) -> None:
    class FailingBucket:
        def __init__(self, *, named_tensors) -> None:
            del named_tensors
            raise RuntimeError("CUDA IPC export failed")

    payload, error = TensorWeightSync._serialize_payload_or_error(
        [("weight", object())],
        FailingBucket,
        object,
    )
    assert payload is None
    assert error == "RuntimeError: CUDA IPC export failed"

    rank_info = SimpleNamespace(
        rank=1,
        dp_rank=0,
        pp_rank=0,
        tp_rank=1,
    )

    def fake_all_gather_object(gathered, local) -> None:
        gathered[:] = [
            {
                "rank": 0,
                "dp_rank": 0,
                "pp_rank": 0,
                "tp_rank": 0,
                "payload": "payload-rank-0",
                "error": None,
            },
            local,
        ]

    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(
        RuntimeError,
        match=r"SGLang TP payload serialization failed on rank 1: RuntimeError: CUDA IPC export failed",
    ):
        TensorWeightSync._gather_sglang_tp_payloads(
            payload,
            local_error=error,
            rank_info=rank_info,
            tp_size=2,
        )
