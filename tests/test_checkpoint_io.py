from __future__ import annotations

from concurrent.futures import Future

import pytest
import torch

from unirl.tools._checkpoint import load_training_checkpoint
from unirl.train.backend import base_backend


def test_pending_dcp_future_is_process_global_and_drained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base_backend, "_PENDING_DCP_SAVE_FUTURE", None)
    future: Future[None] = Future()
    future.set_result(None)

    base_backend._set_pending_dcp_save(future)
    with pytest.raises(RuntimeError, match="already pending"):
        base_backend._set_pending_dcp_save(future)

    base_backend._drain_pending_dcp_save()
    assert base_backend._PENDING_DCP_SAVE_FUTURE is None


def test_pending_dcp_failure_still_clears_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(base_backend, "_PENDING_DCP_SAVE_FUTURE", None)
    future: Future[None] = Future()
    future.set_exception(ValueError("flush failed"))
    base_backend._set_pending_dcp_save(future)

    with pytest.raises(ValueError, match="flush failed"):
        base_backend._drain_pending_dcp_save()
    assert base_backend._PENDING_DCP_SAVE_FUTURE is None


def test_prepare_dcp_directory_invalidates_old_completion_marker(tmp_path) -> None:
    (tmp_path / ".metadata").write_bytes(b"old")
    torch.save({"old": True}, tmp_path / "checkpoint.pt")

    base_backend._prepare_dcp_directory(str(tmp_path), {"step": 7}, process_group=None)

    assert not (tmp_path / ".metadata").exists()
    assert not (tmp_path / "checkpoint.pt").exists()
    assert torch.load(tmp_path / "metadata.pt", weights_only=True) == {"step": 7}


def test_checkpoint_reader_loads_legacy_torch_directory(tmp_path) -> None:
    torch.save({"policy_state_dict": {"weight": torch.ones(2)}, "step": 3}, tmp_path / "checkpoint.pt")

    checkpoint = load_training_checkpoint(str(tmp_path))

    assert checkpoint["step"] == 3
    assert checkpoint["_checkpoint_format"] == "torch"
    assert torch.equal(checkpoint["policy_state_dict"]["weight"], torch.ones(2))


def test_checkpoint_reader_rejects_incomplete_dcp_directory(tmp_path) -> None:
    torch.save({"step": 3}, tmp_path / "metadata.pt")
    (tmp_path / "rank0.distcp").write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="incomplete DCP checkpoint"):
        load_training_checkpoint(str(tmp_path))
