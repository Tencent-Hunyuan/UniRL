from __future__ import annotations

import os

import pytest

from unirl.train.backend.fsdp.wrap import configure_copy_engine_all_gather
from unirl.train.configs import FSDPConfig


def test_copy_engine_all_gather_is_opt_in() -> None:
    assert FSDPConfig().copy_engine_all_gather is False


def test_copy_engine_all_gather_sets_zero_cta_before_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_CTA_POLICY", raising=False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)

    configure_copy_engine_all_gather(True)

    assert os.environ["NCCL_CTA_POLICY"] == "2"


def test_copy_engine_all_gather_rejects_conflicting_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCCL_CTA_POLICY", "1")

    with pytest.raises(ValueError, match="requires NCCL_CTA_POLICY=2"):
        configure_copy_engine_all_gather(True)


def test_copy_engine_all_gather_rejects_late_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_CTA_POLICY", raising=False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    with pytest.raises(ValueError, match="after the default process group"):
        configure_copy_engine_all_gather(True)
