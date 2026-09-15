from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from unirl.train.backend.fsdp.wrap import configure_copy_engine_all_gather
from unirl.train.configs import FSDPConfig
from unirl.utils.distributed_utils import ensure_dist_initialized


def test_copy_engine_all_gather_is_opt_in() -> None:
    assert FSDPConfig().copy_engine_all_gather is False


def test_copy_engine_all_gather_sets_zero_cta_before_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_CTA_POLICY", raising=False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    fake_pg = SimpleNamespace(
        NCCL_CTA_POLICY_ZERO=2,
        Options=lambda: SimpleNamespace(config=SimpleNamespace(cta_policy=None)),
    )
    monkeypatch.setattr("torch.distributed.ProcessGroupNCCL", fake_pg)

    options = configure_copy_engine_all_gather(True)

    assert os.environ["NCCL_CTA_POLICY"] == "2"
    assert options.config.cta_policy == 2


def test_copy_engine_all_gather_rejects_conflicting_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCCL_CTA_POLICY", "1")

    with pytest.raises(ValueError, match="requires NCCL_CTA_POLICY=2"):
        configure_copy_engine_all_gather(True)


def test_copy_engine_all_gather_rejects_late_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NCCL_CTA_POLICY", raising=False)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    with pytest.raises(ValueError, match="after the default process group"):
        configure_copy_engine_all_gather(True)


def test_ensure_dist_initialized_forwards_process_group_options(monkeypatch: pytest.MonkeyPatch) -> None:
    options = object()
    calls = []
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.distributed.init_process_group", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 0)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 1)

    ensure_dist_initialized(pg_options=options)

    assert calls == [{"pg_options": options}]
