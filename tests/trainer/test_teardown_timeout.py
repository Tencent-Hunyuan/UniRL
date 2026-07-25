from types import SimpleNamespace

import pytest

from unirl.trainer import base as base_module
from unirl.trainer.base import BaseTrainer
from unirl.trainer.pe import PETrainer


def test_exception_teardown_timeout_preserves_primary_exception(monkeypatch) -> None:
    trainer = object.__new__(BaseTrainer)
    observed = {}
    trainer.wandb_logger = SimpleNamespace(finish=lambda: observed.setdefault("finished", True))

    def timeout_wait(*, timeout):
        observed["timeout"] = timeout
        raise TimeoutError("worker is wedged")

    trainer._wait_for_checkpoints = timeout_wait
    trainer._cleanup_weight_sync = lambda *, timeout: observed.setdefault("cleanup_timeout", timeout)
    monkeypatch.setattr(base_module, "_TEARDOWN_FLUSH_TIMEOUT_S", 4.0)

    with pytest.raises(RuntimeError, match="primary failure"):
        try:
            raise RuntimeError("primary failure")
        finally:
            trainer._finish_wandb()

    assert observed == {"timeout": 4.0, "finished": True}


@pytest.mark.parametrize("timeout", [None, 7.0])
def test_pe_checkpoint_wait_accepts_and_forwards_optional_timeout(timeout) -> None:
    calls = []

    def wait_for_checkpoint(**kwargs):
        calls.append(kwargs)

    trainer = object.__new__(PETrainer)
    trainer._freeze_llm = True
    trainer.diffusion = SimpleNamespace(backend=SimpleNamespace(wait_for_checkpoint=wait_for_checkpoint))

    trainer._wait_for_checkpoints(timeout=timeout)

    expected = {} if timeout is None else {"_ray_get_timeout": timeout}
    assert calls == [expected]
