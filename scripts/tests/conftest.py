"""Shared fixtures, markers, and fakes for the rollout TP/EP/PP tests.

Two tiers:

- ``*_on_cpu`` tests are pure-Python (rank-layout math, config intent, no-op
  shell construction, stubbed weight-sync transports) and run anywhere.
- ``*_gpu`` tests need real GPUs + a live SGLang server; they are skipped unless
  ``UNIRL_TP_GPU_TEST`` is set and enough CUDA devices are visible.

The on-cpu weight-sync tests share a small family of transport fakes (Ray handle,
FSDP backend, SGLang serializer) collected here so each test module imports them
instead of redefining its own copy.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pytest

# --------------------------------------------------------------------------- #
# GPU gate
# --------------------------------------------------------------------------- #


def _cuda_device_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA GPUs + a live SGLang server")


def requires_gpus(n: int):
    """Skip decorator: needs ``UNIRL_TP_GPU_TEST=1`` and >= ``n`` CUDA devices."""
    gate = os.environ.get("UNIRL_TP_GPU_TEST", "") not in ("", "0", "false", "False")
    have = _cuda_device_count()
    reason = f"needs UNIRL_TP_GPU_TEST=1 and >= {n} GPUs (gate={gate}, visible={have})"
    return pytest.mark.skipif(not (gate and have >= n), reason=reason)


# --------------------------------------------------------------------------- #
# SGLang E2E clean-exit
# --------------------------------------------------------------------------- #


def sglang_e2e_teardown(engine, *, passed: bool) -> None:
    """Shut down an in-process SGLang engine and force-exit on success.

    SGLang boots its scheduler as sibling subprocesses of the pytest runner
    (these tests construct the engine in-process, NOT inside a Ray actor). On
    ``shutdown`` SGLang fires SIGQUIT at those siblings, which can reach the
    pytest process and kill it before it prints the result line. ``os._exit``
    skips Python finalizers and exits cleanly right after a passing test — safe
    because the only unreleased resource is the SGLang server we just shut down.

    Long-term fix: run the engine inside a Ray actor (as verl-omni's rollout
    tests do) so the subprocess signals stay isolated from the runner; then this
    helper and the per-test ``os._exit`` become unnecessary. Because it exits the
    process, only ONE ``*_gpu`` E2E test can run per pytest invocation.
    """
    try:
        engine.shutdown()
    except Exception:
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    if passed:
        os._exit(0)


# --------------------------------------------------------------------------- #
# Transport fakes for the on-cpu weight-sync tests
# --------------------------------------------------------------------------- #


@dataclass
class FakeBackend:
    """FSDP backend stand-in: ``rollout_adapter_name`` is all the base reads."""

    rollout_adapter_name: str = "default"


class FakeRef:
    """Picklable stand-in for a Ray ObjectRef (the fake ``ray.get`` returns None)."""


class FakeRayHandle:
    """Stand-in for a Ray Worker actor handle recording dispatched RPCs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[Tuple[str, Tuple, Dict]] = []
        self.call = self._Call(self)

    class _Call:
        def __init__(self, owner):
            self._owner = owner

        def remote(self, role_name, method_name, args, kwargs, **_):
            self._owner.calls.append((method_name, args, dict(kwargs)))
            return FakeRef()


def make_nccl_sync(monkeypatch):
    """Build an ``NCCLWeightSync`` with its ray / process-group plumbing stubbed.

    Bypasses the real ``init_process_group`` and ``ray.get`` and seeds the
    ``_rollout_targets`` / ``_rollout_role`` state directly, so ``connect`` runs
    without a driver dispatch decorator.
    """
    from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync

    monkeypatch.setattr("unirl.utils.distributed_utils.init_process_group", lambda **kw: ("pg",))
    import ray

    monkeypatch.setattr(ray, "get", lambda refs: None)
    sync = NCCLWeightSync.__new__(NCCLWeightSync)
    sync._group_name = "g"
    sync._model_update_group = None
    sync._rollout_targets = []
    sync._rollout_role = None
    sync._track_prefix = ""
    return sync
