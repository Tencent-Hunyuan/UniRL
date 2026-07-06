"""Shared fixtures / markers for the rollout TP/EP/PP smoke tests.

Tier 0 tests are pure-Python (rank-layout math, config intent, no-op shell
construction) and run anywhere. Tier 1/2 tests need real GPUs + a live SGLang
server and are skipped unless ``UNIRL_TP_GPU_TEST`` is set and enough CUDA
devices are visible.
"""

from __future__ import annotations

import os

import pytest


def _cuda_device_count() -> int:
    try:
        import torch
    except Exception:
        return 0
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA GPUs + a live SGLang server (Tier 1/2)")


def requires_gpus(n: int):
    """Skip decorator: needs ``UNIRL_TP_GPU_TEST=1`` and >= ``n`` CUDA devices."""
    gate = os.environ.get("UNIRL_TP_GPU_TEST", "") not in ("", "0", "false", "False")
    have = _cuda_device_count()
    reason = f"needs UNIRL_TP_GPU_TEST=1 and >= {n} GPUs (gate={gate}, visible={have})"
    return pytest.mark.skipif(not (gate and have >= n), reason=reason)
