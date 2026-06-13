"""Fixtures + tier auto-skip for the transport test suite.

Tier markers (cpu/gpu/multigpu/mooncake/slow) are auto-skipped here based on the
host's GPU count and env, so a plain ``pytest tests/`` runs the maximal subset
the host can support and skips the rest cleanly.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


# ── availability probes ──────────────────────────────────────────────────────


def _gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _tq_lib_available() -> bool:
    try:
        import transfer_queue  # noqa: F401

        return True
    except Exception:
        return False


def _mooncake_available() -> bool:
    return bool(os.environ.get("UNIRL_MOONCAKE_MASTER")) and _tq_lib_available()


def _driver_backends() -> list:
    """Backends a DRIVER-side round-trip can resolve on this host.

    gpu_store is intentionally absent — its CUDA-IPC put/borrow path is
    worker-bound, so it is exercised through a real pool in
    ``test_gpu_store_roundtrip`` instead.
    """
    backends = ["colocate"]
    if _tq_lib_available():
        backends.append("tq_simple")
        if _mooncake_available():
            backends.append("tq_mooncake")
    return backends


# ── session Ray ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def ray_session():
    import ray

    # No num_gpus: a fresh local cluster auto-detects the GPUs; passing num_gpus
    # is rejected when ray.init() instead connects to a pre-existing cluster
    # (e.g. a leftover `ray start`). Run `ray stop` first for a clean session.
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    yield ray
    # Leave teardown to interpreter exit: GPUTensorHandle._release guards on
    # ray.is_initialized(), so finalizers firing during shutdown are safe.


# ── TransferQueue (simple, in-Ray) session runtime ───────────────────────────


@pytest.fixture(scope="session")
def _tq_simple_runtime(ray_session):
    from omegaconf import OmegaConf

    from unirl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime

    cfg = OmegaConf.create(
        {
            "transfer_queue": {
                "_target_": "unirl.distributed.tensor.backend.transfer_queue.simple.SimpleBackend",
                "num_units": 2,
                "unit_size": 256,
            }
        }
    )
    rt = TransferQueueRuntime().install()
    controller_handoff, _actor_handoff = rt.init(cfg)
    rt.create_client("driver", controller_handoff)
    yield rt
    try:
        rt.clear_partition()
    finally:
        TransferQueueRuntime.clear_current()


# ── parametrized driver-side transport (colocate / tq_simple / tq_mooncake) ──


def _install(t):
    from unirl.distributed.tensor.transport import TensorTransportRuntime

    TensorTransportRuntime.install(t)
    return t


def _build_driver_transport(kind, request):
    from unirl.distributed.tensor.backend.transfer_queue.runtime import _DEFAULT_PARTITION_ID
    from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport
    from unirl.distributed.tensor.factory import build_transport

    if kind == "colocate":
        return build_transport(
            "colocate_store", worker_id="dw0", device="cpu", device_id=0, global_rank=0, world_size=1
        )
    if kind == "tq_simple":
        rt = request.getfixturevalue("_tq_simple_runtime")
        return TQTransport(rt, partition_id=_DEFAULT_PARTITION_ID)
    if kind == "tq_mooncake":
        pytest.skip("mooncake driver round-trip requires an on-pod master fixture (see test_tq_mooncake)")
    raise ValueError(f"unknown driver backend {kind!r}")


@pytest.fixture(params=_driver_backends())
def transport(request, ray_session):
    """Active driver-resolvable TensorTransport, parametrized over every backend
    available on this host. Test IDs read ``[colocate]`` / ``[tq_simple]``."""
    from unirl.distributed.tensor.transport import TensorTransportRuntime

    yield _install(_build_driver_transport(request.param, request))
    TensorTransportRuntime.clear_current()


@pytest.fixture
def colocate_transport(ray_session):
    from unirl.distributed.tensor.factory import build_transport
    from unirl.distributed.tensor.transport import TensorTransportRuntime

    yield _install(
        build_transport("colocate_store", worker_id="dw0", device="cpu", device_id=0, global_rank=0, world_size=1)
    )
    TensorTransportRuntime.clear_current()


@pytest.fixture
def tq_simple_transport(_tq_simple_runtime):
    from unirl.distributed.tensor.backend.transfer_queue.runtime import _DEFAULT_PARTITION_ID
    from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport
    from unirl.distributed.tensor.transport import TensorTransportRuntime

    yield _install(TQTransport(_tq_simple_runtime, partition_id=_DEFAULT_PARTITION_ID))
    TensorTransportRuntime.clear_current()


# ── GPU device pools ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def _gpu1_pool(ray_session):
    if _gpu_count() < 1:
        pytest.skip("needs >=1 GPU")
    from unirl.distributed.group.device_pool import DevicePool

    pool = DevicePool(num_devices=1, devices_per_node=1, workers_per_device=2, transport_kind="gpu_store")
    pool.setup()
    yield pool
    pool.shutdown()


@pytest.fixture(params=["gpu_store", "colocate"])
def multigpu_pool(request, ray_session):
    if _gpu_count() < 2:
        pytest.skip("needs >=2 GPU")
    from unirl.distributed.group.device_pool import DevicePool

    kind = request.param
    wpd = 2 if kind == "gpu_store" else 1  # colocate enforces workers_per_device == 1
    pool = DevicePool(num_devices=2, devices_per_node=2, workers_per_device=wpd, transport_kind=kind)
    pool.setup()
    yield pool
    pool.shutdown()


@pytest.fixture
def gpu1_probe(_gpu1_pool):
    from tests.transport import harness

    role = harness.register_probe(_gpu1_pool)
    return SimpleNamespace(pool=_gpu1_pool, role=role, h=harness, backend="gpu_store")


@pytest.fixture
def multigpu_probe(multigpu_pool):
    from tests.transport import harness

    role = harness.register_probe(multigpu_pool)
    return SimpleNamespace(pool=multigpu_pool, role=role, h=harness, backend=multigpu_pool.transport_kind)


# ── tier auto-skip (keyed on GPU count + env) ────────────────────────────────


def pytest_collection_modifyitems(config, items):
    n = _gpu_count()
    moon = _mooncake_available()
    markexpr = config.getoption("markexpr", default="") or ""
    run_slow = bool(os.environ.get("RUN_SLOW")) or ("slow" in markexpr)

    skip_gpu = pytest.mark.skip(reason="needs >=1 GPU")
    skip_multigpu = pytest.mark.skip(reason="needs >=2 GPU")
    skip_moon = pytest.mark.skip(reason="needs Mooncake master (UNIRL_MOONCAKE_MASTER)")
    skip_slow = pytest.mark.skip(reason="slow: enable with RUN_SLOW=1 or -m slow")

    for item in items:
        if "multigpu" in item.keywords and n < 2:
            item.add_marker(skip_multigpu)
        elif "gpu" in item.keywords and n < 1:
            item.add_marker(skip_gpu)
        if "mooncake" in item.keywords and not moon:
            item.add_marker(skip_moon)
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
