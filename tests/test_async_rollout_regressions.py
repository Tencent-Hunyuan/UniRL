import threading

import pytest
from omegaconf import OmegaConf

from unirl.distributed.group.handle import PendingHandleCall
from unirl.rollout.manager.dispatch import RolloutPool
from unirl.trainer import agentic as agentic_module
from unirl.trainer.agentic import AgenticTrainer
from unirl.trainer.base import resolve_worker_concurrency_by_device, resolve_worker_max_concurrency
from unirl.trainer.diffusion import _validate_diffusion_dp_geometry


def test_constructor_inflight_sizes_separate_rollout_workers() -> None:
    cfg = OmegaConf.create({"num_devices": 8})

    rollout_concurrency = resolve_worker_max_concurrency(cfg, per_worker_inflight=1)

    assert rollout_concurrency == 3
    assert resolve_worker_concurrency_by_device(
        cfg,
        rollout_concurrency=rollout_concurrency,
        train_fraction=0.5,
    ) == [1, 1, 1, 1, 3, 3, 3, 3]


def test_environment_concurrency_is_coerced_at_config_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UNIRL_TEST_WORKER_CONCURRENCY", "3")
    cfg = OmegaConf.create(
        {
            "num_devices": 8,
            "worker_max_concurrency": "${oc.env:UNIRL_TEST_WORKER_CONCURRENCY}",
        }
    )

    resolved = resolve_worker_max_concurrency(cfg)

    assert resolved == 3
    assert isinstance(resolved, int)


def test_agentic_constructor_coerces_inflight_at_its_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resolved(Exception):
        pass

    captured = None

    def capture(_, *, per_worker_inflight):
        nonlocal captured
        captured = per_worker_inflight
        raise Resolved

    monkeypatch.setattr(agentic_module, "resolve_worker_max_concurrency", capture)
    with pytest.raises(Resolved):
        AgenticTrainer(
            cfg=OmegaConf.create({"num_devices": 1}),
            batch_size=1,
            bundle_cfg=None,
            pipeline_cfg=None,
            backend_cfg=None,
            rollout_cfg=None,
            reward_cfg=None,
            algorithm_cfg=None,
            stack_cfg=None,
            data_source_cfg=None,
            sampling_cfg=None,
            sync_cfg=None,
            per_worker_inflight="4",
        )

    assert captured == 4
    assert isinstance(captured, int)


def test_prompt_local_diffusion_rollout_skips_only_rollout_dp_divisibility() -> None:
    _validate_diffusion_dp_geometry(
        batch_size=5,
        samples_per_prompt=3,
        num_updates_per_batch=1,
        rollout_dp_size=3,
        reward_dp_size=5,
        train_dp_size=5,
        require_rollout_dp_divisibility=False,
    )

    with pytest.raises(ValueError, match="rollout dp_size=3"):
        _validate_diffusion_dp_geometry(
            batch_size=5,
            samples_per_prompt=3,
            num_updates_per_batch=1,
            rollout_dp_size=3,
            reward_dp_size=5,
            train_dp_size=5,
        )

    with pytest.raises(ValueError, match="reward dp_size=3"):
        _validate_diffusion_dp_geometry(
            batch_size=5,
            samples_per_prompt=3,
            num_updates_per_batch=1,
            rollout_dp_size=3,
            reward_dp_size=3,
            train_dp_size=5,
            require_rollout_dp_divisibility=False,
        )


def test_pending_wait_resolves_outputs_before_discarding() -> None:
    class ResolvingHandle:
        def __init__(self) -> None:
            self.calls = 0

        def _resolve_call(self, collect_fn, refs, *, worker_local, targets):
            self.calls += 1
            assert refs == ["ref"]
            assert worker_local
            assert targets == ["worker"]
            return collect_fn(self, ["payload"])

    handle = ResolvingHandle()
    lease = object()
    pending = PendingHandleCall(
        handle,
        "generate",
        ["ref"],
        True,
        targets=["worker"],
        collect_fn=lambda _, results: results[0],
        leases=[lease],
    )

    pending.wait()
    pending.wait()

    assert handle.calls == 1
    assert pending._consumed
    assert pending._leases is None
    assert pending._value is None


def test_close_does_not_block_on_partially_launched_rpc() -> None:
    launched = threading.Event()
    launch_failed = threading.Event()

    class DeferredFuture:
        def __init__(self) -> None:
            self.callback = None

        def add_done_callback(self, callback) -> None:
            self.callback = callback

        def finish(self) -> None:
            assert self.callback is not None
            self.callback(self)

    class DeferredRef:
        def __init__(self, future) -> None:
            self._future = future

        def future(self):
            return self._future

    class ResolvingHandle:
        def __init__(self) -> None:
            self.calls = 0

        def _resolve_call(self, collect_fn, refs, *, worker_local, targets):
            self.calls += 1
            return collect_fn(self, ["payload"])

    future = DeferredFuture()
    handle = ResolvingHandle()
    lease = object()
    pending = PendingHandleCall(
        handle,
        "generate",
        [DeferredRef(future)],
        True,
        targets=["worker"],
        collect_fn=lambda _, results: results[0],
        leases=[lease],
    )

    def launch_ok(_):
        launched.set()
        return pending

    def launch_error(_):
        launch_failed.set()
        raise RuntimeError("launch failed")

    pool = RolloutPool([launch_ok, launch_error], [1, 1])
    pool.add([object(), object()])
    assert launched.wait(timeout=1)
    assert launch_failed.wait(timeout=1)

    pool.close()

    assert handle.calls == 0
    assert pending._leases == [lease]
    assert not any(thread.name == "rollout-pool-drain" for thread in threading.enumerate())

    future.finish()

    assert handle.calls == 1
    assert pending._consumed
    assert pending._leases is None
    assert pending._value is None
