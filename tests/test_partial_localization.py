from types import SimpleNamespace

import pytest
import torch

import unirl.distributed.group.handle as handle_module
from unirl.distributed.group.dispatch import Dispatch, remap_required_store_keys, required_store_keys
from unirl.distributed.group.handle import Handle
from unirl.distributed.group.worker import Worker
from unirl.distributed.tensor import TensorRef, TensorSpan, TensorTransportRuntime
from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle
from unirl.distributed.tensor.worker_local import WorkerLocalTransport


class _DeviceLocalTransport:
    @classmethod
    def _is_local(cls, ref, dst_worker_id, dst_device_id, pool):
        return pool.device_id_of(ref.source_id) == dst_device_id


class _Pool:
    transport_cls = _DeviceLocalTransport

    @staticmethod
    def device_id_of(worker_id):
        return int(str(worker_id).split("dw", 1)[1].split("_", 1)[0])


def _ref(key: str, *, source_id: str = "dw0") -> TensorRef:
    handle = GPUTensorHandle(
        store_key=key,
        source_id=source_id,
        shape=(1,),
        dtype=torch.float32,
        device="cuda:0",
    )
    return TensorRef(spans=[TensorSpan(handle, 0, 1)], shape=(1,), dtype=torch.float32, device="cuda:0")


def _partial_config(*, reads=None, skips=None) -> dict:
    return {
        "dispatch_mode": Dispatch.BROADCAST,
        "execute_mode": None,
        "reads": reads,
        "skips": skips,
    }


def test_required_store_keys_is_alias_safe_for_skips() -> None:
    shared = _ref("shared")
    config = _partial_config(skips=lambda skipped, **_: skipped)

    required = required_store_keys(config, (shared,), {"also_read": shared})

    assert required == {"shared"}


def test_skips_view_degrades_to_full_localization() -> None:
    original = _ref("source")
    config = _partial_config(skips=lambda value: value[:])

    required = required_store_keys(config, (original,), {})

    assert required == {"source"}


def test_remap_preserves_requiredness_for_same_key_views() -> None:
    source_handle = GPUTensorHandle("source", "dw1", (2,), torch.float32, "cuda:0")
    first = TensorRef.from_handles([source_handle])
    second = TensorRef.from_handles([source_handle])
    required = required_store_keys(_partial_config(reads=lambda selected, _: selected), (first, second), {})
    moved_first = _ref("first", source_id="dw0")
    moved_second = _ref("second", source_id="dw0")

    remapped = remap_required_store_keys(required, (first, second), {}, (moved_first, moved_second), {})

    assert remapped == {"first", "second"}


def test_launch_call_sends_post_localization_store_keys() -> None:
    class MovingTransport(WorkerLocalTransport):
        @classmethod
        def localize(cls, shards, pool, device_ids, worker_ids, required=None):
            assert required == [{"source"}]
            moved = _ref("destination", source_id=worker_ids[0])
            return [((moved,), {})]

    captured = {}

    def execute(method_name, shards, *, grad_mode, call_id, required):
        captured["required"] = required
        return ["rpc-ref"]

    handle = object.__new__(Handle)
    handle.pool = SimpleNamespace(transport_cls=MovingTransport)
    handle.device_ids = [0]
    handle.worker_ids = ["dw0"]
    handle.rank_infos = []
    handle.world_size = 1
    source = _ref("source", source_id="dw1")

    refs, worker_local, passthrough = handle._launch_call(
        "method",
        Dispatch.BROADCAST,
        lambda *_: [((source,), {})],
        execute,
        (source,),
        {},
        grad_mode=False,
        call_id=None,
        config=_partial_config(reads=lambda value: value),
    )

    assert refs == ["rpc-ref"]
    assert worker_local is True
    assert set(passthrough) == {"destination"}
    assert passthrough["destination"].store_key == "destination"
    assert captured["required"] == [{"destination"}]


def test_worker_uses_controller_mask_without_recomputing_selector() -> None:
    class FakeTransport:
        def get_batch(self, metas):
            return {key: torch.ones(1) for key in metas}

        def put_batch(self, tensors):
            assert tensors == {}
            return {}

    class Role:
        def inspect(self, required_value, passthrough_value):
            return isinstance(required_value, torch.Tensor), isinstance(passthrough_value, TensorRef)

    worker = object.__new__(Worker)
    worker._init_local(transport=FakeTransport())
    worker._roles["role"] = Role()
    try:
        result = worker.call(
            "role",
            "inspect",
            (_ref("required"), _ref("passthrough")),
            {},
            required={"required"},
        )
    finally:
        TensorTransportRuntime.clear_current()

    assert result == (True, True)


def test_rebind_validates_all_results_before_mutating(monkeypatch) -> None:
    first = _ref("first", source_id="dw0")
    foreign = _ref("foreign", source_id="dw2")
    handle = object.__new__(Handle)
    handle.pool = _Pool()
    handle.workers = [object(), object()]
    handle.worker_ids = ["dw0_s1", "dw1"]
    monkeypatch.setattr(handle_module.ray, "get", lambda refs, timeout=None: refs)

    with pytest.raises(RuntimeError, match="returned a ref owned by 'dw2'"):
        handle._resolve_call(
            lambda _, results: results,
            [first, foreign],
            worker_local=True,
            passthrough={},
        )

    assert first.spans[0].handle._finalized is False


def test_rebind_accepts_same_device_gpu_store_handle(monkeypatch) -> None:
    result = _ref("same-device", source_id="dw0")
    worker_handle = object()
    handle = object.__new__(Handle)
    handle.pool = _Pool()
    handle.workers = [worker_handle]
    handle.worker_ids = ["dw0_s1"]
    monkeypatch.setattr(handle_module.ray, "get", lambda refs, timeout=None: refs)

    resolved = handle._resolve_call(
        lambda _, results: results[0],
        [result],
        worker_local=True,
        passthrough={},
    )

    assert resolved.spans[0].handle.worker_handle is worker_handle


def test_passthrough_restore_reuses_original_handle() -> None:
    original = _ref("shared")
    returned_copy = _ref("shared")
    handle = object.__new__(Handle)

    restored = handle._rebind_tree(
        returned_copy,
        worker_handle=object(),
        worker_local=True,
        passthrough={"shared": original.spans[0].handle},
    )

    assert restored.spans[0].handle is original.spans[0].handle
    assert returned_copy.spans[0].handle._finalized is False
