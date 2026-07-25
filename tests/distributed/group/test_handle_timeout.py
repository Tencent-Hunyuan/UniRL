from itertools import count
from types import SimpleNamespace

from unirl.distributed.group import handle as handle_module
from unirl.distributed.group.dispatch import Dispatch
from unirl.distributed.group.handle import Handle


class _Transport:
    @classmethod
    def localize(cls, shards, pool, device_ids, worker_ids):
        del pool, device_ids, worker_ids
        return shards


def test_handle_applies_ray_get_timeout_without_forwarding_it(monkeypatch) -> None:
    observed = {}

    def dispatch_fn(handle, args, kwargs, batch_size):
        del handle, batch_size
        observed["worker_kwargs"] = kwargs
        return [(args, kwargs)]

    def execute_fn(method_name, shards, *, grad_mode, call_id):
        del method_name, shards, grad_mode, call_id
        return ["object-ref"]

    def fake_get(refs, *, timeout):
        observed["refs"] = refs
        observed["timeout"] = timeout
        return ["result"]

    fake_handle = SimpleNamespace(
        pool=SimpleNamespace(transport_cls=_Transport),
        device_ids=[0],
        worker_ids=["worker-0"],
        workers=["worker-0"],
        rank_infos=[],
        dp_size=1,
        _grad_call_counter=count(),
        _rebind_tree=lambda value, worker, worker_local: value,
    )
    monkeypatch.setattr(handle_module, "infer_batch_size", lambda args, kwargs: None)
    monkeypatch.setattr(handle_module.ray, "get", fake_get)

    fn = Handle._make_handle_fn(
        fake_handle,
        "wait_for_checkpoint",
        Dispatch.BROADCAST,
        dispatch_fn,
        lambda handle, results: results[0],
        execute_fn,
    )
    result = fn(worker_option="kept", _ray_get_timeout=3.5)

    assert result == "result"
    assert observed == {
        "worker_kwargs": {"worker_option": "kept"},
        "refs": ["object-ref"],
        "timeout": 3.5,
    }
