"""Engine-core tests that aren't tautological: the chunking arithmetic, the
inherited IPC NotImplementedError, and the real class's dispatch markers.

The generate→backend→adapter happy path is pure delegation; per the testing
convention it's covered by the GPU smoke + parity gate, not a mocked CPU test.
"""

from __future__ import annotations

import types

import pytest
import torch

from unirl.distributed.group.dispatch import DISTRIBUTED_CONFIG_ATTR
from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams


def _bare_engine(**attrs):
    """An engine instance without running __init__ (which would spawn SGLang)."""
    e = object.__new__(SGLangDiffusionRolloutEngine)
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


def _req(n, *, num_inference_steps=2):
    return RolloutReq(
        sample_ids=[f"s{i}" for i in range(n)],
        group_ids=[f"g{i}" for i in range(n)],
        primitives={"text": Texts(texts=[f"p{i}" for i in range(n)])},
        sampling_params=DiffusionSamplingParams(num_inference_steps=num_inference_steps),
        sigmas=torch.linspace(1.0, 0.0, num_inference_steps + 1),  # pre-set → ensure_req_sigmas no-ops
    )


# --------------------------------------------------------------------------- #
# IPC + dispatch markers on the REAL class (FakeEngine in the other file has a
# different MRO, so re-assert here).
# --------------------------------------------------------------------------- #


def test_update_weights_from_ipc_not_implemented():
    e = _bare_engine()
    with pytest.raises(NotImplementedError):
        e.update_weights_from_ipc(peft_config=None, base_sync_done=False)


def test_real_engine_dispatch_markers():
    E = SGLangDiffusionRolloutEngine
    assert hasattr(E.generate, DISTRIBUTED_CONFIG_ATTR)
    assert hasattr(E.sleep, DISTRIBUTED_CONFIG_ATTR)
    assert hasattr(E.wake_up, DISTRIBUTED_CONFIG_ATTR)
    for name in (
        "update_weights_from_tensor",
        "init_weights_update_group",
        "update_weights_from_distributed",
        "destroy_weights_update_group",
        "set_lora_from_tensors",
        "loaded_param_checksums",
    ):
        assert not hasattr(getattr(E, name), DISTRIBUTED_CONFIG_ATTR), f"{name} must not be dispatched"


# --------------------------------------------------------------------------- #
# Forward chunking arithmetic (real logic: slice boundaries + RolloutResp.concat)
# --------------------------------------------------------------------------- #


def _stub_batch(calls):
    def _gen(self, req):
        calls.append(int(req.batch_size))
        return RolloutResp(
            tracks={"image": RolloutTrack(sample_ids=list(req.sample_ids), parent_ids=list(req.group_ids))}
        )

    return _gen


def test_generate_single_call_when_no_chunking(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls: list[int] = []
    e = _bare_engine(cfg=types.SimpleNamespace(forward_batch_size=None), schedule_policy=None)
    e._generate_batch = types.MethodType(_stub_batch(calls), e)
    out = e.generate(_req(5))
    assert calls == [5]  # one forward over the whole batch
    assert len(out.tracks["image"].sample_ids) == 5


def test_generate_chunks_and_reassembles(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls: list[int] = []
    e = _bare_engine(cfg=types.SimpleNamespace(forward_batch_size=2), schedule_policy=None)
    e._generate_batch = types.MethodType(_stub_batch(calls), e)
    out = e.generate(_req(5))
    assert calls == [2, 2, 1]  # sliced on the fbs boundary
    assert len(out.tracks["image"].sample_ids) == 5  # reassembled to the full batch
    assert out.tracks["image"].sample_ids == [f"s{i}" for i in range(5)]


def test_generate_rejects_empty_batch():
    e = _bare_engine(cfg=types.SimpleNamespace(forward_batch_size=None), schedule_policy=None)
    with pytest.raises(Exception, match="non-empty req"):
        e.generate(_req(0))
