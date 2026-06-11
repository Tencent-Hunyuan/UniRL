"""Boot dispatch — ``config.backend`` picks the seam impl, intent flows whole.

A real ``SGLangV2RolloutEngine.__init__`` run with both boots faked (recorder
classmethod stand-ins on the engine module's names) and the tokenizer I/O
stubbed: the default routes to HTTP with its client-side knobs
(``advertise_host`` / ``health_timeout_s``), ``backend="native"`` routes to the
native boot, and in both cases the injected reserved ports arrive inside the
intent verbatim.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from conftest import RecordingBackend, StubTokenizer

import unirl.rollout.engine.sglang_v2.engine as engine_mod
from unirl.rollout.engine.sglang_v2.config import SGLangV2EngineConfig, SGLangV2Ports
from unirl.rollout.engine.sglang_v2.engine import SGLangV2RolloutEngine

PORTS = SGLangV2Ports(server_port=30001, nccl_port=30002)


class RecordingBoot:
    """Stands in for a backend class — records ``boot`` calls, returns a fake."""

    def __init__(self):
        self.calls = []
        self.backend = RecordingBackend()

    def boot(self, intent, **kwargs):
        self.calls.append((intent, kwargs))
        return self.backend


@pytest.fixture
def boots(monkeypatch):
    """Fake both backend boots + the tokenizer I/O; return the recorders."""
    http, native = RecordingBoot(), RecordingBoot()
    monkeypatch.setattr(engine_mod, "HTTPBackend", http)
    monkeypatch.setattr(engine_mod, "NativeBackend", native)
    # The ctor imports transformers lazily — inject a stub module so the test
    # never touches the network or a local HF cache.
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: StubTokenizer())),
    )
    return http, native


def test_default_backend_boots_http(boots):
    http, native = boots
    config = SGLangV2EngineConfig(pretrained_model_ckpt_path="stub/model")

    engine = SGLangV2RolloutEngine(config, device=torch.device("cpu"), ports=PORTS)

    assert len(http.calls) == 1 and native.calls == []
    intent, kwargs = http.calls[0]
    assert intent["port"] == 30001
    assert intent["nccl_port"] == 30002
    assert kwargs["concurrency"] == config.concurrency
    assert isinstance(kwargs["advertise_host"], str)  # HTTP-only client knob
    assert kwargs["health_timeout_s"] == 300.0
    assert engine._backend is http.backend


def test_native_backend_boots_in_process_engine(boots):
    http, native = boots
    config = SGLangV2EngineConfig(
        pretrained_model_ckpt_path="stub/model",
        backend="native",
        engine_kwargs={"concurrency": 3},
    )

    engine = SGLangV2RolloutEngine(config, device=torch.device("cpu"), ports=PORTS)

    assert len(native.calls) == 1 and http.calls == []
    intent, kwargs = native.calls[0]
    assert intent["port"] == 30001  # reserved ports flow through the intent
    assert intent["nccl_port"] == 30002
    # The native boot takes only concurrency — no HTTP client knobs.
    assert kwargs == {"concurrency": 3}
    assert engine._backend is native.backend
