"""Bare-engine sequencing — the logic the thin core actually owns.

A real ``VLLMOmniRolloutEngine`` instance constructed without ``__init__``
(seam + components wired by hand) exercises the stateful orchestration:
sleep ⇒ task + flag + ``mark_weights_released``; wake ⇒ task + active LoRA
re-push + flag (stays offloaded on re-push failure); the offloaded generate
guard; and one forward-wiring smoke.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import pytest

from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine
from unirl.rollout.engine.vllm_omni.weight_sync import WeightSync


class FakeBackend:
    def __init__(self) -> None:
        self.events: List[str] = []
        self.generate_args = None

    def sleep_task(self) -> None:
        self.events.append("sleep_task")

    def wake_task(self) -> None:
        self.events.append("wake_task")

    def generate(self, calls, *, attach_lora=False, ar_lora_passthrough=False):
        self.events.append("generate")
        self.generate_args = (list(calls), attach_lora, ar_lora_passthrough)
        return [["raw"]]

    def ping(self) -> bool:
        return True

    def shutdown(self) -> None:
        self.events.append("shutdown")

    def tp_per_stage(self):
        return {0: 1}

    # LoRA verbs for the re-push path
    def set_lora_handle(self, **kw: Any) -> None:
        self.events.append("set_lora_handle")

    def set_lora_copy(self, **kw: Any) -> None:
        self.events.append("set_lora_copy")


class FakeAdapter:
    needs_sigmas = False
    ar_lora_passthrough = True
    lora_copy_transport = False

    def __init__(self) -> None:
        self.validated = []
        self.built = []

    def validate_request(self, req) -> None:
        self.validated.append(req)

    def build_inputs(self, req):
        self.built.append(req)
        return ["call"]

    def build_response(self, req, per_request):
        return ("resp", per_request)


def bare_engine(*, lora_loaded: bool = False):
    engine = object.__new__(VLLMOmniRolloutEngine)
    engine.cfg = SimpleNamespace(modality="hi3_t2i")
    engine._is_offloaded = False
    engine._backend = FakeBackend()
    engine._weight_sync = WeightSync(engine._backend, uses_lora=True, lora_copy_transport=False)
    if lora_loaded:
        engine._weight_sync.set_lora_from_tensors("ad", {"k": 1})
        engine._backend.events.clear()
    engine.adapter = FakeAdapter()
    engine.schedule_policy = None
    return engine


def test_sleep_releases_flags_and_fires_event():
    engine = bare_engine()
    engine.sleep()
    assert engine._backend.events == ["sleep_task"]
    assert engine.is_offloaded
    assert engine._weight_sync.lora_dirty  # mark_weights_released fired
    # Idempotent: a second sleep is a no-op.
    engine.sleep()
    assert engine._backend.events == ["sleep_task"]


def test_wake_restores_lora_and_clears_flag():
    engine = bare_engine(lora_loaded=True)
    engine.sleep()
    engine.wake_up()
    assert engine._backend.events == ["sleep_task", "wake_task", "set_lora_handle"]
    assert not engine.is_offloaded and not engine._weight_sync.lora_dirty
    # Wake when not offloaded is a no-op.
    engine.wake_up()
    assert engine._backend.events == ["sleep_task", "wake_task", "set_lora_handle"]


def test_wake_failure_stays_offloaded_and_generate_guards():
    engine = bare_engine(lora_loaded=True)
    engine.sleep()

    def boom(**kw):
        raise RuntimeError("push failed")

    engine._backend.set_lora_handle = boom
    with pytest.raises(RuntimeError, match="LoRA-WAKE"):
        engine.wake_up()
    assert engine.is_offloaded  # defense-in-depth: caller may swallow the raise
    with pytest.raises(Exception, match="offloaded"):
        engine.generate("req")


def test_generate_forward_wiring_smoke():
    engine = bare_engine(lora_loaded=True)
    resp = engine.generate("req")
    assert resp == ("resp", [["raw"]])
    assert engine.adapter.validated == ["req"] and engine.adapter.built == ["req"]
    calls, attach_lora, ar_passthrough = engine._backend.generate_args
    assert calls == ["call"]
    assert attach_lora is True  # from weight_sync.lora_loaded
    assert ar_passthrough is True  # from adapter.ar_lora_passthrough


def test_lifecycle_forwards():
    engine = bare_engine()
    assert engine.health_check() is True
    assert engine.tp_per_stage() == {0: 1}
    engine.shutdown()
    assert engine._backend.events[-1] == "shutdown"
