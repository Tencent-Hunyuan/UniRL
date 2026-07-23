"""Regressions for the main/agentic vLLM-Omni conflict resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import unirl.rollout.engine.vllm_omni.engine as engine_module
from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine


class _Config:
    modality = "fake"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def server_intent(self, *, model_config, ports, extra):
        del model_config, ports, extra
        self.events.append("intent")
        return {"fake": True}


class _Adapter:
    lora_copy_transport = False
    ar_lora_passthrough = False

    def __init__(self, *, events: list[str], needs_sigmas: bool) -> None:
        self.events = events
        self.needs_sigmas = needs_sigmas

    def schedule_policy(self):
        self.events.append("schedule")
        return "policy"

    def boot_kwargs(self):
        self.events.append("boot_kwargs")
        return {}


@pytest.mark.parametrize(
    ("needs_sigmas", "expected_policy", "expected_events"),
    [
        (True, "policy", ["schedule", "boot_kwargs", "intent", "backend"]),
        (False, None, ["boot_kwargs", "intent", "backend"]),
    ],
)
def test_schedule_is_gated_and_resolved_before_backend_boot(
    monkeypatch: pytest.MonkeyPatch,
    needs_sigmas: bool,
    expected_policy,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    adapter = _Adapter(events=events, needs_sigmas=needs_sigmas)
    monkeypatch.setattr(engine_module, "get_adapter", lambda _key: lambda *_args, **_kwargs: adapter)

    class Backend:
        @staticmethod
        def boot(_intent):
            events.append("backend")
            return object()

    monkeypatch.setattr(engine_module, "VLLMOmniBackend", Backend)
    monkeypatch.setattr(engine_module, "WeightSync", lambda *_args, **_kwargs: SimpleNamespace())

    model_config = SimpleNamespace(use_lora=False)
    engine = VLLMOmniRolloutEngine(
        _Config(events),
        model_config=model_config,
        ports=object(),
    )

    assert engine.schedule_policy == expected_policy
    assert events == expected_events


def test_diffusion_generation_pins_the_sample_before_building_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[object] = []
    sample = object()

    class Adapter:
        needs_sigmas = True
        ar_lora_passthrough = False

        def validate_request(self, value):
            events.append(("validate", value))

        def build_inputs(self, value):
            events.append(("build", value))
            return ["call"]

        def build_response(self, value, raw):
            events.append(("response", value, raw))
            return value

    class Backend:
        def generate(self, calls, **kwargs):
            events.append(("generate", calls, kwargs))
            return [["raw"]]

    engine = object.__new__(VLLMOmniRolloutEngine)
    engine._is_offloaded = False
    engine.adapter = Adapter()
    engine.schedule_policy = "policy"
    engine._backend = Backend()
    engine._weight_sync = SimpleNamespace(lora_loaded=False)
    monkeypatch.setattr(
        engine_module,
        "ensure_sample_sigmas",
        lambda value, policy: events.append(("sigmas", value, policy)),
    )

    assert engine._generate_core(sample) is sample
    assert events[:3] == [
        ("validate", sample),
        ("sigmas", sample, "policy"),
        ("build", sample),
    ]


def test_diffusion_generation_fails_when_schedule_is_missing() -> None:
    engine = object.__new__(VLLMOmniRolloutEngine)
    engine._is_offloaded = False
    engine.adapter = SimpleNamespace(
        needs_sigmas=True,
        validate_request=lambda _sample: None,
    )
    engine.schedule_policy = None

    with pytest.raises(ValueError, match="has no sigma schedule policy"):
        engine._generate_core(object())
