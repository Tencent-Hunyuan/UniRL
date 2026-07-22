"""Focused tests for atomic natural-drain finalization."""

from __future__ import annotations

from typing import Any

import pytest

from unirl.distributed.group.dispatch import Dispatch, Execute
from unirl.rollout.engine.agentic import engine as engine_module
from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine
from unirl.types.sample import Part, Sample


def _engine_with_refs(refs: list[Any]) -> AgenticRolloutEngine:
    engine = object.__new__(AgenticRolloutEngine)
    engine._drain_refs = list(refs)
    return engine


def _unexpected(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("unexpected finalization side effect")


def test_finalize_if_drained_is_a_rank_zero_broadcast_method() -> None:
    assert AgenticRolloutEngine.finalize_if_drained._distributed_config == {
        "dispatch_mode": Dispatch.BROADCAST,
        "execute_mode": Execute.RANK_ZERO,
    }


def test_finalize_if_drained_without_active_refs_is_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine_with_refs([])
    engine._fan = _unexpected
    monkeypatch.setattr(engine_module.ray, "wait", _unexpected)
    monkeypatch.setattr(engine_module.ray, "get", _unexpected)

    assert engine.finalize_if_drained() == []
    assert engine._drain_refs == []


def test_finalize_if_drained_returns_none_while_any_ref_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = [object(), object()]
    engine = _engine_with_refs(refs)
    engine._fan = _unexpected
    waits: list[tuple[list[Any], int, int]] = []

    def wait(actual_refs: list[Any], *, num_returns: int, timeout: int):
        waits.append((actual_refs, num_returns, timeout))
        return [actual_refs[0]], [actual_refs[1]]

    monkeypatch.setattr(engine_module.ray, "wait", wait)
    monkeypatch.setattr(engine_module.ray, "get", _unexpected)

    assert engine.finalize_if_drained() is None
    assert waits == [(refs, len(refs), 0)]
    assert engine._drain_refs == refs


@pytest.mark.parametrize("completions", [[], [Sample.request(Part.input(["root"]))]])
def test_finalize_if_drained_joins_clears_and_fans_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    completions: list[Sample],
) -> None:
    refs = [object(), object()]
    engine = _engine_with_refs(refs)
    gets: list[list[Any]] = []
    fans: list[str] = []

    monkeypatch.setattr(
        engine_module.ray,
        "wait",
        lambda actual_refs, *, num_returns, timeout: (list(actual_refs), []),
    )

    def get(actual_refs: list[Any]) -> list[None]:
        gets.append(actual_refs)
        return [None] * len(actual_refs)

    def fan(method: str) -> list[Sample]:
        assert engine._drain_refs == []
        fans.append(method)
        return completions

    monkeypatch.setattr(engine_module.ray, "get", get)
    engine._fan = fan

    assert engine.finalize_if_drained() is completions
    assert engine.finalize_if_drained() == []
    assert gets == [refs]
    assert fans == ["drain_completed"]


def test_finalize_if_drained_surfaces_join_failure_without_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    refs = [object()]
    engine = _engine_with_refs(refs)
    engine._fan = _unexpected
    monkeypatch.setattr(engine_module.ray, "wait", lambda actual_refs, **_kwargs: (list(actual_refs), []))

    def fail_join(_refs: list[Any]) -> None:
        raise RuntimeError("worker drain failed")

    monkeypatch.setattr(engine_module.ray, "get", fail_join)

    with pytest.raises(RuntimeError, match="worker drain failed"):
        engine.finalize_if_drained()
    assert engine._drain_refs == []
