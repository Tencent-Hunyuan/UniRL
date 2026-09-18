from __future__ import annotations

import pytest

from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine
from unirl.rollout.engine.base import resolve_rollout_capabilities
from unirl.rollout.engine.composed.engine import ComposedRolloutEngine


class _NonTransactionalInner:
    def __init__(self) -> None:
        self.init_kwargs = None

    def init_weights_update_group(self, **kwargs) -> None:
        self.init_kwargs = kwargs


class _TransactionalInner(_NonTransactionalInner):
    def begin_weights_update(self, **kwargs) -> None:
        pass

    def prepare_weights_update(self, **kwargs) -> None:
        pass

    def finish_weights_update(self, **kwargs) -> None:
        pass


def test_agentic_wrapper_does_not_forward_transaction_timeout_to_legacy_inner() -> None:
    engine = AgenticRolloutEngine.__new__(AgenticRolloutEngine)
    engine._inner = _NonTransactionalInner()
    engine.init_weights_update_group(group_name="weights", timeout_s=30.0)
    assert engine._inner.init_kwargs == {"group_name": "weights"}
    with pytest.raises(RuntimeError, match="does not support transactional"):
        engine.begin_weights_update(group_name="weights")


def test_agentic_wrapper_forwards_timeout_to_transactional_inner() -> None:
    engine = AgenticRolloutEngine.__new__(AgenticRolloutEngine)
    engine._inner = _TransactionalInner()
    engine.init_weights_update_group(group_name="weights", timeout_s=30.0)
    assert engine._inner.init_kwargs == {"group_name": "weights", "timeout_s": 30.0}


def test_composed_wrapper_rejects_nontransactional_routed_child_before_broadcast() -> None:
    engine = ComposedRolloutEngine.__new__(ComposedRolloutEngine)
    engine._child_by_name = {"diffusion": _NonTransactionalInner()}
    with pytest.raises(RuntimeError, match="does not support transactional"):
        engine.begin_weights_update(group_name="weights", track_prefix="diffusion")


def test_wrapper_capabilities_follow_configured_child() -> None:
    native = {
        "_target_": "unirl.rollout.engine.native_sd3.config.NativeSD3EngineConfig",
        "fp8_enabled": True,
    }
    legacy = {"_target_": "unirl.rollout.engine.sglang.config.SGLangEngineConfig"}
    agentic_native = {
        "_target_": "unirl.rollout.engine.agentic.config.AgenticRolloutEngineConfig",
        "inner": native,
    }
    composed_legacy = {
        "_target_": "unirl.rollout.engine.composed.config.ComposedRolloutEngineConfig",
        "ar": native,
        "diffusion": legacy,
    }
    assert resolve_rollout_capabilities(agentic_native).transactional_weight_publication
    assert not resolve_rollout_capabilities(composed_legacy).transactional_weight_publication
    assert resolve_rollout_capabilities(
        composed_legacy,
        track_prefix="ar",
    ).transactional_weight_publication
