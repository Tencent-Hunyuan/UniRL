"""Dispatch-marker guard — the base.py footguns, regression-proofed.

Two rules from the engine layout spec: (1) ``@distributed`` is not inherited,
so ``generate`` / ``sleep`` / ``wake_up`` must carry the marker on THIS class
with the right modes; (2) the weight-sync entry points are reached per worker
via the raw ``Worker.call`` RPC by name — decorating one silently changes its
fan-out semantics, so they must NOT carry the marker.
"""

from __future__ import annotations

import pytest

from unirl.distributed.group.dispatch import DISTRIBUTED_CONFIG_ATTR, Dispatch
from unirl.rollout.engine.sglang_v2.engine import SGLangV2RolloutEngine


@pytest.mark.parametrize(
    ("method", "mode"),
    [
        ("generate", Dispatch.DP_SCATTER),
        ("sleep", Dispatch.BROADCAST),
        ("wake_up", Dispatch.BROADCAST),
    ],
)
def test_dispatched_methods_carry_marker_with_mode(method, mode):
    attr = SGLangV2RolloutEngine.__dict__[method]  # most-derived, not inherited
    config = getattr(attr, DISTRIBUTED_CONFIG_ATTR)
    assert config["dispatch_mode"] is mode


@pytest.mark.parametrize(
    "method",
    [
        "update_weights_from_tensor",
        "init_weights_update_group",
        "update_weights_from_distributed",
        "destroy_weights_update_group",
        "set_lora_from_tensors",
        "onload_weights",
        "health_check",
        "shutdown",
    ],
)
def test_raw_rpc_methods_are_undecorated(method):
    attr = getattr(SGLangV2RolloutEngine, method)
    assert not hasattr(attr, DISTRIBUTED_CONFIG_ATTR)


def test_surface_methods_are_real_class_attributes():
    """No ``__getattr__`` delegation — Handle scans ``dir()``; builders
    introspect ``inspect.signature``."""
    for method in (
        "generate",
        "sleep",
        "wake_up",
        "onload_weights",
        "health_check",
        "shutdown",
        "update_weights_from_tensor",
        "init_weights_update_group",
        "update_weights_from_distributed",
        "destroy_weights_update_group",
        "set_lora_from_tensors",
    ):
        assert method in dir(SGLangV2RolloutEngine)
    # update_weights_from_ipc stays the base's NotImplementedError stub —
    # deliberately not overridden (SGLang has no IPC receiver).
    assert "update_weights_from_ipc" not in SGLangV2RolloutEngine.__dict__
