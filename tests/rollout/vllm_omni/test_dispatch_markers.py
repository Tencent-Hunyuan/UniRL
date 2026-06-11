"""Dispatch-marker guard — regression-proofs the decorated/un-decorated split.

The driver-side handle binds the most-derived attribute, so the ``@distributed``
markers must sit exactly where the engine intends: ``generate`` / ``sleep`` /
``wake_up`` decorated (re-applied on the overrides), the weight-sync entry
points un-decorated (reached per worker via the raw ``Worker.call`` RPC) —
EXCEPT ``set_lora_from_tensors_copy``, the documented v1-parity exception
(``RemoteLoraWeightSync(copy=True)`` reaches the disjoint-partition HI3
engines through it).
"""

from __future__ import annotations

from unirl.distributed.group.dispatch import DISTRIBUTED_CONFIG_ATTR, Dispatch
from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine

DECORATED = {
    "generate": Dispatch.DP_SCATTER,
    "sleep": Dispatch.BROADCAST,
    "wake_up": Dispatch.BROADCAST,
    # The documented exception — see the engine docstring.
    "set_lora_from_tensors_copy": Dispatch.BROADCAST,
}

UNDECORATED = {
    "update_weights_from_ipc",
    "init_weights_update_group",
    "update_weights_from_distributed",
    "destroy_weights_update_group",
    "update_weights_from_tensor",
    "set_lora_from_tensors",
    "loaded_param_checksums",
    "loaded_lora_checksums",
    "tp_per_stage",
    "health_check",
    "shutdown",
    "onload_weights",
}


def test_decorated_set_and_modes():
    for name, mode in DECORATED.items():
        config = getattr(getattr(VLLMOmniRolloutEngine, name), DISTRIBUTED_CONFIG_ATTR, None)
        assert config is not None, f"{name} must carry @distributed"
        assert config["dispatch_mode"] is mode, f"{name}: {config['dispatch_mode']} != {mode}"


def test_weight_sync_entry_points_undecorated():
    for name in UNDECORATED:
        attr = getattr(VLLMOmniRolloutEngine, name)
        assert getattr(attr, DISTRIBUTED_CONFIG_ATTR, None) is None, f"{name} must NOT carry @distributed"


def test_surface_methods_are_real_class_attributes():
    """No __getattr__ delegation — the handle scans dir() and the builder
    introspects signatures."""
    for name in list(DECORATED) + sorted(UNDECORATED):
        assert name in dir(VLLMOmniRolloutEngine), name
