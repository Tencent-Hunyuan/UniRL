"""Expert parallelism (EP) for the VeOmni backend (package).

Public API only. The model-agnostic EP-sharded weight loader lives in
:mod:`.load`; per-model EP wiring (fused-expert module + checkpoint converter +
parallel plan + meta-swap) lives in :mod:`.models` (e.g. :mod:`.models.hi3`).
A meta-init bundle's ``materialize`` calls :func:`load_ep_experts` to fill the
EP-sharded fused expert params that DCP's ``set_model_state_dict`` cannot.
"""

from unirl.train.backend.veomni.ep.load import (
    load_ep_experts,
    register_unsharded_param_hooks,
)

__all__ = ["load_ep_experts", "register_unsharded_param_hooks"]
