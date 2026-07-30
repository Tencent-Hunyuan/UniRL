"""Expert parallelism (EP) for the VeOmni backend (package).

Public API only. Model-agnostic broadcast loading lives in :mod:`.load` and
shared DTensor mechanics in :mod:`.placement`; per-model fused layouts live in
:mod:`.models`. A meta-init bundle's ``materialize`` calls
:func:`load_ep_experts` to fill EP-sharded fused expert params that DCP's
rank-0 full-state path cannot represent.
"""

from unirl.train.backend.veomni.ep.load import (
    load_ep_experts,
    register_unsharded_param_hooks,
)
from unirl.train.backend.veomni.ep.placement import assign_local_block

__all__ = ["load_ep_experts", "register_unsharded_param_hooks", "assign_local_block"]
