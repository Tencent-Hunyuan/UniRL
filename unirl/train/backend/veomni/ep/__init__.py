"""Expert parallelism (EP) for the VeOmni backend (package)."""

from unirl.train.backend.veomni.ep.load import (
    load_ep_experts,
    register_unsharded_param_hooks,
)
from unirl.train.backend.veomni.ep.placement import assign_local_block
from unirl.train.backend.veomni.ep.router_replay import (
    MoERouterReplayController,
    ReplayMode,
    get_active_replay,
    maybe_replay_indices,
    patch_moe_router_replay,
    record_moe_routing,
    replay_moe_routing,
    set_active_replay,
)

__all__ = [
    "MoERouterReplayController",
    "ReplayMode",
    "assign_local_block",
    "get_active_replay",
    "load_ep_experts",
    "maybe_replay_indices",
    "patch_moe_router_replay",
    "record_moe_routing",
    "register_unsharded_param_hooks",
    "replay_moe_routing",
    "set_active_replay",
]
