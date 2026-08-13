"""Expert parallelism (EP) for the VeOmni backend (package)."""

from unirl.train.backend.veomni.ep.load import (
    load_ep_experts,
    register_unsharded_param_hooks,
)
from unirl.train.backend.veomni.ep.placement import assign_local_block

__all__ = ["load_ep_experts", "register_unsharded_param_hooks", "assign_local_block"]
