"""Fail-closed installer for the UniRL FastVideo patch suite; install order is load-bearing."""

from __future__ import annotations

from unirl.rollout.engine.fastvideo._patches.compat import (
    require_float_wan_timesteps,
    verify_offload_surface,
    verify_rl_data_surface,
    verify_stock_surface,
    verify_weight_surface,
)
from unirl.rollout.engine.fastvideo._patches.multiproc import patch_multiproc


def patch_worker_runtime() -> None:
    """Install every in-worker patch, fingerprinting the stock seams before they are rewritten."""
    require_float_wan_timesteps()
    verify_stock_surface()
    from unirl.rollout.engine.fastvideo._patches.conditions import patch_conditions
    from unirl.rollout.engine.fastvideo._patches.contracts import patch_contracts
    from unirl.rollout.engine.fastvideo._patches.denoising import patch_denoising
    from unirl.rollout.engine.fastvideo._patches.offload import patch_offload
    from unirl.rollout.engine.fastvideo._patches.unipc import patch_unipc
    from unirl.rollout.engine.fastvideo._patches.weights import patch_weights

    # Contract and denoising first: they add the RLData fields and the eta/sde_type
    # parameters that the UniPC fingerprints and dispatch below rely on.
    patch_contracts()
    patch_denoising()
    verify_rl_data_surface()
    patch_unipc()
    patch_conditions()
    patch_offload()
    patch_weights()


def patch_fastvideo() -> None:
    """Install idempotent parent, worker-entrypoint, and runtime patches after fingerprinting the pinned surface."""
    verify_weight_surface()
    verify_offload_surface()
    patch_worker_runtime()
    patch_multiproc()


__all__ = ["patch_fastvideo", "patch_worker_runtime"]
