"""Fail-closed runtime patch installer for FastVideo."""

from __future__ import annotations

import threading


class FastVideoHijack:
    """Install UniRL's FastVideo integration exactly once per interpreter."""

    _installed = False
    _lock = threading.RLock()

    @classmethod
    def hijack(cls) -> None:
        from unirl.rollout.engine.fastvideo._patches.compat import (
            verify_fastvideo_capabilities,
            verify_fastvideo_compatibility,
        )

        with cls._lock:
            if cls._installed:
                verify_fastvideo_capabilities()
                return

            verify_fastvideo_compatibility()

            from unirl.rollout.engine.fastvideo._patches.conditions import patch_conditions
            from unirl.rollout.engine.fastvideo._patches.contracts import patch_contracts
            from unirl.rollout.engine.fastvideo._patches.denoising import patch_denoising
            from unirl.rollout.engine.fastvideo._patches.multiproc import patch_multiproc
            from unirl.rollout.engine.fastvideo._patches.timesteps import patch_timesteps
            from unirl.rollout.engine.fastvideo._patches.weights import patch_weights

            # Contract first: later patches and worker serialization refer to
            # the extended RLData class. Spawn target is installed last, after
            # every patch it will re-install has been validated in the parent.
            patch_contracts()
            patch_timesteps()
            patch_denoising()
            patch_conditions()
            patch_weights()
            patch_multiproc()

            verify_fastvideo_capabilities()
            cls._installed = True


__all__ = ["FastVideoHijack"]
