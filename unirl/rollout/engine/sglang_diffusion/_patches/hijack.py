"""In-process monkey-patch installer for stock-upstream sglang diffusion."""

from __future__ import annotations

import logging
from multiprocessing.process import BaseProcess as _MpBaseProcess

logger = logging.getLogger(__name__)


class _DiffrlPatchedTarget:
    """Pickleable wrapper that installs sglang patches in a spawn child first."""

    def __init__(self, target):
        self._target = target

    def __call__(self, *args, **kwargs):
        import os as _os

        if _os.environ.get("UNIRL_SGLANG_KEEP_NCCL_ENV") not in ("1", "true"):
            _topo = _os.environ.get("NCCL_TOPO_FILE")
            if _topo is not None and not _os.path.exists(_topo):
                _os.environ.pop("NCCL_TOPO_FILE", None)
            for _k in (
                "NCCL_SOCKET_IFNAME",
                "NCCL_BUFFSIZE",
                "NCCL_NET_FORCE_FLUSH",
                "NCCL_NVLSTREE_MAX_CHUNKSIZE",
                "NCCL_NVLS_CHUNKSIZE",
                "NCCL_P2P_NET_CHUNKSIZE",
                "NCCL_TUNER_PLUGIN",
            ):
                _os.environ.pop(_k, None)

        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline as _lp  # noqa: F401
        except Exception:
            pass

        SglangDiffusionHijack.hijack()
        return self._target(*args, **kwargs)


_WRAP_SENTINEL = "_unirl_sglang_target_wrapped"


def wrap_mp_process_for_children() -> None:
    """Replace ``BaseProcess.__init__`` so spawned targets install patches first."""
    if getattr(_MpBaseProcess, _WRAP_SENTINEL, False):
        return

    orig_init = _MpBaseProcess.__init__

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs=None,
        *,
        daemon=None,
    ):
        if target is not None and not isinstance(target, _DiffrlPatchedTarget):
            target = _DiffrlPatchedTarget(target)
        orig_init(
            self,
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs or {},
            daemon=daemon,
        )

    _MpBaseProcess.__init__ = __init__
    setattr(_MpBaseProcess, _WRAP_SENTINEL, True)


def _safe_apply(patch_fn) -> None:
    """Apply one patch; log-and-skip if its upstream target is unavailable."""
    try:
        patch_fn()
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning(
            "sglang patch %s skipped: %r",
            getattr(patch_fn, "__name__", patch_fn),
            exc,
        )


class SglangDiffusionHijack:
    """Installs all UniRL sglang patches. Mirrors ``VLLMOmniHijack``."""

    @staticmethod
    def hijack() -> None:
        # Spawn shim MUST run first so the scheduler/worker child re-installs.
        wrap_mp_process_for_children()

        from unirl.rollout.engine.sglang_diffusion._patches.patch_conditions import (
            patch_conditions,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_dance import patch_dance
        from unirl.rollout.engine.sglang_diffusion._patches.patch_denoising import (
            patch_denoising,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_gpu_worker import (
            patch_gpu_worker,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_grouped_dispatch import (
            patch_grouped_dispatch,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_latent_prep import (
            patch_latent_prep,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_lora_slice_2d import (
            patch_lora_slice_2d,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_lora_tensors import (
            patch_lora_tensors,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_ltx2_rollout_sde import (
            patch_ltx2_rollout_sde,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_pipeline import (
            patch_pipeline,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_platform_device import (
            patch_platform_device,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_rollout_trajectory import (
            patch_rollout_trajectory,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_safe_unpickler import (
            patch_safe_unpickler,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_sampling_io import (
            patch_sampling_io,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_scheduler import (
            patch_scheduler,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_sd3_lora_pipeline import (
            patch_sd3_lora_pipeline,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_set_timesteps import (
            patch_set_timesteps,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_srt import patch_srt
        from unirl.rollout.engine.sglang_diffusion._patches.patch_vae_decode_safe import (
            patch_vae_decode_safe,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_wan_scheduler import (
            patch_wan_scheduler,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_weights_updater import (
            patch_weights_updater,
        )

        for patch in (
            patch_srt,
            patch_platform_device,
            patch_sampling_io,
            patch_conditions,
            patch_latent_prep,
            patch_rollout_trajectory,
            patch_pipeline,
            patch_grouped_dispatch,
            patch_gpu_worker,
            patch_weights_updater,
            patch_sd3_lora_pipeline,
            patch_lora_tensors,
            patch_lora_slice_2d,
            patch_scheduler,
            patch_denoising,
            patch_dance,
            patch_set_timesteps,
            patch_vae_decode_safe,
            patch_wan_scheduler,
            patch_ltx2_rollout_sde,
            patch_safe_unpickler,
        ):
            _safe_apply(patch)
