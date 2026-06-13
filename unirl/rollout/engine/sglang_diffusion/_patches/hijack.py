"""In-process monkey-patch installer for stock-upstream sglang diffusion (LIN-365).

Mirrors ``unirl/rollout/engine/vllm_omni/vllm_patches.py``: a
spawn-propagating hijack that re-hosts the ``sglang-drl`` fork's RL additions on
top of stock upstream sglang, so UniRL can track upstream instead of
carrying a hard fork.

Why direct setattr/REPLACE (not sglang's HookRegistry): the diffusion
scheduler/worker runs under forced spawn
(``diffusion_generator.py: mp.set_start_method("spawn", force=True)``) and the
diffusion path never calls srt ``load_plugins()``, so the official
``HookRegistry`` is not wired in. A parent-only patch would silently no-op in
the worker; ``wrap_mp_process_for_children`` propagates the install into every
spawn child instead.

Install once when the native backend boots -- BEFORE importing
``DiffGenerator`` (which forces spawn at import) and before ``from_pretrained``
spawns the scheduler. Idempotent; safe to call from both parent and child.
"""

from __future__ import annotations

import logging
from multiprocessing.process import BaseProcess as _MpBaseProcess

logger = logging.getLogger(__name__)


# ============================================================
# Subprocess propagation -- make spawn children also run hijack
# ============================================================
#
# The diffusion scheduler is launched via a spawn-context ``mp.Process``; the
# child is a fresh interpreter that does not inherit the parent's patches.
# Wrapping the target so it re-runs ``hijack()`` before the scheduler loop
# guarantees the child's ``Scheduler``/``GPUWorker``/``SchedulerRLMixin`` are
# patched before any request is served.


class _DiffrlPatchedTarget:
    """Pickleable wrapper that installs sglang patches in a spawn child first.

    Must be module-level so spawn's pickler can serialise the wrapped target
    across the process boundary (closures cannot be pickled).
    """

    def __init__(self, target):
        self._target = target

    def __call__(self, *args, **kwargs):
        SglangDiffusionHijack.hijack()
        return self._target(*args, **kwargs)


_WRAP_SENTINEL = "_unirl_sglang_target_wrapped"


def wrap_mp_process_for_children() -> None:
    """Replace ``BaseProcess.__init__`` so spawned targets install patches first.

    All mp-context Process classes (incl. the ``SpawnProcess`` the diffusion
    scheduler uses) inherit from ``BaseProcess``, so patching the root catches
    every context in one shot. Idempotent via ``_WRAP_SENTINEL``.
    """
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
    """Apply one patch; log-and-skip if its upstream target is unavailable.

    Patches are import-safe and idempotent, so a target missing in a given
    interpreter (e.g. a CPU-only unit-test process importing only the rollout
    math) must not abort the remaining patches.
    """
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

        # Import all patch entrypoints. Each is import-safe + idempotent; a
        # target unavailable in this interpreter is logged and skipped by
        # _safe_apply, so partial availability (e.g. a CPU-only unit-test
        # process importing only the rollout math) never aborts the rest.
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
        from unirl.rollout.engine.sglang_diffusion._patches.patch_pipeline import (
            patch_pipeline,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_platform_device import (
            patch_platform_device,
        )
        from unirl.rollout.engine.sglang_diffusion._patches.patch_rollout_trajectory import (
            patch_rollout_trajectory,
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
        from unirl.rollout.engine.sglang_diffusion._patches.patch_weights_updater import (
            patch_weights_updater,
        )

        # (A) Additive infra: srt is_available shim; SamplingParams/Req IO
        #     fields; GPUWorker RL methods + sleep/wake; weight-sync;
        #     in-memory LoRA; RL Scheduler handlers.
        # (B) post1 bridge: grouped-stage dispatch (v0.5.12.post1 predates the
        #     3142278c5 grouped-path fix; no-op on any sglang that has it).
        # (C) The one REPLACE: the DanceGRPO objective upstream lacks.
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
        ):
            _safe_apply(patch)
