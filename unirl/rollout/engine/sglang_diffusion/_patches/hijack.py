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
        # Two independent NCCL hazards in the sglang scheduler subprocess:
        #
        # (1) launch_server() deadlock — stale NCCL env vars inherited from the
        #     Ray worker's train-side FSDP setup (Remote.setup writes them into
        #     os.environ at remote.py:104). The scheduler subprocess is a fresh
        #     single-process dist world (num_gpus=1), but it inherits
        #     NCCL_TOPO_FILE pointing at a parent-process fd (/proc/self/fd/NNN)
        #     that doesn't exist here → NCCL ``new_group`` →
        #     ``eager_connect_single_device`` hangs on a dead pipe. The other
        #     NCCL knobs (SOCKET_IFNAME, BUFFSIZE, ...) are train-mesh-specific
        #     and have no correct value in this subprocess; let NCCL use
        #     defaults. gpu_worker.init_device_and_model re-sets MASTER_ADDR/
        #     MASTER_PORT/WORLD_SIZE/RANK before init_process_group, so those
        #     are not cleared. This hang is nondeterministic across workers
        #     (timing of NCCL topo detection), hence only some workers hit it.
        #
        # (2) HeartbeatMonitor SIGSEGV — after init_process_group succeeds, the
        #     NCCL HeartbeatMonitor thread starts and calls glibc ``getenv``
        #     (via DumpPipe::DumpPipe(int) → getCvarString) on a background
        #     thread. glibc ``getenv`` is NOT thread-safe: sglang's model-load
        #     path concurrently calls ``os.environ[k]=v`` (putenv, which may
        #     realloc the environ array) — e.g. lora_pipeline.py:33 sets
        #     TOKENIZERS_PARALLELISM at import, gpu_worker.py:126-130 sets
        #     MASTER_ADDR/MASTER_PORT/... If the windows overlap →
        #     use-after-free → SIGSEGV in getenv. This is also timing-dependent
        #     (FlowGRPO 06-24 didn't hit it; NFT 06-25 did). Setting
        #     TORCH_NCCL_ENABLE_MONITORING=0 (verified against libtorch_cuda.so
        #     strings) fully disables the monitor thread; the monitor provides
        #     no value in a single-process (world_size=1) NCCL world.
        #
        # NOTE (torch 2.11): ``TORCH_NCCL_ENABLE_MONITORING`` does not exist in
        # this build (the monitor has no disable flag; only
        # ``TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC``). The monitor thread therefore
        # always starts after ``init_process_group``. The remaining race is the
        # monitor thread's glibc ``getenv`` vs ``os.environ[k]=v`` (``putenv``)
        # during model load — e.g. ``lora_pipeline.py:33`` sets
        # ``TOKENIZERS_PARALLELISM`` at *import* time, and the lora module is
        # imported during model construction (after init_process_group).
        # Pre-setting ``TOKENIZERS_PARALLELISM`` and pre-importing the lora
        # pipeline module here (before the scheduler target runs, hence before
        # ``init_process_group`` starts the monitor) eliminates the concurrent
        # ``putenv``: by the time the monitor thread calls ``getenv``, no
        # further ``putenv`` will fire.
        import os as _os

        # PRECONDITION: this scrub assumes the scheduler subprocess hosts a
        # single-process NCCL world (num_gpus=1 / tp_size=1 — the only
        # validated colocate topology). Deployments that need these knobs
        # inside the subprocess (engine TP>1, multi-NIC hosts pinning
        # NCCL_SOCKET_IFNAME) can set UNIRL_SGLANG_KEEP_NCCL_ENV=1 to skip it.
        if _os.environ.get("UNIRL_SGLANG_KEEP_NCCL_ENV") not in ("1", "true"):
            # NCCL_TOPO_FILE is the actual deadlock trigger, but only when it
            # dangles (a /proc/self/fd/NNN path of the dead parent). A real,
            # readable topo file is a legitimate host-level setting — keep it.
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

        # Pre-set the env vars that sglang's gpu_worker.py:126-130 and
        # lora_pipeline.py:33 will (re)set later, so the later ``os.environ[k]=v``
        # assignments are no-ops on the glibc ``environ`` array only if the value
        # is unchanged — they still call ``putenv``. The robust guard is to
        # pre-import lora_pipeline (triggering its module-level ``putenv`` for
        # TOKENIZERS_PARALLELISM) HERE, before init_process_group.
        _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        try:
            import sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline as _lp  # noqa: F401
        except Exception:
            pass

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
            patch_wan_scheduler,
            patch_safe_unpickler,
        ):
            _safe_apply(patch)
