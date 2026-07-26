"""The backend seam contract — the ``Backend`` protocol + the wire types.

Every ``sglang`` collaborator reaches the SGLang SRT runtime through this
protocol; the real implementation lives beside it (``http.py`` — SRT server
subprocess + HTTP). This module also owns the small, CPU-only environment guard
shared by both spawn implementations. Keeping that mutation scoped to the
child-spawn boundary prevents a Ray Worker from leaking its daemon's stale CUDA
library path into SGLang scheduler children.

**No RL types cross this seam.** ``generate`` takes ready-to-POST ``/generate``
payload dicts (one per prompt) and returns ``list[RawResult]`` (a structural view
of one parsed ``/generate`` candidate); the adapters do the
``RolloutReq``↔``RolloutResp`` translation. The impl absorbs its transport
asymmetries (async fan-out, retries, SGLang's dict-vs-list response shape for
``n``) behind these signatures.

Deliberate divergences from the ``sglang_diffusion`` seam:

- No ``target_modules`` on the update verbs — the diffusion-side default
  ``["transformer"]`` doesn't match LLM module naming; omitting the field lets
  the SRT server accept all incoming weights correctly.
- No ``weights_checksum`` — the checksum/verify path is vLLM-Omni-only.
- ``flush_cache`` is a first-class verb so the engine can orchestrate
  flush-before-sleep as a visible line.
"""

from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from glob import glob
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)


def _split_preloads(value: str) -> List[str]:
    return [item for item in re.split(r"[\s:]+", value) if item]


def _python_cuda_library_dirs() -> List[str]:
    """Return CUDA library directories shipped with the active Python env."""
    result: List[str] = []

    def _add(path: str) -> None:
        path = os.path.abspath(path)
        if os.path.isdir(path) and path not in result:
            result.append(path)

    for entry in sys.path:
        for path in sorted(glob(os.path.join(entry, "nvidia", "*", "lib"))):
            _add(path)

    try:
        import torch

        _add(os.path.join(os.path.dirname(torch.__file__), "lib"))
    except (ImportError, AttributeError):
        pass
    return result


def _preloaded_cuda_driver_libraries() -> List[str]:
    """Return explicitly preloaded CUDA forward-compatibility drivers."""
    result: List[str] = []
    for item in _split_preloads(os.environ.get("LD_PRELOAD", "")):
        if not os.path.basename(item).startswith("libcuda.so"):
            continue
        path = os.path.abspath(item)
        if os.path.isfile(path) and path not in result:
            result.append(path)
    return result


def _preloaded_cuda_driver_dirs() -> List[str]:
    """Return directories containing explicitly preloaded CUDA driver shims.

    ``torch-memory-saver`` replaces ``LD_PRELOAD`` with its own hook while it
    starts SGLang schedulers. Keeping a forward-compatibility driver's
    directory in ``LD_LIBRARY_PATH`` lets the child resolve that same
    ``libcuda`` without changing torch-memory-saver's single-library preload
    contract.
    """
    return list(dict.fromkeys(os.path.dirname(path) for path in _preloaded_cuda_driver_libraries()))


@contextmanager
def _preserve_cuda_driver_preloads(
    libraries: Optional[Sequence[str]] = None,
) -> Iterator[None]:
    """Keep driver shims loaded when torch-memory-saver starts schedulers.

    ``torch-memory-saver`` intentionally replaces ``LD_PRELOAD`` with its hook
    path. On Python builds whose executable has an ``RPATH`` to the host driver,
    ``LD_LIBRARY_PATH`` alone cannot make that hook resolve a forward-compatible
    ``libcuda``. Preload the driver first for the child process; the scheduler
    target teaches torch-memory-saver to select its hook from the composite
    preload while leaving the driver in the environment for compiler and worker
    subprocesses spawned later.
    """
    driver_libraries = list(libraries) if libraries is not None else _preloaded_cuda_driver_libraries()
    if not driver_libraries:
        yield
        return

    try:
        import torch_memory_saver
    except ImportError:
        yield
        return

    original_configure_subprocess = torch_memory_saver.configure_subprocess

    @contextmanager
    def _configure_subprocess_with_driver() -> Iterator[None]:
        with original_configure_subprocess():
            saved_preload = os.environ.get("LD_PRELOAD")
            current = _split_preloads(saved_preload or "")
            # Driver first is load-bearing: the Python executable's host-driver
            # RPATH otherwise wins while resolving the memory-saver hook.
            os.environ["LD_PRELOAD"] = os.pathsep.join(list(dict.fromkeys(driver_libraries + current)))
            try:
                yield
            finally:
                if saved_preload is None:
                    os.environ.pop("LD_PRELOAD", None)
                else:
                    os.environ["LD_PRELOAD"] = saved_preload

    torch_memory_saver.configure_subprocess = _configure_subprocess_with_driver
    try:
        yield
    finally:
        torch_memory_saver.configure_subprocess = original_configure_subprocess


_TMS_COMPOSITE_PRELOAD_SENTINEL = "_unirl_composite_preload"


def _configure_torch_memory_saver_composite_preload() -> None:
    """Let torch-memory-saver read its hook from a composite ``LD_PRELOAD``.

    Hook mode normally returns the entire environment value as one ``ctypes``
    path, so ``driver.so:memory_saver.so`` cannot initialize. Replacing that
    narrow lookup keeps initialization lazy while preserving the composite
    value for TorchInductor and other scheduler descendants that must load the
    forward-compatible CUDA driver themselves.
    """
    try:
        from torch_memory_saver.hooks.mode_preload import HookUtilModePreload
    except ImportError:
        return

    current = HookUtilModePreload.__dict__.get("get_path_binary")
    if getattr(current, _TMS_COMPOSITE_PRELOAD_SENTINEL, False):
        return

    def _get_path_binary(_self: Any) -> str:
        hooks = [
            item
            for item in _split_preloads(os.environ.get("LD_PRELOAD", ""))
            if "torch_memory_saver" in os.path.basename(item)
        ]
        if len(hooks) != 1:
            raise RuntimeError(
                f"SGLang CUDA compatibility spawn expected exactly one torch-memory-saver preload hook; got {hooks!r}"
            )
        return hooks[0]

    setattr(_get_path_binary, _TMS_COMPOSITE_PRELOAD_SENTINEL, True)
    HookUtilModePreload.get_path_binary = _get_path_binary


def _run_sglang_scheduler_with_cuda_driver_preload(*args: Any, **kwargs: Any) -> Any:
    """Scheduler target for a child started with driver + memory-saver hooks."""
    _configure_torch_memory_saver_composite_preload()

    from sglang.srt.managers.scheduler import run_scheduler_process

    return run_scheduler_process(*args, **kwargs)


def _normalize_cuda_visible_devices(
    cuda_visible_devices: Optional[Sequence[str]],
    *,
    tp_size: int,
) -> Optional[List[str]]:
    """Validate explicit scheduler CUDA tokens without interpreting them.

    Ray may expose numeric ordinals, GPU UUIDs, or MIG UUIDs. They are opaque
    tokens here; only cardinality and comma/empty ambiguity are rejected.
    ``None`` means preserve the Worker's existing CUDA visibility.
    """
    if cuda_visible_devices is None:
        return None
    tokens = [str(token).strip() for token in cuda_visible_devices]
    if len(tokens) != int(tp_size):
        raise ValueError(
            "SGLang scheduler CUDA visibility must contain exactly tp_size "
            f"tokens; got tp_size={tp_size}, tokens={tokens!r}"
        )
    if any(not token for token in tokens):
        raise ValueError(f"SGLang scheduler CUDA visibility contains an empty token: {tokens!r}")
    if any("," in token for token in tokens):
        raise ValueError(
            f"SGLang scheduler CUDA visibility expects one token per entry; comma-containing entry found in {tokens!r}"
        )
    return tokens


@contextmanager
def _scheduler_spawn_environment(
    cuda_visible_devices: Optional[Sequence[str]],
) -> Iterator[None]:
    """Quarantine environment changes to the SGLang child-spawn boundary.

    Ray Workers inherit environment variables from the already-running Ray
    daemon, not from the command that later submits a job. In particular, a
    stale ``LD_LIBRARY_PATH`` can point at a different CUDA toolkit and make a
    scheduler load the wrong runtime. Put the active Python environment's CUDA
    wheel directories first instead of clearing the variable: torch-memory-
    saver's preload hook is dynamically linked against ``libcudart`` and cannot
    start if that runtime directory is removed. The Worker's values are restored
    on both success and failure, so colocated training is unaffected.
    """
    saved_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
    saved_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    active_library_dirs = _preloaded_cuda_driver_dirs() + _python_cuda_library_dirs()
    inherited_library_dirs = (saved_ld_library_path or "").split(os.pathsep)
    child_library_dirs = list(dict.fromkeys(active_library_dirs + [path for path in inherited_library_dirs if path]))
    if child_library_dirs:
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(child_library_dirs)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible_devices)
    try:
        yield
    finally:
        if saved_ld_library_path is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = saved_ld_library_path
        if saved_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = saved_cuda_visible_devices


class RawResult(Protocol):
    """Structural view of one parsed SRT ``/generate`` candidate — the wire
    fields this engine consumes. The HTTP impl deserializes responses into this
    shape (``n>1`` returns a list of candidates per prompt; the impl flattens
    them prompt-major: candidate ``k`` of prompt ``i`` at index ``i*n + k``);
    test fakes stand in structurally.

    Population: ``text`` and ``finish_reason`` are always set. ``token_ids`` /
    ``logprobs`` both come from the ``meta_info['output_token_logprobs']``
    items — the runtime's only source of generated token ids (there is no
    separate ``output_token_ids`` field) — so they are length-aligned by
    construction, and both empty when the request didn't ask for logprobs.
    """

    #: The raw generated text (``<think>`` tags intact — stripping is a
    #: driver-side concern, applied by the adapter at decode time).
    text: str
    #: Generated token ids, always length-aligned with ``logprobs``.
    token_ids: List[int]
    #: Per-token log-probs; both lists empty when ``return_logprob`` was off.
    logprobs: List[float]
    #: Normalized finish reason (SRT returns a dict or a bare string).
    finish_reason: str


@runtime_checkable
class Backend(Protocol):
    """The seam every ``sglang`` collaborator reaches the runtime through."""

    # generation
    def generate(self, requests: List[Dict[str, Any]]) -> List[RawResult]: ...
    # memory / lifecycle / health
    def flush_cache(self) -> None: ...
    def release_memory(self, *, tags: Optional[Sequence[str]] = None) -> None: ...
    def resume_memory(self, *, tags: Optional[Sequence[str]] = None) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
    # weight-sync verbs (serialization stays inside the impl)
    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None: ...
    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None: ...
    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        flush_cache: bool,
    ) -> None: ...
    def destroy_weights_group(self, *, group_name: str) -> None: ...
    def set_lora(
        self,
        *,
        lora_name: str,
        lora_tensors: Dict[str, Any],
        config_dict: Optional[dict] = None,
    ) -> None: ...

    # update_from_ipc is intentionally absent — SGLang has no IPC receiver.


__all__ = ["Backend", "RawResult"]
