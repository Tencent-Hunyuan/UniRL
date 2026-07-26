"""The backend seam contract — the ``Backend`` protocol + the wire types."""

from __future__ import annotations

import os
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

_REQUIRED_SERVER_ARGS_METADATA_KEY = "_unirl_required_server_args"


def _filter_server_args_or_raise(
    server_intent: Dict[str, Any],
    *,
    allowed: set[str],
    backend_name: str,
) -> Dict[str, Any]:
    """Filter ``server_intent`` against real SGLang ``ServerArgs`` fields."""
    raw_required = server_intent.get(_REQUIRED_SERVER_ARGS_METADATA_KEY, ())
    if isinstance(raw_required, str):
        required = [raw_required]
    else:
        required = sorted({str(key) for key in raw_required})
    missing_required = [key for key in required if key not in allowed]
    if missing_required:
        raise RuntimeError(
            f"SGLang {backend_name} backend cannot apply required ServerArgs fields: {missing_required}. "
            "Upgrade SGLang to a build that supports these fields, or remove the explicit UniRL "
            "rollout config that depends on them."
        )
    return {k: v for k, v in server_intent.items() if k != _REQUIRED_SERVER_ARGS_METADATA_KEY and k in allowed}


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


def _normalize_cuda_visible_devices(
    cuda_visible_devices: Optional[Sequence[str]],
    *,
    tp_size: int,
) -> Optional[List[str]]:
    """Validate explicit scheduler CUDA tokens without interpreting them."""
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
    active_library_dirs = _python_cuda_library_dirs()
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
    """Structural view of one parsed SRT ``/generate`` candidate — the wire fields this engine consumes."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str


@runtime_checkable
class Backend(Protocol):
    """The seam every ``sglang`` collaborator reaches the runtime through."""

    # ``NativeBackend.update_from_ipc`` drives the engine event loop and must
    # run on the engine-owning thread. HTTP transport is thread-safe. Weight
    # sync uses this capability instead of coupling to concrete class names.
    requires_main_thread_ipc_receiver: bool

    def generate(self, requests: List[Dict[str, Any]]) -> List[RawResult]: ...
    def abort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def flush_cache(self) -> None: ...
    def release_memory(self, *, tags: Optional[Sequence[str]] = None) -> None: ...
    def resume_memory(self, *, tags: Optional[Sequence[str]] = None) -> None: ...
    def shutdown(self) -> None: ...
    def ping(self) -> bool: ...
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

    def update_from_ipc(
        self,
        *,
        zmq_handles: Dict[str, str],
        flush_cache: bool = True,
    ) -> None: ...


__all__ = ["Backend", "RawResult"]
