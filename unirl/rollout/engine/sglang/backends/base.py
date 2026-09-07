"""The backend seam contract — the ``Backend`` protocol + the wire types."""

from __future__ import annotations

import base64
import io
import logging
import os
import pickle
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

logger = logging.getLogger(__name__)

_REQUIRED_SERVER_ARGS_METADATA_KEY = "_unirl_required_server_args"
_STRICT_SERVER_ARGS_ENV = "UNIRL_SGLANG_STRICT_SERVER_ARGS"
_UNIRL_ONLY_INTENT_KEYS = frozenset(
    {
        _REQUIRED_SERVER_ARGS_METADATA_KEY,
        "advertise_host",
        "concurrency",
        "health_timeout_s",
    }
)


def _serialize_lora_tensors(
    lora_tensors: Dict[str, Any],
    *,
    tp_size: int,
    multiprocessing_serializer: Any,
) -> str:
    """Use SGLang fd sharing for TP1 and byte-copy tensors for TP broadcast — see ../../README.md."""
    if int(tp_size) == 1:
        return multiprocessing_serializer.serialize(lora_tensors, output_str=True)
    buf = io.BytesIO()
    pickle.dump(lora_tensors, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _strict_dropped_server_args() -> bool:
    return os.environ.get(_STRICT_SERVER_ARGS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _unknown_server_arg_keys(server_intent: Dict[str, Any], allowed: set[str]) -> List[str]:
    """Intent keys that are neither live ``ServerArgs`` fields nor UniRL-only."""
    return sorted(key for key in server_intent if key not in allowed and key not in _UNIRL_ONLY_INTENT_KEYS)


def _filter_server_args_or_raise(
    server_intent: Dict[str, Any],
    *,
    allowed: set[str],
    backend_name: str,
) -> Dict[str, Any]:
    """Filter intent against live ServerArgs; unknown keys warn or raise — see ../../README.md."""
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
    dropped = _unknown_server_arg_keys(server_intent, allowed)
    if dropped:
        message = (
            f"SGLang {backend_name} backend dropping unknown ServerArgs keys: {dropped}. "
            "They are not fields on the installed SGLang ServerArgs (typo or version skew). "
            f"Set {_STRICT_SERVER_ARGS_ENV}=1 to fail closed."
        )
        if _strict_dropped_server_args():
            raise RuntimeError(message)
        logger.warning(message)
    return {k: v for k, v in server_intent.items() if k != _REQUIRED_SERVER_ARGS_METADATA_KEY and k in allowed}


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


class RawResult(Protocol):
    """Structural view of one parsed SRT ``/generate`` candidate — the wire fields this engine consumes."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str


@runtime_checkable
class Backend(Protocol):
    """The seam every ``sglang`` collaborator reaches the runtime through."""

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


__all__ = ["Backend", "RawResult"]
