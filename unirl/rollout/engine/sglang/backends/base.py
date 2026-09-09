"""The backend seam contract — the ``Backend`` protocol + the wire types."""

from __future__ import annotations

from typing import (
    Any,
    Dict,
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
