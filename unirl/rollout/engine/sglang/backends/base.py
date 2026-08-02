"""The backend seam contract — the ``Backend`` protocol + the wire types.

Every ``sglang`` collaborator reaches the SGLang SRT runtime through this
protocol; the real implementations live beside it (``http.py`` — SRT server
subprocess + sync HTTP; ``native.py`` — in-process Engine). This module owns only small CPU-only helpers, so it is trivially CPU-importable.

**No RL types cross this seam.** ``generate`` takes ready-to-POST ``/generate``
payload dicts (one per prompt) and returns ``list[RawResult]`` (a structural view
of one parsed ``/generate`` candidate); the adapters do the
``Sample``↔wire translation. The impl absorbs its transport
asymmetries (thread fan-out, retries, SGLang's dict-vs-list response shape for
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
    """Filter ``server_intent`` against real SGLang ``ServerArgs`` fields.

    Escape-hatch keys may still drop, but first-class UniRL config fields
    recorded under ``_REQUIRED_SERVER_ARGS_METADATA_KEY`` must survive the
    installed runtime's ``ServerArgs`` filter. Those fields are correctness or
    memory-lifecycle knobs, so silently dropping them would make the recipe say
    one thing while SGLang runs another.
    """
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

    # generation — synchronous and safe for CONCURRENT callers: the agentic
    # drain calls it from one thread per trajectory, and the impl must keep the
    # in-flight requests batching together on the runtime (never serialize them).
    def generate(self, requests: List[Dict[str, Any]]) -> List[RawResult]: ...
    # best-effort controls
    def abort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
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
