"""``sglang`` engine config — wired by ``_target_`` (like every engine config).

No port math: the engine reserves its own :class:`SGLangPorts` at boot, so
there is no ``find_free_port`` here and the ``port`` field is accepted but
ignored (kept for recipe-shape stability). ``model_family`` selects the adapter
and defaults from the ``image_token`` VLM switch, so text/VLM recipes need no
extra key.

``server_intent`` (the successor of the hand-maintained ServerArgs allowlist)
spells this config + the reserved ports as the SGLang ServerArgs intent dict;
the backend filters it against the real ServerArgs fields and spawns.
"""

from __future__ import annotations

import random
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig
from unirl.rollout.engine.ports import ReservedPorts

_SGLANG_GRPC_PORT_OFFSET = 30000
_SGLANG_MAX_DERIVED_GRPC_BASE_PORT = 65535 - _SGLANG_GRPC_PORT_OFFSET
_SGLANG_SAFE_SERVER_PORT_MIN = 1024


def _bind_tcp_port(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("", int(port)))
    except Exception:
        sock.close()
        raise
    return sock


def _reserve_safe_server_port() -> socket.socket:
    """Reserve a SGLang server port whose derived gRPC port cannot overflow."""
    last_error: Optional[Exception] = None
    for _ in range(1024):
        server_port = random.randint(_SGLANG_SAFE_SERVER_PORT_MIN, _SGLANG_MAX_DERIVED_GRPC_BASE_PORT)
        try:
            return _bind_tcp_port(server_port)
        except OSError as exc:
            last_error = exc
            continue
    raise OSError(
        f"no free SGLang server port in [{_SGLANG_SAFE_SERVER_PORT_MIN}, {_SGLANG_MAX_DERIVED_GRPC_BASE_PORT}]"
    ) from last_error


@dataclass(frozen=True)
class SGLangPorts(ReservedPorts):
    """The ports one SRT server spawn consumes.

    - ``server_port`` — the HTTP bind (``ServerArgs.port``). Some SGLang
      runtimes derive gRPC as ``port + 30000``, so reservation keeps this
      <= 35535.
    - ``nccl_port`` — ``ServerArgs.nccl_port``: colocate runs N engines per
      node, each initializing its own torch.distributed env. SGLang left with
      ``nccl_port=None`` calls get_free_port() at model-init time, so instances
      that finish loading together race onto the *same* port → EADDRINUSE.
      Reserving it here (de-synchronized across workers, like ``server_port``)
      hands SGLang an explicit port so it never re-picks at the synchronized
      post-load moment.
    """

    server_port: int
    nccl_port: int

    def __post_init__(self) -> None:
        super().__post_init__()
        require(
            self.server_port <= _SGLANG_MAX_DERIVED_GRPC_BASE_PORT,
            "SGLangPorts.server_port must be <= "
            f"{_SGLANG_MAX_DERIVED_GRPC_BASE_PORT} because SGLang derives grpc_port as port + "
            f"{_SGLANG_GRPC_PORT_OFFSET}; got {self.server_port}",
        )

    @classmethod
    def reserve(cls) -> "SGLangPorts":
        """Reserve SGLang HTTP and NCCL ports on this node."""
        socks = []
        try:
            server_sock = _reserve_safe_server_port()
            socks.append(server_sock)
            nccl_sock = _bind_tcp_port(0)
            socks.append(nccl_sock)
            return cls(
                server_port=server_sock.getsockname()[1],
                nccl_port=nccl_sock.getsockname()[1],
            )
        finally:
            for sock in socks:
                sock.close()


@dataclass
class SGLangEngineConfig(BaseEngineConfig):
    """Configuration for the ``sglang`` rollout engine."""

    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

        return SGLangRolloutEngine(config=self, **deps)

    # --- Model ---
    pretrained_model_ckpt_path: str = ""

    # --- Adapter selection (registry key; None = derived from image_token) ---
    model_family: Optional[str] = None

    # --- Parallelism & GPU ---
    tp_size: Optional[int] = None

    # --- SGLang network ---
    # ``host`` is the SRT bind address (default 0.0.0.0 so the server accepts
    # cross-node connections). ``port`` is kept for config-shape parity with
    # the predecessor; the engine self-reserves its ports — inject a typed
    # ``SGLangPorts`` (tests) instead of pinning this field.
    host: Optional[str] = None
    port: Optional[int] = None

    # --- Backend transport selection ---
    # "http" (default): SRT server subprocess + HTTP client. "native":
    # in-process sglang.Engine (no HTTP hop; the schedulers are still
    # subprocesses).
    backend: str = "http"

    # --- Concurrency / async ---
    concurrency: int = 8

    # --- Sample expansion contract ---
    # VLMTrainer pre-expands the request by samples_per_prompt (P prompts → P*N
    # entries, one per GRPO sibling), so the engine must emit exactly ONE
    # completion per entry (n=1) — matching the trainside pipeline, else samples
    # double-count (P*N entries × N each). Standalone callers (e.g. the smoke
    # driver) pass unexpanded prompts and want the engine to fan out
    # n=samples_per_prompt itself; they leave this False.
    samples_pre_expanded: bool = False

    # --- VLM multimodal ---
    # Image token placeholder injected into the chat template at image
    # positions.  Model-specific: e.g. "<|vision_start|><|image_pad|><|vision_end|>"
    # for Qwen2.5-VL.  None (default) = text-only mode.
    image_token: Optional[str] = None

    # --- LLM sampling (forwarded to SGLang /generate sampling_params) ---
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 1024

    # --- Chat template ---
    # System message prepended to every prompt (e.g. "/no_think" to suppress
    # Qwen3's thinking mode), used as the fallback when a per-request stage
    # config doesn't carry one. Must match the trainside pipeline's
    # system_instruction so generation and replay see the same prompt.
    system_instruction: Optional[str] = None
    # Extra kwargs forwarded to tokenizer.apply_chat_template (e.g.
    # {enable_thinking: false} for Qwen3 — without it the model emits a long
    # <think> block that overruns max_new_tokens before reaching the answer).
    chat_template_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    # --- Escape hatch for advanced ServerArgs / engine knobs ---
    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        require(
            bool(self.pretrained_model_ckpt_path),
            "SGLangEngineConfig.pretrained_model_ckpt_path must be set",
        )
        require(
            self.tp_size is None or self.tp_size >= 1,
            f"SGLangEngineConfig.tp_size must be >= 1 when set; got {self.tp_size!r}",
        )
        require(
            self.concurrency >= 1,
            f"SGLangEngineConfig.concurrency must be >= 1; got {self.concurrency!r}",
        )
        require(
            self.max_new_tokens >= 1,
            f"SGLangEngineConfig.max_new_tokens must be >= 1; got {self.max_new_tokens!r}",
        )
        require(
            self.temperature > 0,
            f"SGLangEngineConfig.temperature must be > 0; got {self.temperature!r}",
        )
        require(
            0.0 < self.top_p <= 1.0,
            f"SGLangEngineConfig.top_p must be in (0, 1]; got {self.top_p!r}",
        )

        self.backend = str(self.backend).strip().lower()
        require(
            self.backend in ("http", "native"),
            f"SGLangEngineConfig.backend must be 'http' or 'native'; got {self.backend!r}",
        )

        # Adapter selection: derive from the predecessor's VLM switch when not
        # explicit, then validate against the live registry (importing it
        # registers the families).
        if self.model_family is None:
            self.model_family = "vlm" if self.image_token is not None else "text"
        self.model_family = str(self.model_family).strip().lower()
        from unirl.rollout.engine.sglang.adapters import registered_adapters

        valid_families = registered_adapters()
        require(
            self.model_family in valid_families,
            f"SGLangEngineConfig.model_family must be one of {set(valid_families)}; got {self.model_family!r}",
        )

    # ------------------------------------------------------------------
    # SGLang ServerArgs intent (successor of the hand-maintained allowlist)
    # ------------------------------------------------------------------

    def server_intent(
        self,
        *,
        ports: SGLangPorts,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Spell this config (+ the reserved ports) as ServerArgs intent.

        Unfiltered: the backend filters against the real ServerArgs fields and
        spawns (non-ServerArgs escape-hatch keys drop harmlessly there).
        Precedence (low → high): ``engine_kwargs`` escape-hatch < typed cfg
        fields < adapter ``extra`` < the reserved ports. The trailing
        ``setdefault``s supply the predecessor's defaults (bind-all host so the
        server accepts cross-node connections; mem_fraction 0.88) without
        shadowing an escape-hatch override.
        """
        intent: Dict[str, Any] = {}

        # Layer 1: escape-hatch (lowest priority).
        intent.update(self.engine_kwargs or {})

        # Layer 2: typed cfg fields.
        intent["model_path"] = self.pretrained_model_ckpt_path
        if self.tp_size is not None:
            intent["tp_size"] = int(self.tp_size)
        if self.host is not None:
            intent["host"] = str(self.host)

        # Layer 3: adapter model-specific extras (override hook).
        if extra:
            intent.update(extra)

        # Layer 4: the reserved ports (highest) — real ServerArgs fields.
        intent["port"] = ports.server_port
        intent["nccl_port"] = ports.nccl_port

        intent.setdefault("host", "0.0.0.0")
        intent.setdefault("tp_size", 1)
        intent.setdefault("mem_fraction_static", 0.88)

        return intent


__all__ = ["SGLangEngineConfig", "SGLangPorts"]
