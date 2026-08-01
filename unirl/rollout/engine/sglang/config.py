"""``sglang`` engine config — wired by ``_target_`` (like every engine config).

No port math: the engine reserves its own :class:`SGLangPorts` at boot, so
there is no ``find_free_port`` here and the ``port`` field is accepted but
ignored (kept for recipe-shape stability). ``model_family`` selects the adapter
and defaults from the ``image_token`` VLM switch, so text/VLM recipes need no
extra key.

``server_intent`` (the successor of the hand-maintained ServerArgs allowlist)
spells this config + the reserved ports as the SGLang ServerArgs intent dict;
the backend filters it against the real ServerArgs fields and spawns.
Explicit first-class correctness flags are marked as required so the backend
fails closed when the installed SGLang ``ServerArgs`` cannot accept them.
"""

from __future__ import annotations

import random
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig
from unirl.rollout.engine.ports import ReservedPorts

_SGLANG_GRPC_PORT_OFFSET = 30000
_SGLANG_MAX_DERIVED_GRPC_BASE_PORT = 65535 - _SGLANG_GRPC_PORT_OFFSET
_SGLANG_SAFE_SERVER_PORT_MIN = 1024
_REQUIRED_SERVER_ARGS_METADATA_KEY = "_unirl_required_server_args"
_LOAD_BEARING_SERVER_ARGS = frozenset(
    {
        "ep_size",
        "enable_expert_parallel",
        "enable_memory_saver",
        "enable_weights_cpu_backup",
        "skip_server_warmup",
    }
)


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
    # ``tp_size`` / ``pp_size`` / ``ep_size`` are read by Handle to build the
    # rollout rank layout; ``tp_size>1`` is what makes a multi-GPU SGLang engine.
    # ``dp_size`` is forwarded to SGLang ServerArgs but NOT read by UniRL's
    # Handle (UniRL derives dp_size = world_size // (tp*pp) internally); it is
    # kept as an escape hatch for SGLang's own data-parallel semantics. Leave
    # None unless a SGLang server-level dp override is explicitly needed.
    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    ep_size: Optional[int] = None
    dp_size: Optional[int] = None
    enable_expert_parallel: Optional[bool] = None

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

    # --- Colocated memory lifecycle ---
    # Optional so existing SGLang recipes retain upstream ServerArgs defaults.
    # Colocated trainers opt in explicitly when they need sleep/wake to hand
    # GPU ownership between rollout and FSDP.
    enable_memory_saver: Optional[bool] = None
    enable_weights_cpu_backup: Optional[bool] = None
    skip_server_warmup: Optional[bool] = None

    # --- Sample expansion contract ---
    # VLMTrainer pre-expands the request by samples_per_prompt (P prompts → P*N
    # entries, one per GRPO sibling), so the engine must emit exactly ONE
    # completion per entry (n=1) — matching the trainside pipeline, else samples
    # double-count (P*N entries × N each). Standalone callers (e.g. the smoke
    # driver) pass unexpanded prompts and want the engine to fan out
    # n=samples_per_prompt itself; they leave this False.
    samples_pre_expanded: bool = False

    # --- VLM multimodal ---
    # Non-None selects the VLM adapter and loads AutoProcessor. Typed image
    # content still goes through the checkpoint's official chat template and
    # processor; this value is not injected as the complete image marker.
    # None (default) selects text-only mode.
    image_token: Optional[str] = None

    # --- LLM sampling (forwarded to SGLang /generate sampling_params) ---
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    # Exact tokenizer tokens excluded from response sampling via SGLang's
    # logit_bias. This never changes prompt tokens. A train-side replay path
    # must exclude the same ids before log-softmax to preserve log-prob parity.
    response_forbidden_tokens: Optional[List[str]] = None

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
            self.pp_size is None or self.pp_size >= 1,
            f"SGLangEngineConfig.pp_size must be >= 1 when set; got {self.pp_size!r}",
        )
        # pp_size>1 is currently unsupported end-to-end: ``Handle`` would build
        # one Remote per pp_rank while a single SGLang ``Engine`` also spawns
        # its own tp*pp scheduler subprocesses internally, so the two layouts
        # would double-book the GPUs (see backends' spawn-scoped CVD mapping).
        # ``NCCLWeightSync.connect`` already
        # raises NotImplementedError for pp_size>1; fail closed at config-time
        # so users hit a clear error before rollout boot.
        require(
            self.pp_size is None or self.pp_size == 1,
            "SGLangEngineConfig.pp_size>1 is not supported yet: UniRL Handle "
            "would spawn one engine per pp_rank while SGLang Engine also spawns "
            "its own PP scheduler subprocesses, double-booking the GPUs. Set "
            "pp_size=1 (or leave it unset) for now; per-stage rank_offset "
            "routing and single-engine PP fan-out are future work "
            f"(got pp_size={self.pp_size!r}).",
        )
        require(
            self.ep_size is None or self.ep_size >= 1,
            f"SGLangEngineConfig.ep_size must be >= 1 when set; got {self.ep_size!r}",
        )
        # SGLang derives moe_tp_size = tp_size // ep_size with plain integer
        # division, so a non-divisible ep_size silently builds a wrong MoE
        # group layout instead of erroring. Fail closed at config-time.
        effective_tp = self.tp_size if self.tp_size is not None else 1
        require(
            self.ep_size is None or (self.ep_size <= effective_tp and effective_tp % self.ep_size == 0),
            "SGLangEngineConfig.ep_size must divide tp_size: SGLang derives "
            "moe_tp_size = tp_size // ep_size, so ep_size must be a divisor of "
            f"tp_size (got ep_size={self.ep_size!r}, tp_size={self.tp_size!r}).",
        )
        require(
            self.dp_size is None or self.dp_size >= 1,
            f"SGLangEngineConfig.dp_size must be >= 1 when set; got {self.dp_size!r}",
        )
        # dp_size>1 is unsupported for the same double-booking reason as
        # pp_size>1: UniRL's Handle sizes its Remote layout from tp*pp only,
        # while SGLang ServerArgs.dp_size>1 spawns dp_size*tp_size scheduler
        # subprocesses — the extra replicas would silently claim GPUs the
        # Handle believes are free. Fail closed at config-time.
        require(
            self.dp_size is None or self.dp_size == 1,
            "SGLangEngineConfig.dp_size>1 is not supported yet: UniRL Handle "
            "derives data parallelism from world_size // (tp*pp) and does not "
            "account for SGLang server-level DP replicas, which would "
            "double-book GPUs. Set dp_size=1 (or leave it unset) "
            f"(got dp_size={self.dp_size!r}).",
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
        runtime_overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Spell this config (+ the reserved ports) as ServerArgs intent.

        Unfiltered: the backend filters against the real ServerArgs fields and
        spawns. Non-ServerArgs escape-hatch keys still drop harmlessly there,
        but explicitly requested first-class correctness flags are recorded so
        the backend can fail closed if the installed runtime cannot accept them.
        Precedence (low → high): ``engine_kwargs`` escape-hatch < typed cfg
        fields < adapter ``extra`` < runtime overrides < the reserved ports. The
        trailing ``setdefault``s supply the predecessor's defaults (bind-all
        host so the server accepts cross-node connections; mem_fraction 0.88)
        without shadowing an escape-hatch override.
        """
        intent: Dict[str, Any] = {}

        # Layer 1: escape-hatch (lowest priority).
        intent.update(self.engine_kwargs or {})

        # Layer 2: typed cfg fields.
        intent["model_path"] = self.pretrained_model_ckpt_path
        if self.tp_size is not None:
            intent["tp_size"] = int(self.tp_size)
        if self.pp_size is not None:
            intent["pp_size"] = int(self.pp_size)
        if self.ep_size is not None:
            intent["ep_size"] = int(self.ep_size)
        if self.dp_size is not None:
            intent["dp_size"] = int(self.dp_size)
        if self.enable_expert_parallel is not None:
            intent["enable_expert_parallel"] = bool(self.enable_expert_parallel)
        if self.enable_memory_saver is not None:
            intent["enable_memory_saver"] = bool(self.enable_memory_saver)
        if self.enable_weights_cpu_backup is not None:
            intent["enable_weights_cpu_backup"] = bool(self.enable_weights_cpu_backup)
        if self.skip_server_warmup is not None:
            intent["skip_server_warmup"] = bool(self.skip_server_warmup)
        if self.host is not None:
            intent["host"] = str(self.host)

        # Layer 3: adapter model-specific extras (override hook).
        if extra:
            intent.update(extra)

        # Layer 4: runtime overrides (per-rank rollout layout; higher than cfg).
        if runtime_overrides:
            intent.update(runtime_overrides)

        # Record explicit load-bearing UniRL fields before adding default
        # fallbacks. If a runtime lacks one of these ServerArgs, silently
        # dropping it would change correctness or memory-lifecycle semantics.
        required_server_args = sorted(set(intent) & _LOAD_BEARING_SERVER_ARGS)
        if required_server_args:
            intent[_REQUIRED_SERVER_ARGS_METADATA_KEY] = required_server_args

        # Layer 5: the reserved ports (highest) — real ServerArgs fields.
        intent["port"] = ports.server_port
        intent["nccl_port"] = ports.nccl_port

        intent.setdefault("host", "0.0.0.0")
        intent.setdefault("tp_size", 1)
        intent.setdefault("pp_size", 1)
        intent.setdefault("ep_size", 1)
        intent.setdefault("mem_fraction_static", 0.88)

        return intent


__all__ = ["SGLangEngineConfig", "SGLangPorts"]
