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
from unirl.rollout.engine.ports import ReservedPorts
from unirl.rollout.engine.synchronous import BaseEngineConfig

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

    pretrained_model_ckpt_path: str = ""

    model_family: Optional[str] = None

    tp_size: Optional[int] = None
    pp_size: Optional[int] = None
    ep_size: Optional[int] = None
    dp_size: Optional[int] = None
    enable_expert_parallel: Optional[bool] = None

    host: Optional[str] = None
    port: Optional[int] = None

    backend: str = "http"

    concurrency: int = 8

    enable_memory_saver: Optional[bool] = None
    enable_weights_cpu_backup: Optional[bool] = None
    skip_server_warmup: Optional[bool] = None

    samples_pre_expanded: bool = False

    image_token: Optional[str] = None

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    response_forbidden_tokens: Optional[List[str]] = None

    system_instruction: Optional[str] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

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

        if self.model_family is None:
            self.model_family = "vlm" if self.image_token is not None else "text"
        self.model_family = str(self.model_family).strip().lower()
        from unirl.rollout.engine.sglang.adapters import registered_adapters

        valid_families = registered_adapters()
        require(
            self.model_family in valid_families,
            f"SGLangEngineConfig.model_family must be one of {set(valid_families)}; got {self.model_family!r}",
        )

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

        intent.update(self.engine_kwargs or {})

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

        if extra:
            intent.update(extra)

        if runtime_overrides:
            intent.update(runtime_overrides)

        required_server_args = sorted(set(intent) & _LOAD_BEARING_SERVER_ARGS)
        if required_server_args:
            intent[_REQUIRED_SERVER_ARGS_METADATA_KEY] = required_server_args

        intent["port"] = ports.server_port
        intent["nccl_port"] = ports.nccl_port

        intent.setdefault("host", "0.0.0.0")
        intent.setdefault("tp_size", 1)
        intent.setdefault("pp_size", 1)
        intent.setdefault("ep_size", 1)
        intent.setdefault("mem_fraction_static", 0.88)

        return intent


__all__ = ["SGLangEngineConfig", "SGLangPorts"]
