"""Service configuration: YAML schema → dataclasses.

One RewardModelCfg per reward (e.g. hpsv2, clip, unified_reward). The
scorer field selects which BaseScorer subclass to instantiate; params is
forwarded to its constructor verbatim. num_gpus and replicas control how
many Ray actors start and how many GPUs each one owns exclusively.

For multi-host deployments, ClusterCfg tells the WorkerPool to
``ray.init(address=...)`` against an externally-managed Ray cluster
instead of spinning up a local runtime. See scripts/cluster_up.sh for
the matching pdsh bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Single source of truth for the per-reward /score deadline. Referenced by
# both ServerCfg's dataclass default and load_config's parser default so
# the two can never drift.
_DEFAULT_SCORE_TIMEOUT_S = 120.0

# Ray scheduling strategies we surface in YAML. "pack" is Ray's default
# behaviour (no explicit strategy; the scheduler prefers filling a node
# before spilling to the next). "spread" asks Ray to distribute this
# group's actors across nodes — a soft hint, not a hard placement group.
_VALID_SCHEDULING = ("pack", "spread")


@dataclass(frozen=True)
class RewardModelCfg:
    name: str
    scorer: str
    # Path to a pip requirements file (e.g. "envs/clip.txt"). Ray creates
    # an isolated virtualenv per unique requirements set and runs the actor
    # inside it, so each scorer can pin its own transformers / vllm version
    # without conflicting with others.
    runtime_env: str
    num_replicas: int = 1
    num_gpus: float = 1.0
    num_cpus: float = 1.0
    # Ray actor max_concurrency: how many in-flight score() calls a single
    # actor may serve simultaneously. Keep 1 for GPU-bound scorers (vLLM,
    # hpsv3) whose internal batcher already handles concurrency; raise for
    # lightweight transformer scorers (clip, pickscore) where the GIL is
    # released during the forward pass.
    max_concurrency: int = 1
    # "pack" (default, Ray default behaviour) or "spread" (hint Ray to
    # distribute replicas across nodes). Only meaningful in a multi-node
    # cluster with num_replicas > 1. Single-replica groups ignore this.
    scheduling: str = "pack"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ServerCfg:
    host: str = "0.0.0.0"
    port: int = 8080
    # Per-reward timeout (seconds) applied inside /score. A reward that
    # exceeds this is reported as an error for that request but does not
    # block other rewards in the same batch.
    score_timeout_s: float = _DEFAULT_SCORE_TIMEOUT_S


@dataclass(frozen=True)
class ClusterCfg:
    """How the WorkerPool connects to Ray.

    - ``ray_address=None`` (default): ``ray.init()`` starts a local
      single-host runtime. Matches the original single-machine behaviour.
    - ``ray_address="auto"``: connect to a Ray cluster already running on
      this host (e.g. started by scripts/cluster_up.sh).
    - ``ray_address="ray://host:10001"`` / ``"<ip>:6379"``: connect to a
      remote Ray cluster's GCS or Ray Client endpoint.

    ``namespace`` is forwarded to ``ray.init(namespace=...)`` and lets
    multiple reward services coexist on the same cluster without seeing
    each other's named actors.
    """

    ray_address: str | None = None
    namespace: str | None = None


@dataclass(frozen=True)
class ServiceCfg:
    server: ServerCfg
    rewards: list[RewardModelCfg]
    cluster: ClusterCfg = field(default_factory=ClusterCfg)

    def reward_names(self) -> list[str]:
        return [r.name for r in self.rewards]


def _parse_bounded_number(
    raw: dict, key: str, default, *, caster, min_value, context: str
):
    """Coerce ``raw[key]`` with ``caster`` and require ``value >= min_value``.

    Used for the four numeric fields in a reward config that share the same
    "coerce to type, check lower bound, raise with context" shape. Returns
    the validated value.
    """
    value = caster(raw.get(key, default))
    if value < min_value:
        raise ValueError(f"{context} must be >= {min_value}, got {value}")
    return value


def _parse_cluster_cfg(raw: dict | None) -> ClusterCfg:
    """Parse the optional ``cluster:`` YAML section into a ClusterCfg.

    Both fields are optional; omitting the section keeps the legacy
    single-host behaviour (``ray.init()`` with no address).
    """
    if raw is None:
        return ClusterCfg()
    if not isinstance(raw, dict):
        raise ValueError(f"`cluster` must be a mapping, got {type(raw).__name__}")

    def _opt_str(key: str) -> str | None:
        value = raw.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"cluster.{key} must be a non-empty string or omitted, got {value!r}"
            )
        return value.strip()

    return ClusterCfg(ray_address=_opt_str("ray_address"), namespace=_opt_str("namespace"))


def load_config(path: str | Path) -> ServiceCfg:
    """Parse YAML into ServiceCfg, validating required fields and name uniqueness."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")

    server_raw = raw.get("server") or {}
    score_timeout_s = float(server_raw.get("score_timeout_s", _DEFAULT_SCORE_TIMEOUT_S))
    if score_timeout_s <= 0:
        raise ValueError(f"server.score_timeout_s must be > 0, got {score_timeout_s}")
    server = ServerCfg(
        host=server_raw.get("host", "0.0.0.0"),
        port=int(server_raw.get("port", 8080)),
        score_timeout_s=score_timeout_s,
    )

    cluster = _parse_cluster_cfg(raw.get("cluster"))

    rewards_raw = raw.get("rewards") or []
    if not isinstance(rewards_raw, list):
        raise ValueError("`rewards` must be a list")

    rewards: list[RewardModelCfg] = []
    seen: set[str] = set()
    for i, entry in enumerate(rewards_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"rewards[{i}] must be a mapping, got {type(entry).__name__}")
        name = entry.get("name")
        scorer = entry.get("scorer")
        if not name or not scorer:
            raise ValueError(f"rewards[{i}] missing required `name` or `scorer`")
        runtime_env = entry.get("runtime_env")
        if not runtime_env or not isinstance(runtime_env, str):
            raise ValueError(
                f"rewards[{i}] ({name}) missing required `runtime_env` "
                f"(path to a pip requirements file, e.g. envs/clip.txt)"
            )
        runtime_env_path = Path(runtime_env)
        if not runtime_env_path.is_file():
            raise FileNotFoundError(
                f"rewards[{i}] ({name}) runtime_env file not found: {runtime_env}"
            )
        if name in seen:
            raise ValueError(f"duplicate reward name: {name}")
        seen.add(name)
        ctx = f"rewards[{i}] ({name})"
        num_replicas = _parse_bounded_number(
            entry, "num_replicas", 1, caster=int, min_value=1,
            context=f"{ctx} num_replicas",
        )
        num_gpus = _parse_bounded_number(
            entry, "num_gpus", 1.0, caster=float, min_value=0,
            context=f"{ctx} num_gpus",
        )
        max_concurrency = _parse_bounded_number(
            entry, "max_concurrency", 1, caster=int, min_value=1,
            context=f"{ctx} max_concurrency",
        )
        scheduling = str(entry.get("scheduling", "pack")).lower()
        if scheduling not in _VALID_SCHEDULING:
            raise ValueError(
                f"{ctx} scheduling must be one of {_VALID_SCHEDULING}, got {scheduling!r}"
            )
        params = dict(entry.get("params") or {})
        # vLLM scorers place `tensor_parallel_size` N GPUs inside one Ray
        # actor via `ScorerActor.options(num_gpus=...)`. If the actor was
        # allocated fewer than N GPUs, vLLM's initialize_ray_cluster can't
        # find enough GPU resources in the node and crashes at startup with
        # a confusing "Current node has no GPU available" error. Catching it
        # here turns that into a fast, specific config-time failure.
        if "tensor_parallel_size" in params:
            try:
                tp = int(params["tensor_parallel_size"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{ctx} params.tensor_parallel_size must be an int, "
                    f"got {params['tensor_parallel_size']!r}"
                ) from e
            if tp > num_gpus:
                raise ValueError(
                    f"{ctx} params.tensor_parallel_size={tp} requires "
                    f"num_gpus>={tp}, got num_gpus={num_gpus}. vLLM's Ray "
                    f"executor will crash at actor init otherwise."
                )
        rewards.append(
            RewardModelCfg(
                name=name,
                scorer=scorer,
                runtime_env=runtime_env,
                num_replicas=num_replicas,
                num_gpus=num_gpus,
                num_cpus=float(entry.get("num_cpus", 1.0)),
                max_concurrency=max_concurrency,
                scheduling=scheduling,
                params=params,
            )
        )

    if not rewards:
        raise ValueError("at least one reward must be configured")

    return ServiceCfg(server=server, rewards=rewards, cluster=cluster)
