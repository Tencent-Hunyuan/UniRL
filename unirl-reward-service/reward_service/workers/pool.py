"""WorkerPool: one WorkerGroup per configured reward.

Handles Ray init/shutdown and batch dispatch across groups. The service
layer asks the pool to score a mixed batch: the pool splits by reward
name and returns per-reward Ray futures, leaving aggregation to callers.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import Future
from typing import Any, Awaitable

import ray

from reward_service.config import ClusterCfg, ServiceCfg
from reward_service.logging_utils import get_logger
from reward_service.scorers import ScoreItem
from reward_service.workers.group import WorkerGroup

logger = get_logger(__name__)


def _init_ray(cluster: ClusterCfg) -> None:
    """Bring Ray up according to ClusterCfg, logging which mode we took.

    Three mutually exclusive cases:
      1. Ray already initialised in this process — reuse it. Happens in
         tests or when an outer harness called ``ray.init`` first.
      2. ``cluster.ray_address`` is set — connect to that address
         (``"auto"`` for a cluster on this host, or a GCS / Ray Client
         URI for a remote one). ``namespace`` is forwarded so multiple
         services can coexist without colliding on named actors.
      3. ``ray_address`` is None — start a fresh local single-host
         runtime (original behaviour).
    """
    if ray.is_initialized():
        gcs = ray.get_runtime_context().gcs_address
        logger.info("reusing existing Ray runtime; gcs_address=%s", gcs)
        return

    if cluster.ray_address:
        init_kwargs: dict[str, Any] = {"address": cluster.ray_address}
        if cluster.namespace:
            init_kwargs["namespace"] = cluster.namespace
        ray.init(**init_kwargs)
        gcs = ray.get_runtime_context().gcs_address
        logger.info(
            "connected to Ray cluster at %s (namespace=%s); gcs_address=%s",
            cluster.ray_address, cluster.namespace, gcs,
        )
        return

    ray.init()
    logger.info("initialized new local Ray runtime")


class WorkerPool:
    def __init__(self, cfg: ServiceCfg) -> None:
        self.cfg = cfg
        _init_ray(cfg.cluster)
        self._groups: dict[str, WorkerGroup] = {}
        for reward_cfg in cfg.rewards:
            self._groups[reward_cfg.name] = WorkerGroup(reward_cfg)
        self._wait_until_all_actors_ready()

    def _wait_until_all_actors_ready(self) -> None:
        """Block until every actor's ``__init__`` has finished loading its model.

        ``ScorerActor.__init__`` does the heavy lifting (HF download, vLLM
        engine boot, etc.) but Ray schedules it asynchronously — the handle
        returned from ``actor_cls.remote()`` is usable immediately even if
        the actor is still loading. Calling ``ping.remote()`` and blocking
        on the result forces Ray to serialize this method behind the actor's
        constructor, so the call only returns after ``__init__`` is done.

        Without this barrier, uvicorn would start accepting requests while
        some scorers are still initializing — early requests would either
        queue for minutes (transformers cold load) or surface a Ray
        ``ActorDiedError`` (vLLM / hpsv3 crashed in ``__init__``). Failing
        fast at startup matches the semantics of a k8s readiness probe.
        """
        logger.info("waiting for all actors to finish __init__ ...")
        t0 = time.perf_counter()
        pending: list[tuple[str, int, Any]] = [
            (name, idx, actor.ping.remote())
            for name, group in self._groups.items()
            for idx, actor in enumerate(group.actors)
        ]
        for name, idx, ref in pending:
            ready_msg = ray.get(ref)
            logger.info(
                "actor[%s#%d] ready (%.1fs elapsed) — %s",
                name, idx, time.perf_counter() - t0, ready_msg,
            )
        logger.info(
            "all %d actors ready in %.1fs", len(pending), time.perf_counter() - t0,
        )

    def reward_names(self) -> list[str]:
        return sorted(self._groups)

    def has_reward(self, name: str) -> bool:
        return name in self._groups

    def dispatch(self, reward_name: str, items: list[ScoreItem]) -> Any:
        if reward_name not in self._groups:
            raise KeyError(
                f"unknown reward: {reward_name}. available: {sorted(self._groups)}"
            )
        return self._groups[reward_name].dispatch(items)

    @staticmethod
    def as_awaitable(ref: Any) -> Awaitable[Any]:
        """Adapt a dispatch handle into an asyncio-awaitable.

        Keeps the Ray-specific ``ObjectRef.future()`` bridge inside the
        workers/ package so callers (server.py, tests) can stay backend-
        agnostic. Accepts any handle that exposes ``.future()`` returning a
        ``concurrent.futures.Future``, which covers both real Ray refs and
        the test stubs.
        """
        fut: Future = ref.future()
        return asyncio.wrap_future(fut)

    def health(self) -> dict[str, list[str]]:
        return {name: group.ping() for name, group in self._groups.items()}

    def shutdown(self) -> None:
        for group in self._groups.values():
            group.shutdown()
        self._groups.clear()
