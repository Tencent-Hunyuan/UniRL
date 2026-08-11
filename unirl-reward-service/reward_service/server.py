"""FastAPI gateway.

Accepts batches of RewardRequest. For each request, the required_rewards
list is fanned out to the corresponding WorkerGroup. Ray object refs
are awaited concurrently on the event loop so slow rewards never block
fast ones within the same batch.

Error isolation: if one reward group fails (OOM / parse error / crash
/ timeout), its error is captured per-reward and reported in
`errors[i][reward]`; the remaining rewards for the same request still
return scores. The per-reward deadline is `server.score_timeout_s`.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from reward_service.config import ServiceCfg
from reward_service.logging_utils import get_logger
from reward_service.schemas import PROTOCOL_VERSION, RewardRequest, ScoreRequest, ScoreResponse
from reward_service.scorers import ScoreItem
from reward_service.wire import request_to_item
from reward_service.workers.pool import WorkerPool

logger = get_logger(__name__)


def _bucket_by_reward(
    requests: list[RewardRequest],
    items: list[ScoreItem],
    pool: WorkerPool,
) -> dict[str, tuple[list[int], list[ScoreItem]]]:
    """Group items by reward name, keeping original request indices alongside.

    Returns a mapping ``reward_name -> (indices, items)`` where the two lists
    are parallel. Building them together avoids a second pass over the
    buckets when aggregating results in the /score handler.
    """
    indices: dict[str, list[int]] = defaultdict(list)
    bucketed_items: dict[str, list[ScoreItem]] = defaultdict(list)
    for i, req in enumerate(requests):
        for reward_name in req.required_rewards:
            if not pool.has_reward(reward_name):
                raise HTTPException(
                    status_code=400,
                    detail=f"unknown reward: {reward_name}. available: {pool.reward_names()}",
                )
            indices[reward_name].append(i)
            bucketed_items[reward_name].append(items[i])
    return {name: (indices[name], bucketed_items[name]) for name in indices}


async def _await_ref(
    pool: WorkerPool, name: str, ref, timeout_s: float
) -> tuple[str, list[dict[str, float]] | Exception]:
    """Await one dispatch handle with a per-reward deadline.

    Timeout leaves the underlying Ray task running: cancelling vLLM actors
    would cost a multi-minute reload. The actor's ``max_concurrency`` and
    ``num_replicas`` absorb the overhang.
    """
    try:
        result = await asyncio.wait_for(pool.as_awaitable(ref), timeout=timeout_s)
        return name, result
    except TimeoutError as e:
        logger.warning("reward %s timed out after %.1fs", name, timeout_s)
        wrapped = TimeoutError(f"reward {name!r} exceeded {timeout_s}s")
        wrapped.__cause__ = e
        return name, wrapped
    except Exception as e:
        logger.exception("reward %s failed: %s", name, e)
        return name, e


def create_app(cfg: ServiceCfg) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # WorkerPool() blocks until every actor's __init__ has finished
        # loading its model, so uvicorn only starts accepting requests
        # after the service is truly ready (readiness-probe semantics).
        app.state.pool = await asyncio.to_thread(WorkerPool, cfg)
        logger.info(
            "reward service ready on %s:%d — rewards=%s",
            cfg.server.host,
            cfg.server.port,
            app.state.pool.reward_names(),
        )
        try:
            yield
        finally:
            app.state.pool.shutdown()

    app = FastAPI(title="Reward Service", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        pool: WorkerPool = app.state.pool
        health_info = await asyncio.to_thread(pool.health)
        return {"status": "ok", "rewards": health_info}

    @app.get("/rewards")
    async def list_rewards() -> dict:
        pool: WorkerPool = app.state.pool
        return {"rewards": pool.reward_names()}

    @app.post("/score", response_model=ScoreResponse)
    async def score(body: ScoreRequest) -> ScoreResponse:
        pool: WorkerPool = app.state.pool
        if body.protocol_version != PROTOCOL_VERSION:
            raise HTTPException(status_code=400, detail=f"unsupported protocol_version={body.protocol_version!r}")
        if any(request.scorer_version is not None for request in body.requests):
            # One gateway request may name several rewards and the pool reports no
            # per-scorer version, so this identity field can never be filled in
            # truthfully here. Say so instead of echoing None and making the client
            # reject every response as a version mismatch.
            raise HTTPException(
                status_code=400,
                detail=(
                    "scorer_version pinning is not supported by the multi-reward gateway; "
                    "omit the scorer_version pin (expected_scorer_version in RemoteRewardSpec) "
                    "or point the client at a managed single-scorer child"
                ),
            )
        if not body.requests:
            return ScoreResponse(results=[], errors=[])

        items = await asyncio.to_thread(lambda: [request_to_item(r) for r in body.requests])
        buckets = _bucket_by_reward(body.requests, items, pool)

        timeout_s = cfg.server.score_timeout_s
        name_to_ref = {name: pool.dispatch(name, bucket_items) for name, (_, bucket_items) in buckets.items()}
        gathered = dict(
            await asyncio.gather(*(_await_ref(pool, name, ref, timeout_s) for name, ref in name_to_ref.items()))
        )

        results: list[dict[str, dict[str, float]]] = [dict() for _ in body.requests]
        errors: list[dict[str, str]] = [dict() for _ in body.requests]
        for name, bucket_scores in gathered.items():
            indices, _ = buckets[name]
            if isinstance(bucket_scores, Exception):
                error_msg = repr(bucket_scores)
                for i in indices:
                    errors[i][name] = error_msg
                continue
            for i, sub_metrics in zip(indices, bucket_scores):
                results[i][name] = sub_metrics
        return ScoreResponse(
            results=results,
            errors=errors,
            identities=[request.identity() for request in body.requests],
        )

    return app
