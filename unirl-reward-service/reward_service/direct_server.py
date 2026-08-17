"""Single-scorer loopback server for rank-affine managed image rewards."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import hashlib
import importlib
import json
import math
import os
import signal
import socket
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from reward_service.logging_utils import get_logger
from reward_service.schemas import PROTOCOL_VERSION, RewardIdentity, ScoreRequest, ScoreResponse
from reward_service.scorers.registry import SCORER_MODULES, get_scorer_cls
from reward_service.wire import request_to_item

logger = get_logger(__name__)

_PR_SET_PDEATHSIG = 1


def _arm_parent_death_signal() -> None:
    parent_pid = int(os.environ.get("UNIRL_REWARD_PARENT_PID", "0") or 0)
    if parent_pid <= 0:
        return
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() != parent_pid:
        raise SystemExit("reward parent exited before child initialization")
    _start_orphan_reaper(parent_pid)


def _start_orphan_reaper(parent_pid: int, grace_seconds: float = 20.0) -> None:
    """Guarantee no scorer subprocess outlives an orphaned server.

    PDEATHSIG only covers this process: an engine-backed scorer (e.g. a vLLM
    judge) spawns its own workers, and when this server dies by signal those
    grandchildren reparent to init still holding GPU memory — engine-side
    parent monitors have been observed not to fire, an idle/slept engine in
    particular. This daemon thread watches for reparenting and, after a grace
    period for the SIGTERM-driven graceful shutdown to finish, SIGKILLs the
    server's own process group (the parent starts us with start_new_session,
    so the group is exactly this server plus everything it spawned).
    """

    def _watch() -> None:
        while True:
            if os.getppid() != parent_pid:
                time.sleep(grace_seconds)
                # Only nuke the group when we lead it (start_new_session in the
                # managed parent). Under a debug shell the group is the user's
                # session — exit alone instead of killing their terminal.
                if os.getpgid(0) == os.getpid():
                    try:
                        os.killpg(os.getpgid(0), signal.SIGKILL)
                    except OSError:
                        pass
                os._exit(1)
            time.sleep(1.0)

    threading.Thread(target=_watch, name="orphan-reaper", daemon=True).start()


def _normalize_score(
    result: Any,
    *,
    required_metrics: tuple[str, ...],
) -> tuple[dict[str, float], str | None]:
    if not isinstance(result, Mapping):
        return {}, f"scorer returned {type(result).__name__}, expected a metric mapping"
    missing = sorted(set(required_metrics) - set(result))
    if not result or missing:
        return {}, f"scorer omitted required metrics: {missing or list(required_metrics)}"
    normalized: dict[str, float] = {}
    invalid: list[str] = []
    for name, value in result.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            invalid.append(str(name))
            continue
        if not math.isfinite(numeric):
            invalid.append(str(name))
            continue
        normalized[str(name)] = numeric
    if invalid:
        return {}, f"non-finite or non-numeric metrics: {sorted(invalid)}"
    return normalized, None


def _request_fingerprint(request: Any) -> str:
    payload = request.model_dump(mode="json", exclude={"idempotency_key"})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_put(
    cache: OrderedDict,
    key: str,
    value: tuple[str, dict[str, float], str | None],
    limit: int,
) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > limit:
        cache.popitem(last=False)


def create_direct_app(
    scorer_name: str,
    params: dict[str, Any],
    *,
    history_kind: str = "image",
    cache_size: int = 1024,
    boot_offloaded: bool = False,
) -> FastAPI:
    if history_kind not in {"image", "image_edit"}:
        raise ValueError(f"history_kind must be image or image_edit, got {history_kind!r}")
    if cache_size < 1:
        raise ValueError(f"cache_size must be positive, got {cache_size}")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        module_path = SCORER_MODULES.get(scorer_name)
        if module_path:
            importlib.import_module(module_path)
        scorer_cls = get_scorer_cls(scorer_name)
        scorer_input_kind = str(getattr(scorer_cls, "input_kind", "image"))
        if scorer_input_kind != "image":
            raise ValueError(
                f"managed image server requires scorer input_kind='image'; "
                f"{scorer_name!r} declares {scorer_input_kind!r}"
            )
        if boot_offloaded and not getattr(scorer_cls, "supports_offload", False):
            # Checked on the CLASS, before construction: a per_call
            # misconfiguration must not pay a full model load (possibly onto the
            # exact GPU the protocol protects) just to be told no.
            raise ValueError(
                f"--boot-offloaded requires scorer {scorer_name!r} to support offload (needed for gpu_residency='per_call')"
            )
        logger.info("direct scorer loading name=%s params=%s", scorer_name, params)
        scorer = await asyncio.to_thread(scorer_cls, **params)
        if not tuple(scorer.sub_metric_names):
            raise ValueError(f"managed scorer {scorer_name!r} declares no sub_metric_names")
        app.state.scorer = scorer
        app.state.score_lock = asyncio.Lock()
        app.state.result_cache = OrderedDict()
        if boot_offloaded:
            # per_call boot protocol: leave the GPU BEFORE the parent can see us
            # ready. Waiting for the parent's first /lifecycle/offload would keep
            # the scorer resident next to whatever the trainer already holds for
            # the whole readiness poll — exactly the window that OOMs a colocated
            # boot. Scorers that honor UNIRL_SCORER_BOOT_OFFLOADED construct on
            # CPU and make this offload a no-op; engine-backed scorers at least
            # shrink the window to construction itself.
            await asyncio.to_thread(scorer.offload)
            app.state.lifecycle_state = "offloaded"
        else:
            app.state.lifecycle_state = "resident"
        logger.info(
            "direct scorer ready name=%s version=%s state=%s",
            scorer_name,
            scorer.version,
            app.state.lifecycle_state,
        )
        try:
            yield
        finally:
            if app.state.lifecycle_state != "stopping":
                await asyncio.to_thread(scorer.close)

    app = FastAPI(title=f"Direct Reward Service ({scorer_name})", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        scorer = app.state.scorer
        return {
            "status": "ok",
            "state": app.state.lifecycle_state,
            "scorer": scorer.health(),
            "rewards": {scorer_name: [app.state.lifecycle_state]},
        }

    @app.get("/rewards")
    async def rewards() -> dict:
        return {"rewards": [scorer_name]}

    @app.post("/score", response_model=ScoreResponse)
    async def score(body: ScoreRequest) -> ScoreResponse:
        if body.protocol_version != PROTOCOL_VERSION:
            raise HTTPException(status_code=400, detail=f"unsupported protocol_version={body.protocol_version!r}")
        if not body.requests:
            return ScoreResponse(results=[], errors=[], identities=[])
        for request in body.requests:
            unknown = [name for name in request.required_rewards if name != scorer_name]
            if unknown:
                raise HTTPException(status_code=400, detail=f"unknown rewards for this worker: {unknown}")
            if history_kind == "image_edit" and len(request.history) < 2:
                raise HTTPException(status_code=400, detail="image_edit requests require source and edited turns")

        # Fingerprinting (and the decode below) is CPU work proportional to the b64
        # media in the chunk. Run on the event loop it stalls /health and the
        # lifecycle endpoints for the whole walk, so the parent sees a live child
        # as unavailable mid-scoring.
        fingerprints = await asyncio.to_thread(lambda: [_request_fingerprint(request) for request in body.requests])

        async with app.state.score_lock:
            if app.state.lifecycle_state != "resident":
                raise HTTPException(
                    status_code=409,
                    detail=f"scorer is not resident (state={app.state.lifecycle_state})",
                )
            results: list[dict[str, dict[str, float]]] = [{} for _ in body.requests]
            errors: list[dict[str, str]] = [{} for _ in body.requests]
            identities: list[RewardIdentity] = [
                request.identity(actual_scorer_version=app.state.scorer.version) for request in body.requests
            ]
            # One in-flight chunk must never evict its own rows, or a retry after a
            # lost response recomputes exactly what the cache exists to deduplicate.
            cache_limit = max(cache_size, len(body.requests))
            uncached_requests = []
            uncached_indices: list[int] = []
            pending_fingerprints: dict[str, str] = {}
            pending_key_indices: dict[str, int] = {}
            duplicate_indices: dict[int, int] = {}
            for index, request in enumerate(body.requests):
                key = request.idempotency_key
                fingerprint = fingerprints[index]
                if key:
                    prior = pending_fingerprints.setdefault(key, fingerprint)
                    if prior != fingerprint:
                        raise HTTPException(
                            status_code=409,
                            detail=f"idempotency key {key!r} was reused for different payloads in one batch",
                        )
                cached = app.state.result_cache.get(key) if key else None
                if cached is not None:
                    fingerprint, normalized, error = cached
                    if fingerprint != pending_fingerprints[key]:
                        raise HTTPException(
                            status_code=409,
                            detail=f"idempotency key {key!r} was reused for a different payload",
                        )
                    results[index] = {scorer_name: dict(normalized)} if error is None else {}
                    errors[index] = {} if error is None else {scorer_name: error}
                    app.state.result_cache.move_to_end(key)
                    continue
                if key and key in pending_key_indices:
                    duplicate_indices[index] = pending_key_indices[key]
                    continue
                if key:
                    pending_key_indices[key] = index
                uncached_requests.append(request)
                uncached_indices.append(index)

            if uncached_requests:
                uncached_items = await asyncio.to_thread(
                    lambda: [request_to_item(request, allow_video=False) for request in uncached_requests]
                )
                try:
                    scores = await asyncio.to_thread(app.state.scorer.score, uncached_items)
                except Exception as exc:
                    logger.exception("direct scorer failed: %s", exc)
                    error = repr(exc)
                    for index in uncached_indices:
                        errors[index] = {scorer_name: error}
                else:
                    if len(scores) != len(uncached_items):
                        raise RuntimeError(
                            f"direct scorer returned {len(scores)} results for {len(uncached_items)} items"
                        )
                    for index, scorer_result in zip(uncached_indices, scores, strict=True):
                        normalized, error = _normalize_score(
                            scorer_result,
                            required_metrics=tuple(app.state.scorer.sub_metric_names),
                        )
                        results[index] = {scorer_name: normalized} if error is None else {}
                        errors[index] = {} if error is None else {scorer_name: error}
                        key = body.requests[index].idempotency_key
                        if key:
                            _cache_put(
                                app.state.result_cache,
                                key,
                                (fingerprints[index], normalized, error),
                                cache_limit,
                            )
            for index, source_index in duplicate_indices.items():
                results[index] = dict(results[source_index])
                errors[index] = dict(errors[source_index])

        return ScoreResponse(results=results, errors=errors, identities=identities)

    @app.post("/lifecycle/{action}")
    async def lifecycle(action: str) -> dict:
        if action not in {"onload", "offload", "drain", "shutdown"}:
            raise HTTPException(status_code=404, detail=f"unknown lifecycle action {action!r}")
        async with app.state.score_lock:
            scorer = app.state.scorer
            if app.state.lifecycle_state == "stopping" and action != "shutdown":
                raise HTTPException(status_code=409, detail="scorer is stopping")
            if action == "onload":
                if app.state.lifecycle_state == "resident":
                    pass
                else:
                    app.state.lifecycle_state = "loading"
                    try:
                        await asyncio.to_thread(scorer.onload)
                    except Exception:
                        if not scorer.supports_offload:
                            app.state.lifecycle_state = "error"
                        else:
                            try:
                                await asyncio.to_thread(scorer.offload)
                            except Exception:
                                app.state.lifecycle_state = "error"
                                logger.exception("failed to roll back scorer after onload error")
                            else:
                                app.state.lifecycle_state = "offloaded"
                        raise
                    app.state.lifecycle_state = "resident"
            elif action == "offload":
                if not scorer.supports_offload:
                    raise HTTPException(status_code=409, detail=f"scorer {scorer_name!r} does not support offload")
                if app.state.lifecycle_state != "offloaded":
                    app.state.lifecycle_state = "draining"
                    try:
                        await asyncio.to_thread(scorer.drain)
                        await asyncio.to_thread(scorer.offload)
                    except Exception:
                        app.state.lifecycle_state = "error"
                        raise
                    app.state.lifecycle_state = "offloaded"
            elif action == "drain":
                previous_state = app.state.lifecycle_state
                app.state.lifecycle_state = "draining"
                await asyncio.to_thread(scorer.drain)
                app.state.lifecycle_state = previous_state
            else:
                app.state.lifecycle_state = "stopping"
                await asyncio.to_thread(scorer.drain)
                await asyncio.to_thread(scorer.close)
            return {"status": "ok", "state": app.state.lifecycle_state}

    return app


def _spawn_group_reaper() -> None:
    """Sweep the process group if this frontend dies without running cleanup.

    PDEATHSIG covers parent->frontend only: when the frontend itself is
    SIGKILLed (ray worker teardown at driver exit, an external kill), engine
    subprocesses spawned by the scorer — e.g. a vLLM EngineCore — survive
    re-parented to init and keep their GPU memory. Observed repeatedly with
    both sleeping and resident engines whose own frontend-liveness watchdog
    never fired. The parent's _stop_child() killpg covers the graceful path;
    this sibling covers every path where nobody calls it: it shares this
    process group, watches this PID, and sweeps the group (TERM, 5s grace,
    KILL) the moment the frontend is gone. On a normal exit the group by then
    holds only the reaper itself, so the sweep is a self-kill.
    """
    import subprocess
    import sys

    if os.getpgrp() != os.getpid():
        # Not the group leader: the group belongs to whoever launched us (a
        # wrapper script, a Popen without start_new_session), and sweeping it
        # would kill unrelated processes. The managed parent always starts the
        # frontend with start_new_session, so this only skips manual launches.
        logger.warning(
            "group reaper disabled: frontend pid %d is not its process-group leader (pgrp %d)",
            os.getpid(),
            os.getpgrp(),
        )
        return

    source = (
        "import os,signal,sys,time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "watched = int(os.environ['_UNIRL_REAPER_WATCH_PID'])\n"
        "while os.getppid() == watched:\n"
        "    time.sleep(2.0)\n"
        "group = os.getpgrp()\n"
        "print(f'reward group reaper: frontend {watched} gone, sweeping pgid {group}', file=sys.stderr, flush=True)\n"
        "try:\n"
        "    os.killpg(group, signal.SIGTERM)\n"
        "except ProcessLookupError:\n"
        "    pass\n"
        "time.sleep(5.0)\n"
        "try:\n"
        "    os.killpg(group, signal.SIGKILL)\n"
        "except ProcessLookupError:\n"
        "    pass\n"
    )
    env = dict(os.environ, _UNIRL_REAPER_WATCH_PID=str(os.getpid()))
    subprocess.Popen(
        [sys.executable, "-c", source],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        # The parent already routes this frontend's stderr into the per-child
        # log; a sweep that vaporizes the group must leave a trace there.
        stderr=None,
        close_fds=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", required=True)
    parser.add_argument("--params-json", required=True)
    parser.add_argument("--history-kind", default="image")
    parser.add_argument("--cache-size", type=int, default=1024)
    parser.add_argument(
        "--boot-offloaded",
        action="store_true",
        help="offload the scorer before first reporting ready (per_call boot protocol)",
    )
    parser.add_argument("--fd", type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    _arm_parent_death_signal()
    _spawn_group_reaper()
    params = json.loads(args.params_json)
    if not isinstance(params, dict):
        raise TypeError("--params-json must decode to an object")
    app = create_direct_app(
        args.scorer,
        params,
        history_kind=args.history_kind,
        cache_size=args.cache_size,
        boot_offloaded=args.boot_offloaded,
    )

    import uvicorn

    config = uvicorn.Config(app, host=args.host, port=args.port, log_level=args.log_level)
    server = uvicorn.Server(config)
    if args.fd is None:
        server.run()
        return
    with socket.socket(fileno=args.fd) as listener:
        server.run(sockets=[listener])


if __name__ == "__main__":
    main()
