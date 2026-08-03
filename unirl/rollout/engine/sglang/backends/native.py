"""The native ``Backend`` impl — in-process ``sglang.Engine`` (no HTTP hop).

The in-process twin of :mod:`.http`: the SGLang import is lazy (only
:func:`_import_sglang_engine`, called from :meth:`NativeBackend.boot`), so the
module imports on CPU. "In-process" means the *handle* — ``Engine`` still
spawns the scheduler subprocesses (one per TP rank) + the detokenizer; only the
TokenizerManager lives in the calling process. GPU memory layout, CUDA-IPC
weight transfer, and the NCCL env quarantine are therefore unchanged from the
HTTP impl; what disappears is the SRT HTTP server, the health poll, the proxy
whitelist, and per-request JSON serialization.

Loop discipline (the load-bearing invariant): ``Engine.__init__`` creates and
owns ``engine.loop``, and the TokenizerManager's handler task binds to it at
the first await — so EVERY coroutine here must run on that loop, never on a
fresh one (a fresh loop would work once and deadlock on the second call).
:class:`LoopThread` enforces it with a serve/park lifecycle: while generation
is in flight the loop runs in one dedicated thread and any number of caller
threads submit coroutines onto it (``run_coroutine_threadsafe`` — this is what
lets the agentic drain call ``generate`` concurrently from one thread per
trajectory while the scheduler batches the in-flight requests). The
weight/memory verbs require quiesced generation, PARK the loop (stop + join
the thread), then run the Engine's own synchronous wrappers exactly as before
— those wrappers drive ``engine.loop`` themselves and need it idle.

Verb routing: public ``Engine`` methods where the seam signatures match
(memory, NCCL group, distributed update); the two verbs whose seam payloads
arrive pre-serialized (``update_from_tensor``, ``set_lora``) construct the
installed runtime's io_struct request and call the ``tokenizer_manager``
coroutine directly — exactly what the HTTP endpoints do server-side, so the
payloads are version-matched to the runtime by construction (same rationale as
the HTTP impl's io_struct usage).

Deliberate divergence from the HTTP impl: ``generate`` has NO retry loop. The
HTTP 60-retry absorbs transport flakiness that does not exist in-process; an
in-process exception is a real failure and must surface immediately.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import multiprocessing
import os
import threading
import time
from typing import Any, Callable, Coroutine, Dict, List, Optional, Sequence, TypeVar

from unirl.rollout.engine.sglang.backends.base import _filter_server_args_or_raise
from unirl.rollout.engine.sglang.backends.http import parse_generate_response

logger = logging.getLogger(__name__)
T = TypeVar("T")


_GENERATE_PASSTHROUGH = (
    "input_ids",
    "sampling_params",
    "return_logprob",
    "logprob_start_len",
    "image_data",
    "lora_path",
)


def payload_to_generate_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map one ready-to-POST ``/generate`` payload to ``async_generate`` kwargs.

    ``text`` → ``prompt``; the rest pass through by name. An unknown key
    raises: the HTTP path would have forwarded it to the server, so silently
    dropping it here would be an invisible behavioral divergence. The payload
    is not mutated.
    """
    unknown = set(payload) - set(_GENERATE_PASSTHROUGH) - {"text"}
    if unknown:
        raise ValueError(f"sglang native backend: unmapped /generate payload keys: {sorted(unknown)}")
    kwargs = {k: payload[k] for k in _GENERATE_PASSTHROUGH if k in payload}
    if "text" in payload:
        kwargs["prompt"] = payload["text"]
    return kwargs


def _import_sglang_engine() -> Dict[str, Any]:
    """Lazy import of the Engine entrypoint + the io_struct request types.

    Only called from :meth:`NativeBackend.boot`, so the module imports on CPU.
    Only the two io_structs whose verbs bypass the public Engine methods are
    needed (see the module docstring's verb routing).
    """
    from sglang.srt.entrypoints.engine import Engine
    from sglang.srt.managers.io_struct import (
        LoadLoRAAdapterFromTensorsReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import MultiprocessingSerializer

    return {
        "Engine": Engine,
        "ServerArgs": ServerArgs,
        "MultiprocessingSerializer": MultiprocessingSerializer,
        "UpdateWeightsFromTensorReqInput": UpdateWeightsFromTensorReqInput,
        "LoadLoRAAdapterFromTensorsReqInput": LoadLoRAAdapterFromTensorsReqInput,
    }


class LoopThread:
    """Drive one externally-owned event loop: serve generation, park for verbs.

    Serving = the loop runs in one dedicated thread; any number of caller
    threads submit coroutines with :meth:`run` and block on their results
    (``run_coroutine_threadsafe``), so concurrent submissions stay in flight
    together. Parked = the thread is stopped and joined, leaving the loop idle
    for callers that drive it themselves (:meth:`run_parked` — SGLang's sync
    ``Engine`` wrappers ``run_until_complete`` on this very loop).

    One ``threading.Condition`` guards ``{thread, inflight, closed}``. Two
    rules keep it deadlock- and race-free: the loop thread itself NEVER takes
    the condition (nothing submitted here touches this state from the loop),
    and submissions register in ``inflight`` inside the same critical section
    that starts the thread — so :meth:`run_parked`'s ``inflight == 0`` check
    can never miss a submission that already won the lock.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, *, label: str) -> None:
        self._loop = loop
        self._label = label
        self._cond = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._inflight = 0
        self._closed = False

    @property
    def serving(self) -> bool:
        """Whether the loop thread is currently running (unlocked snapshot)."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    def _ensure_serving_locked(self) -> None:
        if self._thread is not None and not self._thread.is_alive():
            self._thread.join()
            self._thread = None
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop.run_forever, name=f"{self._label} loop", daemon=True)
            self._thread.start()

    def _park_locked(self) -> None:
        if self._thread is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._thread = None

    def run(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Submit one coroutine onto the serving loop; block for its result."""
        with self._cond:
            if self._closed:
                coroutine.close()
                raise RuntimeError(f"{self._label} loop thread is closed")
            self._ensure_serving_locked()
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            self._inflight += 1
        try:
            return future.result()
        finally:
            with self._cond:
                self._inflight -= 1
                self._cond.notify_all()

    def run_control(self, coroutine: Coroutine[Any, Any, Any], *, timeout_s: float = 10.0) -> Any:
        """Best-effort control: no-op ``None`` when parked/closed, bounded wait
        when serving. Controls are not counted in ``inflight`` — a park racing a
        pending control freezes it, and the caller eats the bounded timeout."""
        with self._cond:
            if self._closed or self._thread is None or not self._thread.is_alive():
                coroutine.close()
                return None
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        try:
            return future.result(timeout=timeout_s)
        except Exception as exc:  # noqa: BLE001 — best-effort by contract
            future.cancel()
            logger.warning("%s control failed or timed out (best-effort): %s", self._label, exc)
            return None

    def run_parked(self, fn: Callable[[], T]) -> T:
        """Run ``fn`` with the loop parked (idle), holding the lifecycle lock.

        Requires quiesced generation — raises on in-flight submissions instead
        of waiting, because the callers (weight/memory verbs) are only legal
        after the trainer's abort/barrier quiesce. Holding the condition across
        ``fn`` blocks a concurrent :meth:`run` from re-serving mid-verb; it
        proceeds once ``fn`` returns.
        """
        with self._cond:
            if self._closed:
                raise RuntimeError(f"{self._label} loop thread is closed")
            if self._inflight:
                raise RuntimeError(
                    f"{self._label}: verb requires quiesced generation "
                    f"({self._inflight} submissions in flight) — abort/drain first"
                )
            self._park_locked()
            return fn()

    def close(self, *, finalizer: Optional[Callable[[], None]] = None) -> None:
        """Close once: wait for in-flight submissions, park, then finalize."""
        with self._cond:
            if self._closed:
                return
            while self._inflight:
                self._cond.wait()
            self._closed = True
            self._park_locked()
            if finalizer is not None:
                finalizer()


class NativeBackend:
    """The native ``Backend`` impl over an in-process ``sglang.Engine``."""

    def __init__(
        self,
        engine: Any,
        *,
        concurrency: int,
        runtime: Dict[str, Any],
    ) -> None:
        self._engine: Optional[Any] = engine
        self._concurrency = int(concurrency)
        self._rt = runtime
        self._logged_first_response = False
        self._lt = LoopThread(engine.loop, label="sglang NativeBackend")
        self._sem = asyncio.Semaphore(int(concurrency))

    @classmethod
    def boot(
        cls,
        server_intent: Dict[str, Any],
        *,
        concurrency: int,
    ) -> "NativeBackend":
        """Filter intent against ServerArgs, construct the in-process Engine.

        ``server_intent`` is the same config-spelled ServerArgs intent the HTTP
        impl consumes (reserved ports already overlaid — ``nccl_port`` is kept
        deliberately: the colocate de-sync rationale is unchanged, Engine left
        with ``nccl_port=None`` still races get_free_port() at the synchronized
        post-load moment; ``port`` flows through as a harmless unused
        ServerArgs field). ``Engine(**kwargs)`` blocks until the schedulers are
        up and the model is loaded — no health poll, no timeout knob.
        """
        rt = _import_sglang_engine()

        allowed = {f.name for f in dataclasses.fields(rt["ServerArgs"])}
        engine_kwargs = _filter_server_args_or_raise(
            server_intent,
            allowed=allowed,
            backend_name="native",
        )
        tp_size = int(engine_kwargs.get("tp_size", 1) or 1)
        if tp_size > 1:
            raise NotImplementedError(
                "SGLang native backend does not support rollout tp_size>1 in UniRL yet; "
                "use backend='http' for rollout TP/EP."
            )
        engine_kwargs.setdefault("log_level", "info")

        try:
            import torch

            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        except Exception:
            pass

        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")

        logger.info(
            "Constructing in-process SGLang Engine: model=%s tp=%s nccl_port=%s",
            engine_kwargs.get("model_path"),
            engine_kwargs.get("tp_size"),
            engine_kwargs.get("nccl_port"),
        )

        multiprocessing.set_start_method("spawn", force=True)
        engine = rt["Engine"](**engine_kwargs)

        settled = getattr(engine, "server_args", None)
        logger.info(
            "SGLang Engine ready (settled ServerArgs: port=%s nccl_port=%s)",
            getattr(settled, "port", None),
            getattr(settled, "nccl_port", None),
        )
        return cls(engine, concurrency=concurrency, runtime=rt)

    def generate(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Generate the payloads on engine.loop; flatten prompt-major.

        Safe for concurrent callers: each call submits onto the serving loop
        and blocks for its own result, so N trajectory threads keep N requests
        in flight together. A length-1 wire (the agentic per-turn path) skips
        the gather and the per-batch INFO log.
        """
        self._require_alive("generate")
        if len(requests) == 1:
            return self._lt.run(self._agen_one(requests[0]))
        t0 = time.perf_counter()
        results = self._lt.run(self._agen_many(requests))
        elapsed = time.perf_counter() - t0
        logger.info(
            "sglang NativeBackend.generate: %d requests -> %d results in %.2fs",
            len(requests),
            len(results),
            elapsed,
        )
        return results

    async def _agen_one(self, payload: Dict[str, Any]) -> List[Any]:
        """Generate ONE ``/generate`` payload on engine.loop, bounded by the
        shared semaphore. The per-request unit concurrent callers submit."""
        kwargs = payload_to_generate_kwargs(payload)
        async with self._sem:
            try:
                response = await self._engine.async_generate(**kwargs)
            except Exception as exc:
                raise RuntimeError(f"sglang NativeBackend.generate failed: {exc}") from exc
        parsed = parse_generate_response(response)
        if not self._logged_first_response and parsed:
            self._logged_first_response = True
            first = parsed[0]
            logger.info(
                "sglang first response: token_ids=%d logprobs=%d raw_text[:200]=%r",
                len(first.token_ids),
                len(first.logprobs),
                first.text[:200],
            )
        return parsed

    async def _agen_many(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Fan payloads out concurrently; flatten prompt-major.

        ``return_exceptions=True`` is load-bearing: the outer submission must
        stay in flight until EVERY sibling settles, else ``run_parked``'s
        "no in-flight submissions ⇒ loop quiescent" invariant breaks and a
        park could freeze a still-running ``async_generate`` mid-decode. The
        first failure re-raises after the siblings settle.
        """
        nested = await asyncio.gather(*(self._agen_one(p) for p in requests), return_exceptions=True)
        for item in nested:
            if isinstance(item, BaseException):
                raise item
        return [item for sublist in nested for item in sublist]

    def abort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None:
        self._lt.run_control(self._aabort(abort_all=abort_all, rid=rid))

    def pause(self) -> None:
        self._lt.run_control(self._apause())

    def resume(self) -> None:
        self._lt.run_control(self._aresume())

    async def _aabort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None:
        """Abort in-flight generation (best-effort across sglang versions)."""
        tm = getattr(self._engine, "tokenizer_manager", None)
        fn = getattr(tm, "abort_request", None)
        if fn is None:
            logger.warning("sglang NativeBackend: tokenizer_manager.abort_request unavailable; abort is a no-op")
            return
        try:
            res = fn(rid=rid, abort_all=abort_all) if rid is not None else fn(abort_all=abort_all)
        except TypeError:
            res = fn(rid) if rid is not None else fn()
        if asyncio.iscoroutine(res):
            await res

    async def _apause(self) -> None:
        """Pause generation admission if the runtime supports it (best-effort)."""
        await self._call_optional("pause_generation")

    async def _aresume(self) -> None:
        """Resume generation admission if the runtime supports it (best-effort)."""
        await self._call_optional("continue_generation")

    async def _call_optional(self, name: str) -> None:
        tm = getattr(self._engine, "tokenizer_manager", None)
        fn = getattr(tm, name, None) or getattr(self._engine, name, None)
        if fn is None:
            return
        res = fn()
        if asyncio.iscoroutine(res):
            await res

    @staticmethod
    def _check_result(result: Any, operation: str) -> None:
        """Raise on failure; absent success means ok (HTTP-checker parity).

        Absorbs the three native result shapes: ``(success, message)`` tuples
        from the tokenizer_manager coroutines, plain dicts, and io_struct
        ReqOutput objects with a ``success`` attribute.
        """
        success, detail = True, "unknown"
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            success, detail = bool(result[0]), result[1]
        elif isinstance(result, dict):
            success = result.get("success", True)
            detail = result.get("error_message") or result.get("message", "unknown")
        elif hasattr(result, "success"):
            success = bool(result.success)
            detail = getattr(result, "error_message", None) or getattr(result, "message", "unknown")
        if not success:
            raise RuntimeError(f"sglang NativeBackend.{operation} failed: {detail}")

    def _require_alive(self, operation: str) -> None:
        if self._engine is None:
            raise RuntimeError(f"Cannot {operation}: native sglang engine is shut down.")

    def flush_cache(self) -> None:
        """Flush the sglang scheduler cache; retry until it succeeds.

        Mirrors the HTTP impl: the scheduler reports failure while pending
        requests exist (the condition that made /flush_cache return non-200);
        retry up to 60 × 1s. Precondition for sleep so release actually frees
        the KV pool.
        """
        self._require_alive("flush cache")

        def _flush() -> None:
            last: Any = None
            for _ in range(60):
                last = self._engine.flush_cache()
                if getattr(last, "success", True):
                    return
                time.sleep(1.0)
            raise TimeoutError(
                f"sglang NativeBackend: flush_cache did not succeed after 60 attempts (last result: {last})"
            )

        self._lt.run_parked(_flush)

    def release_memory(self, *, tags: Optional[Sequence[str]] = None) -> None:
        self._require_alive("release memory")
        result = self._lt.run_parked(
            lambda: self._engine.release_memory_occupation(tags=list(tags) if tags is not None else None)
        )
        self._check_result(result, "release_memory")

    def resume_memory(self, *, tags: Optional[Sequence[str]] = None) -> None:
        self._require_alive("resume memory")
        result = self._lt.run_parked(
            lambda: self._engine.resume_memory_occupation(tags=list(tags) if tags is not None else None)
        )
        self._check_result(result, "resume_memory")

    def ping(self) -> bool:
        """Liveness of the Engine's child processes (schedulers + detokenizer).

        Weaker than the HTTP impl's /health_generate (existence probe, not a
        generation probe) — acceptable because health_check() short-circuits
        while offloaded and a wedged-but-alive scheduler surfaces in generate.
        """
        if self._engine is None:
            return False
        try:
            pids = self._engine.get_all_child_pids()
            for pid in pids:
                os.kill(pid, 0)
            return bool(pids)
        except Exception:
            return False

    def shutdown(self) -> None:
        """Shut the Engine down once; tolerate the re-entrant callers.

        Engine registers its own atexit shutdown and the rollout engine's
        ``__del__`` re-enters ours — the None-swap makes our side idempotent.
        ``close`` waits for in-flight generation to settle before parking, so
        teardown stays graceful.
        """
        engine = self._engine
        if engine is None:
            return

        def _shutdown() -> None:
            self._engine = None
            logger.info("Shutting down in-process SGLang Engine")
            engine.shutdown()

        self._lt.close(finalizer=_shutdown)

    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None:
        """Update weights from pre-serialized per-TP-rank tensor bags.

        The public ``Engine.update_weights_from_tensor`` takes RAW tensors and
        re-serializes — our seam carries the bags already serialized, so this
        constructs the io_struct and calls the tokenizer_manager coroutine
        directly (exactly what the HTTP endpoint does server-side).
        """
        self._require_alive("update_from_tensor")
        obj = self._rt["UpdateWeightsFromTensorReqInput"](
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        engine = self._engine
        result = self._lt.run_parked(
            lambda: engine.loop.run_until_complete(engine.tokenizer_manager.update_weights_from_tensor(obj, None))
        )
        self._check_result(result, "update_from_tensor")

    def init_weights_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
    ) -> None:
        self._require_alive("init_weights_group")
        result = self._lt.run_parked(
            lambda: self._engine.init_weights_update_group(
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=int(rank_offset),
                world_size=int(world_size),
                group_name=str(group_name),
                backend=str(backend),
            )
        )
        self._check_result(result, "init_weights_group")
        logger.info(
            "sglang NativeBackend: NCCL group %r initialized (rank_offset=%d, world_size=%d)",
            group_name,
            rank_offset,
            world_size,
        )

    def update_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        flush_cache: bool,
    ) -> None:
        logger.info(
            "sglang NativeBackend: update_weights_from_distributed group=%s, %d params, first=%s last=%s, flush=%s",
            group_name,
            len(names),
            names[0] if names else "<empty>",
            names[-1] if names else "<empty>",
            flush_cache,
        )
        self._require_alive("update_from_distributed")
        result = self._lt.run_parked(
            lambda: self._engine.update_weights_from_distributed(
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(s) for s in shapes],
                group_name=str(group_name),
                flush_cache=flush_cache,
            )
        )
        self._check_result(result, "update_from_distributed")

    def destroy_weights_group(self, *, group_name: str) -> None:
        self._require_alive("destroy_weights_group")
        result = self._lt.run_parked(lambda: self._engine.destroy_weights_update_group(group_name=str(group_name)))
        self._check_result(result, "destroy_weights_group")

    def set_lora(
        self,
        *,
        lora_name: str,
        lora_tensors: Dict[str, Any],
        config_dict: Optional[dict] = None,
    ) -> None:
        """Serialize the LoRA tensor bag and hot-load it on the Engine.

        Parity with the HTTP impl: same MultiprocessingSerializer call, same
        io_struct — delivered to the tokenizer_manager coroutine instead of
        POSTed (the /load_lora_adapter_from_tensors endpoint does exactly
        this server-side).
        """
        self._require_alive("set_lora")
        serialized = self._rt["MultiprocessingSerializer"].serialize(lora_tensors, output_str=True)
        obj = self._rt["LoadLoRAAdapterFromTensorsReqInput"](
            lora_name=str(lora_name),
            config_dict=dict(config_dict or {}),
            serialized_tensors=serialized,
        )
        engine = self._engine
        result = self._lt.run_parked(
            lambda: engine.loop.run_until_complete(engine.tokenizer_manager.load_lora_adapter_from_tensors(obj, None))
        )
        self._check_result(result, "set_lora")


__all__ = ["LoopThread", "NativeBackend", "payload_to_generate_kwargs"]
