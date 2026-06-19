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
the first await — so EVERY coroutine here must run on that loop (via
:meth:`NativeBackend._run`), never on a fresh one (a fresh loop would work once
and deadlock on the second call). Engine's own sync wrappers already run on
``self.loop``. Corollary: all verbs must be invoked from one thread — the Ray
actor's single RPC thread today (``max_concurrency=1``); a concurrent
``run_until_complete`` on the same loop raises.

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
import time
from typing import Any, Dict, List, Optional, Sequence

from unirl.rollout.engine.sglang.backends.http import parse_generate_response

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wire mapping — pure, module-level (CPU-testable without sglang)
# ---------------------------------------------------------------------------

#: /generate payload keys that map 1:1 onto ``Engine.async_generate`` kwargs.
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


# ---------------------------------------------------------------------------
# Lazy runtime import — the only place sglang is named (once per process)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


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

    # ------------------------------------------------------------------ #
    # Boot — the only place the sglang import / spawn / env quarantine live
    # ------------------------------------------------------------------ #

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
        engine_kwargs = {k: v for k, v in server_intent.items() if k in allowed}
        # The Engine entrypoint defaults log_level to "error" (the HTTP
        # server path runs at "info") — restore parity so scheduler logs and
        # this module's post-init lines stay visible. Intent overrides win.
        engine_kwargs.setdefault("log_level", "info")

        # --- Env quarantine: the HTTP impl's block minus the no_proxy
        # whitelist (no HTTP warmup self-check to misroute). The rest is
        # unchanged because the schedulers are still subprocesses.

        # CUDA-IPC tensor sync requires the non-expandable allocator on older
        # kernels (<5.10) that lack pidfd_getfd; matches PE rollout_actor.py.
        try:
            import torch

            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        except Exception:
            pass

        # NCCL transport defaults — required for cross-process NCCL groups
        # used by weight sync to establish P2P/CUMEM channels. sglang's
        # _set_envs_and_config() defaults these to "0" when enable_symm_mem
        # is False, breaking broadcast with "Cuda failure 'invalid argument'".
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")

        logger.info(
            "Constructing in-process SGLang Engine: model=%s tp=%s nccl_port=%s",
            engine_kwargs.get("model_path"),
            engine_kwargs.get("tp_size"),
            engine_kwargs.get("nccl_port"),
        )

        # ``set_start_method`` is process-global; matches the HTTP impl so
        # torch CUDA init in the scheduler children happens cleanly.
        multiprocessing.set_start_method("spawn", force=True)
        engine = rt["Engine"](**engine_kwargs)

        # Bind-mapping gate twin: the settled ServerArgs must echo the
        # reserved ports verbatim — a runtime upgrade that silently re-settles
        # them shows up here.
        settled = getattr(engine, "server_args", None)
        logger.info(
            "SGLang Engine ready (settled ServerArgs: port=%s nccl_port=%s)",
            getattr(settled, "port", None),
            getattr(settled, "nccl_port", None),
        )
        return cls(engine, concurrency=concurrency, runtime=rt)

    # ------------------------------------------------------------------ #
    # The single loop seam — every coroutine runs on engine.loop
    # ------------------------------------------------------------------ #

    def _run(self, coro: Any) -> Any:
        return self._engine.loop.run_until_complete(coro)

    # ------------------------------------------------------------------ #
    # Generation — per-payload fan-out on engine.loop (no retry; see module
    # docstring)
    # ------------------------------------------------------------------ #

    def generate(self, requests: List[Dict[str, Any]]) -> List[Any]:
        """Fan the per-prompt payloads out on engine.loop; flatten prompt-major."""
        self._require_alive("generate")
        t0 = time.perf_counter()
        results = self._run(self._generate_async(requests))
        elapsed = time.perf_counter() - t0
        logger.info(
            "sglang NativeBackend.generate: %d requests -> %d results in %.2fs",
            len(requests),
            len(results),
            elapsed,
        )
        return results

    async def _generate_async(self, requests: List[Dict[str, Any]]) -> List[Any]:
        sem = asyncio.Semaphore(self._concurrency)

        async def _generate_one(payload: Dict[str, Any]) -> List[Any]:
            kwargs = payload_to_generate_kwargs(payload)
            async with sem:
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

        nested = await asyncio.gather(*(_generate_one(p) for p in requests))
        return [item for sublist in nested for item in sublist]

    # ------------------------------------------------------------------ #
    # Result normalization — the native twin of _check_update_response
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Memory / lifecycle / health
    # ------------------------------------------------------------------ #

    def flush_cache(self) -> None:
        """Flush the sglang scheduler cache; retry until it succeeds.

        Mirrors the HTTP impl: the scheduler reports failure while pending
        requests exist (the condition that made /flush_cache return non-200);
        retry up to 60 × 1s. Precondition for sleep so release actually frees
        the KV pool.
        """
        self._require_alive("flush cache")
        last: Any = None
        for _ in range(60):
            last = self._engine.flush_cache()
            if getattr(last, "success", True):
                return
            time.sleep(1.0)
        raise TimeoutError(f"sglang NativeBackend: flush_cache did not succeed after 60 attempts (last result: {last})")

    def release_memory(self, *, tags: Optional[Sequence[str]] = None) -> None:
        self._require_alive("release memory")
        result = self._engine.release_memory_occupation(tags=list(tags) if tags is not None else None)
        self._check_result(result, "release_memory")

    def resume_memory(self, *, tags: Optional[Sequence[str]] = None) -> None:
        self._require_alive("resume memory")
        result = self._engine.resume_memory_occupation(tags=list(tags) if tags is not None else None)
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
            # An empty pid list after a successful boot means the children
            # are gone.
            return bool(pids)
        except Exception:
            return False

    def shutdown(self) -> None:
        """Shut the Engine down once; tolerate the re-entrant callers.

        Engine registers its own atexit shutdown and the rollout engine's
        ``__del__`` re-enters ours — the None-swap makes our side idempotent.
        """
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        logger.info("Shutting down in-process SGLang Engine")
        engine.shutdown()

    # ------------------------------------------------------------------ #
    # Weight-sync verbs — public Engine methods where signatures match;
    # io_struct + tokenizer_manager for the pre-serialized payloads
    # ------------------------------------------------------------------ #

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
        result = self._run(self._engine.tokenizer_manager.update_weights_from_tensor(obj, None))
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
        result = self._engine.init_weights_update_group(
            master_address=master_address,
            master_port=int(master_port),
            rank_offset=int(rank_offset),
            world_size=int(world_size),
            group_name=str(group_name),
            backend=str(backend),
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
        result = self._engine.update_weights_from_distributed(
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(s) for s in shapes],
            group_name=str(group_name),
            flush_cache=flush_cache,
        )
        self._check_result(result, "update_from_distributed")

    def destroy_weights_group(self, *, group_name: str) -> None:
        self._require_alive("destroy_weights_group")
        result = self._engine.destroy_weights_update_group(group_name=str(group_name))
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
        result = self._run(self._engine.tokenizer_manager.load_lora_adapter_from_tensors(obj, None))
        self._check_result(result, "set_lora")


__all__ = ["NativeBackend", "payload_to_generate_kwargs"]
