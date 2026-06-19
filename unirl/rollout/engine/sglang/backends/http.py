"""The HTTP ``Backend`` impl — SGLang SRT server subprocess + HTTP client.

The ONLY module that imports the SGLang runtime or does I/O — including the
spawn. :meth:`HTTPBackend.boot` filters the config-spelled intent against the
real ``ServerArgs`` fields (the only place that knows them), quarantines the env
the SRT subprocess needs at the spawn boundary, launches the server, and polls
``/health_generate``. Generation fans the per-prompt payloads out concurrently
(persistent event loop + semaphore + retry — ``slime``-style HTTP plumbing); weight/memory
verbs are synchronous POSTs with
the long weight-op timeout tier.

Control-plane payloads (weight sync, memory, LoRA) are constructed from the
installed runtime's own ``io_struct`` request dataclasses rather than hand-built
dicts: the actor and the SRT subprocess share one install, so the payloads are
version-matched to the server by construction — a field-name drift fails loudly
at construction instead of as an opaque HTTP 422 mid-training.

Because the SGLang import is lazy (only :func:`_import_sglang_runtime`, called
from :meth:`boot`), the module imports on CPU — the rest of the package is
exercisable without a GPU, and :func:`parse_generate_response` (the
``/generate`` JSON → :class:`RawResult` deserialization) is a pure module-level
function.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import multiprocessing
import os
import signal
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only when httpx is missing
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Process / health helpers (SRT subprocess lifecycle + health polling)
# ---------------------------------------------------------------------------


def kill_process_tree(pid: int) -> None:
    """Send SIGTERM to ``pid`` and its descendants."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def wait_server_healthy(
    base_url: str,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 2.0,
    is_alive_fn: Optional[Callable[[], bool]] = None,
) -> None:
    """Poll server ``/health_generate`` until 200 OK or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base_url}/health_generate", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        if is_alive_fn is not None and not is_alive_fn():
            raise RuntimeError("SGLang SRT server process terminated unexpectedly.")
        time.sleep(poll_interval_s)
    raise TimeoutError(f"SGLang SRT server at {base_url} did not become healthy within {timeout_s}s")


# ---------------------------------------------------------------------------
# Lazy runtime import — the only place sglang is named (once per process)
# ---------------------------------------------------------------------------


def _import_sglang_runtime() -> Dict[str, Any]:
    """Lazy import of the server entrypoints + the io_struct request types.

    Only called from :meth:`HTTPBackend.boot`, so the module imports on CPU.
    The verbs construct these installed-runtime request dataclasses instead of
    hand-built dicts (see the module docstring for why).
    """
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.managers.io_struct import (
        DestroyWeightsUpdateGroupReqInput,
        InitWeightsUpdateGroupReqInput,
        LoadLoRAAdapterFromTensorsReqInput,
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
        UpdateWeightsFromDistributedReqInput,
        UpdateWeightsFromTensorReqInput,
    )
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.utils import MultiprocessingSerializer

    return {
        "launch_server": launch_server,
        "ServerArgs": ServerArgs,
        "MultiprocessingSerializer": MultiprocessingSerializer,
        "UpdateWeightsFromTensorReqInput": UpdateWeightsFromTensorReqInput,
        "UpdateWeightsFromDistributedReqInput": UpdateWeightsFromDistributedReqInput,
        "InitWeightsUpdateGroupReqInput": InitWeightsUpdateGroupReqInput,
        "DestroyWeightsUpdateGroupReqInput": DestroyWeightsUpdateGroupReqInput,
        "LoadLoRAAdapterFromTensorsReqInput": LoadLoRAAdapterFromTensorsReqInput,
        "ReleaseMemoryOccupationReqInput": ReleaseMemoryOccupationReqInput,
        "ResumeMemoryOccupationReqInput": ResumeMemoryOccupationReqInput,
    }


def asdict_drop_none(req: Any) -> Dict[str, Any]:
    """The wire view of an io_struct request: its fields minus the ``None``s.

    Unset Optionals (incl. ``BaseReq``'s ``rid`` / ``http_worker_ipc``) drop;
    ``False`` / ``0`` / empty containers survive (``flush_cache=False`` must
    reach the server).
    """
    return {k: v for k, v in dataclasses.asdict(req).items() if v is not None}


# ---------------------------------------------------------------------------
# Wire deserialization — pure, module-level (CPU-testable without sglang)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _HTTPRawResult:
    """The HTTP impl's :class:`~.base.RawResult` — one parsed candidate."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str


def parse_generate_response(response: Any) -> List[_HTTPRawResult]:
    """Parse one SRT ``/generate`` response into per-candidate results.

    SGLang returns a single dict for ``n=1`` and a list of dicts for ``n>1``;
    both normalize to a list here (the per-prompt candidate order is SRT's).
    Token ids and log-probs both ride the ``output_token_logprobs``
    ``(logprob, token_id[, token_text])`` items — the runtime's only source of
    generated token ids — so the two lists are length-aligned by construction;
    ``finish_reason`` arrives as a dict or a bare string.
    """
    if isinstance(response, list):
        candidates = response
    elif isinstance(response, dict):
        candidates = [response]
    else:
        raise RuntimeError(f"Unexpected sglang response type: {type(response)}")

    results: List[_HTTPRawResult] = []
    for candidate in candidates:
        meta = candidate.get("meta_info", {})

        token_logprobs: List[float] = []
        output_token_ids: List[int] = []
        for item in meta.get("output_token_logprobs", []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                token_logprobs.append(float(item[0]))
                output_token_ids.append(int(item[1]))

        raw_finish = meta.get("finish_reason", "unknown")
        if isinstance(raw_finish, dict):
            finish_reason = str(raw_finish.get("type", "unknown"))
        else:
            finish_reason = str(raw_finish)

        raw_text = str(candidate.get("text", ""))

        results.append(
            _HTTPRawResult(
                text=raw_text,
                token_ids=output_token_ids,
                logprobs=token_logprobs,
                finish_reason=finish_reason,
            )
        )

    return results


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class HTTPBackend:
    """The HTTP ``Backend`` impl over a spawned SGLang SRT server."""

    def __init__(
        self,
        server_process: multiprocessing.Process,
        base_url: str,
        *,
        concurrency: int,
        runtime: Dict[str, Any],
    ) -> None:
        self._server_process: Optional[multiprocessing.Process] = server_process
        self._base_url = base_url
        self._concurrency = int(concurrency)
        self._rt = runtime
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._client: Any = None
        if httpx is not None:
            self._async_loop = asyncio.new_event_loop()
            self._client = self._run_async(self._make_client())
        self._logged_first_response = False

    # ------------------------------------------------------------------ #
    # Boot — the only place the sglang import / spawn / env quarantine live
    # ------------------------------------------------------------------ #

    @classmethod
    def boot(
        cls,
        server_intent: Dict[str, Any],
        *,
        advertise_host: str,
        concurrency: int,
        health_timeout_s: float = 300.0,
    ) -> "HTTPBackend":
        """Filter intent against ServerArgs, spawn the SRT server, await health.

        ``server_intent`` is the config-spelled ServerArgs intent (reserved
        ports already overlaid as ``port`` / ``nccl_port`` — real ServerArgs
        fields, so no port env manipulation happens anywhere). We filter it to
        the real ServerArgs fields here (the only place that knows them —
        non-ServerArgs escape-hatch keys drop harmlessly), then spawn.
        """
        rt = _import_sglang_runtime()

        allowed = {f.name for f in dataclasses.fields(rt["ServerArgs"])}
        server_kwargs = {k: v for k, v in server_intent.items() if k in allowed}

        # --- Env quarantine: everything the SRT subprocess needs, set at the
        # spawn boundary (the spec's documented last resort) — never in the
        # engine ctor. Each line carries the predecessor's rationale.

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

        # SGLang's warmup self-check issues requests.get(...) against
        # http://{host}:{port}/model_info which honors HTTP(S)_PROXY env vars
        # and routes loopback through Squid (returns 503, kills SRT). Whitelist
        # the bind + advertise + loopback hosts.
        _extra_no_proxy = f"0.0.0.0,127.0.0.1,localhost,{advertise_host}"
        _cur_no_proxy = os.environ.get("no_proxy", "") or os.environ.get("NO_PROXY", "")
        os.environ["no_proxy"] = f"{_cur_no_proxy},{_extra_no_proxy}" if _cur_no_proxy else _extra_no_proxy
        os.environ["NO_PROXY"] = os.environ["no_proxy"]

        logger.info(
            "Launching SGLang SRT server: model=%s tp=%s port=%s nccl_port=%s",
            server_kwargs.get("model_path"),
            server_kwargs.get("tp_size"),
            server_kwargs.get("port"),
            server_kwargs.get("nccl_port"),
        )

        # ``set_start_method`` is process-global; PE-tested, Ray-compatible.
        # Forcing matches the predecessor so torch CUDA init in the child
        # happens cleanly.
        multiprocessing.set_start_method("spawn", force=True)
        server_args = rt["ServerArgs"](**server_kwargs)
        process = multiprocessing.Process(target=rt["launch_server"], args=(server_args,))
        process.start()

        base_url = f"http://{advertise_host}:{server_kwargs['port']}"
        wait_server_healthy(
            base_url,
            timeout_s=float(health_timeout_s),
            is_alive_fn=lambda: process.is_alive(),
        )
        # Bind-mapping gate (GPU smoke): the settled ServerArgs must echo the
        # reserved ports verbatim — a runtime upgrade that silently re-settles
        # them shows up here.
        logger.info(
            "SGLang SRT server healthy at %s (settled ServerArgs: port=%s nccl_port=%s host=%s)",
            base_url,
            getattr(server_args, "port", None),
            getattr(server_args, "nccl_port", None),
            getattr(server_args, "host", None),
        )
        return cls(process, base_url, concurrency=concurrency, runtime=rt)

    # ------------------------------------------------------------------ #
    # Generation — async fan-out owned here (event loop, semaphore, retry)
    # ------------------------------------------------------------------ #

    async def _make_client(self) -> Any:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(None),
            trust_env=False,
        )

    def _run_async(self, awaitable: Any) -> Any:
        if self._async_loop is None or self._async_loop.is_closed():
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise RuntimeError("sglang HTTPBackend async event loop is not available.")
        return self._async_loop.run_until_complete(awaitable)

    def generate(self, requests: List[Dict[str, Any]]) -> List[_HTTPRawResult]:
        """POST the per-prompt payloads concurrently; flatten prompt-major."""
        if self._client is None:
            raise RuntimeError("httpx is required for sglang generate. Install httpx: pip install httpx")
        t0 = time.perf_counter()
        results = self._run_async(self._generate_async(requests))
        elapsed = time.perf_counter() - t0
        logger.info(
            "sglang HTTPBackend.generate: %d requests -> %d results in %.2fs",
            len(requests),
            len(results),
            elapsed,
        )
        return results

    async def _generate_async(self, requests: List[Dict[str, Any]]) -> List[_HTTPRawResult]:
        sem = asyncio.Semaphore(self._concurrency)

        async def _generate_one(payload: Dict[str, Any]) -> List[_HTTPRawResult]:
            async with sem:
                response = await self._apost("/generate", payload)
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

    async def _apost(
        self,
        path: str,
        payload: Dict[str, Any],
        max_retries: int = 60,
    ) -> Any:
        """Async POST with retry. Mirrors slime/utils/http_utils.py:165-198."""
        url = f"{self._base_url}{path}"
        for attempt in range(max_retries):
            response = None
            try:
                response = await self._client.post(url, json=payload)
                response.raise_for_status()
                content = await response.aread()
                return json.loads(content) if content else {}
            except Exception as exc:
                if attempt >= max_retries - 1:
                    error_detail = ""
                    if response is not None:
                        try:
                            error_detail = response.text[:500]
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"SGLang SRT POST {url} failed after {max_retries} retries: {exc} | response={error_detail}"
                    ) from exc
                logger.debug(
                    "SGLang SRT POST %s attempt %d/%d failed: %s",
                    url,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                await asyncio.sleep(1)
            finally:
                if response is not None:
                    await response.aclose()
        return {}  # unreachable

    # ------------------------------------------------------------------ #
    # Sync HTTP for non-generation endpoints (weight sync, memory)
    # ------------------------------------------------------------------ #

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        """Synchronous POST JSON to the SRT server."""
        url = f"{self._base_url}{path}"
        # Weight-update + LoRA hot-reload endpoints can stall server-side
        # (NCCL init / broadcast, or SGLang's LoRA-pool unload+reload which
        # takes ~2 min from the 2nd sync on — LIN-287). Give them the long
        # timeout so a legitimately-slow-but-succeeding op isn't killed at 120s.
        timeout = 600 if ("weights" in path or "update" in path or "lora" in path) else 120
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            error_body = ""
            try:
                error_body = exc.read().decode("utf-8")[:1000]
            except Exception:
                pass
            raise RuntimeError(f"SGLang SRT HTTP {exc.code} for {url}: {error_body}") from exc

    def _post_struct(self, path: str, req: Any, operation: str) -> None:
        """POST a typed io_struct request (its non-``None`` fields) and check."""
        resp = self._post(path, asdict_drop_none(req))
        self._check_update_response(resp, operation)

    @staticmethod
    def _check_update_response(response: Any, operation: str) -> None:
        if isinstance(response, dict):
            if not response.get("success", True):
                detail = response.get("error_message") or response.get("message", "unknown")
                raise RuntimeError(f"sglang HTTPBackend.{operation} failed: {detail}")

    def _require_alive(self, operation: str) -> None:
        if self._server_process is None or not self._server_process.is_alive():
            raise RuntimeError(f"Cannot {operation}: SRT server is not alive.")

    # ------------------------------------------------------------------ #
    # Memory / lifecycle / health
    # ------------------------------------------------------------------ #

    def flush_cache(self) -> None:
        """Flush the sglang scheduler cache; retry until 200.

        Mirrors slime's flush_cache: /flush_cache returns non-200 while
        pending requests exist; retry up to 60 × 1s. Precondition for
        sleep so /release_memory_occupation actually frees the KV pool.
        """
        url = f"{self._base_url}/flush_cache"
        last_err: Optional[Exception] = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    if resp.status == 200:
                        return
            except urllib.error.HTTPError as exc:
                last_err = exc
            except Exception as exc:
                last_err = exc
            time.sleep(1.0)
        raise TimeoutError(
            f"sglang HTTPBackend: /flush_cache did not return 200 after 60 attempts (last error: {last_err})"
        )

    def release_memory(self, *, tags: Optional[Sequence[str]] = None) -> None:
        self._require_alive("release memory")
        self._post_struct(
            "/release_memory_occupation",
            self._rt["ReleaseMemoryOccupationReqInput"](tags=list(tags) if tags is not None else None),
            "release_memory",
        )

    def resume_memory(self, *, tags: Optional[Sequence[str]] = None) -> None:
        self._require_alive("resume memory")
        self._post_struct(
            "/resume_memory_occupation",
            self._rt["ResumeMemoryOccupationReqInput"](tags=list(tags) if tags is not None else None),
            "resume_memory",
        )

    def ping(self) -> bool:
        if self._server_process is None or not self._server_process.is_alive():
            return False
        try:
            wait_server_healthy(self._base_url, timeout_s=5, poll_interval_s=1)
            return True
        except (TimeoutError, RuntimeError):
            return False

    def shutdown(self) -> None:
        """Kill the SRT server and close the HTTP client."""
        if self._client is not None:
            try:
                self._run_async(self._client.aclose())
            except Exception:
                pass
            self._client = None
        if self._async_loop is not None:
            try:
                if not self._async_loop.is_closed():
                    self._async_loop.close()
            except Exception:
                pass
            self._async_loop = None
        if self._server_process is not None:
            logger.info("Shutting down SGLang SRT server (pid=%s)", self._server_process.pid)
            kill_process_tree(self._server_process.pid)
            self._server_process.join(timeout=10)
            self._server_process = None

    # ------------------------------------------------------------------ #
    # Weight-sync verbs — HTTP POSTs to the SRT post-training endpoints
    # ------------------------------------------------------------------ #

    def update_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str],
        flush_cache: bool,
    ) -> None:
        self._post_struct(
            "/update_weights_from_tensor",
            self._rt["UpdateWeightsFromTensorReqInput"](
                serialized_named_tensors=serialized_named_tensors,
                load_format=load_format,
                flush_cache=flush_cache,
            ),
            "update_from_tensor",
        )

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
        self._post_struct(
            "/init_weights_update_group",
            self._rt["InitWeightsUpdateGroupReqInput"](
                master_address=master_address,
                master_port=int(master_port),
                rank_offset=int(rank_offset),
                world_size=int(world_size),
                group_name=str(group_name),
                backend=str(backend),
            ),
            "init_weights_group",
        )
        logger.info(
            "sglang HTTPBackend: NCCL group %r initialized (rank_offset=%d, world_size=%d)",
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
            "sglang HTTPBackend: update_weights_from_distributed group=%s, %d params, first=%s last=%s, flush=%s",
            group_name,
            len(names),
            names[0] if names else "<empty>",
            names[-1] if names else "<empty>",
            flush_cache,
        )
        self._post_struct(
            "/update_weights_from_distributed",
            self._rt["UpdateWeightsFromDistributedReqInput"](
                names=list(names),
                dtypes=list(dtypes),
                shapes=[list(s) for s in shapes],
                group_name=str(group_name),
                flush_cache=flush_cache,
            ),
            "update_from_distributed",
        )

    def destroy_weights_group(self, *, group_name: str) -> None:
        self._post_struct(
            "/destroy_weights_update_group",
            self._rt["DestroyWeightsUpdateGroupReqInput"](group_name=str(group_name)),
            "destroy_weights_group",
        )

    def set_lora(
        self,
        *,
        lora_name: str,
        lora_tensors: Dict[str, Any],
        config_dict: Optional[dict] = None,
    ) -> None:
        """Serialize the LoRA tensor bag and hot-load it on the SRT server.

        SGLang SRT exposes ``POST /load_lora_adapter_from_tensors`` which
        accepts serialized LoRA tensors + a PEFT config dict and hot-loads the
        adapter on all TP workers internally.
        """
        serialized = self._rt["MultiprocessingSerializer"].serialize(lora_tensors, output_str=True)
        self._post_struct(
            "/load_lora_adapter_from_tensors",
            self._rt["LoadLoRAAdapterFromTensorsReqInput"](
                lora_name=str(lora_name),
                config_dict=dict(config_dict or {}),
                serialized_tensors=serialized,
            ),
            "set_lora",
        )


__all__ = [
    "HTTPBackend",
    "asdict_drop_none",
    "kill_process_tree",
    "parse_generate_response",
    "wait_server_healthy",
]
