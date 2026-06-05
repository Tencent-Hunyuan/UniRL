"""SGLang SRT LLM rollout engine (new ``BaseRolloutEngine`` protocol).

One-shot construction: the ctor launches the sglang SRT HTTP server subprocess,
waits for ``/health_generate`` to return 200, and lazy-initializes an
``httpx.AsyncClient`` + tokenizer for prompt encoding. After ``__init__`` returns
the engine is fully usable: no separate ``initialize(device)`` step.

Speaks the typed ``RolloutReq`` / ``RolloutResp`` contract:

- Reads prompts from ``req.primitives['text'].texts`` (typed :class:`Texts`).
- Reads sampling overrides from ``req.stage_params.get('ar', {})`` (typed
  bag-of-options, falls back to config defaults).
- Emits ``resp.tracks['ar'].decoded: Texts`` (generated text, one per prompt × n).
- Emits ``resp.tracks['ar'].segment: TextSegment`` (packed varlen token ids
  and per-token log-probs); segment rows are 1:1 with samples in
  prompt-major order.
- Echoes ``sample_ids`` / ``group_ids`` from the request; for ``n > 1`` the
  sample-id is mangled as ``f"{sid}#{k}"`` to keep uniqueness while group
  membership stays intact.

Weight sync (NCCL + HTTP-tensor-bag) and memory management (sleep/wake via
sglang SRT's ``/release_memory_occupation`` + ``/resume_memory_occupation``)
mirror the legacy PE engine at
``unirl-pe/unirl/samplers/sglang_llm/engine.py``.

What this engine intentionally does NOT do (vs the legacy PE engine):

- No ``initialize(device)`` / ``is_initialized`` flag — one-shot ctor.
- No ``update_weights(state_dict)`` — HTTP path doesn't accept raw state dicts.
- No diffusion-style ``generate(RolloutRequest)`` — that contract is replaced
  by the typed ``generate(req: RolloutReq) -> RolloutResp``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

try:
    import httpx
except ImportError:  # pragma: no cover - exercised only when httpx is missing
    httpx = None  # type: ignore[assignment]

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.sglang_llm._server import (
    find_free_port,
    kill_process_tree,
    run_router_process,
    wait_router_ready,
    wait_server_healthy,
)
from unirl.rollout.engine.sglang_llm.config import SGLangLLMEngineConfig
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_ar_params
from unirl.types.segments.text import TextSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# <think>...</think> tag parsing (Qwen3-style thinking models)
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_UNCLOSED_THINK_PATTERN = re.compile(r"<think>(.*)$", re.DOTALL)


def _strip_thinking_tags(text: str) -> Tuple[str, str]:
    """Split LLM output into ``(content, reasoning_content)``.

    Handles both closed ``<think>...</think>`` and unclosed ``<think>...`` (when
    ``max_new_tokens`` cuts off before the closing tag).
    """
    matches = _THINK_PATTERN.findall(text)
    if matches:
        reasoning = "\n".join(matches)
        content = _THINK_PATTERN.sub("", text).strip()
        return content, reasoning

    unclosed = _UNCLOSED_THINK_PATTERN.search(text)
    if unclosed:
        reasoning = unclosed.group(1).strip()
        content = text[: unclosed.start()].strip()
        return content, reasoning

    return text.strip(), ""


# ---------------------------------------------------------------------------
# Image serialization for HTTP transport
# ---------------------------------------------------------------------------


def _pil_to_base64(image: Any) -> str:
    """Encode a PIL image as a ``data:image/png;base64,...`` URI for SRT."""
    import base64
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@dataclass
class MMEncoding:
    """One VLM sample's multimodal input for the SRT rollout.

    ``image`` (PIL) is always set for a VLM sample — it is what gets base64'd
    into the ``/generate`` ``image_data`` so the server actually attends the
    image. The remaining fields are populated only when the HF processor ran
    (see :meth:`SGLangLLMRolloutEngine._encode_mm`):

    - ``text``: the chat-templated string with a SINGLE ``<|image_pad|>`` — sent
      to SRT, whose processor re-expands the placeholder. (Sending the
      pre-expanded ``input_ids`` + ``image_data`` instead makes SRT return 500.)
    - ``input_ids``: the processor's EXPANDED id sequence — stored as the replay
      prompt so rollout and replay teacher-force over the identical token stream.
    - ``pixel_values`` / ``image_grid_thw``: attached to the response conditions
      so the replay teacher-forces over the IDENTICAL multimodal input.

    When no processor is available these stay None and the plain chat-template
    path runs (the image is still attended via ``image_data``).
    """

    image: Any = None
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    pixel_values: Any = None
    image_grid_thw: Any = None


# ---------------------------------------------------------------------------
# Response parsing — module-level for unit testing without a live engine
# ---------------------------------------------------------------------------


def _parse_one_response(
    response: Any,
    prompt: str,
    known_prompt_token_ids: Optional[List[int]] = None,
    tokenizer: Any = None,
) -> List[Dict[str, Any]]:
    """Parse a sglang SRT ``/generate`` response into a list of candidate dicts.

    Each candidate dict carries ``text``, ``content``, ``reasoning_content``,
    ``token_ids``, ``logprobs``, ``prompt_token_ids``, ``finish_reason``,
    ``prompt``. SGLang returns either a single dict (n=1) or a list of dicts
    (n>1); both are normalized to a list here.

    ``text`` is the raw sampler output and is what ``build_rollout_resp`` emits
    as ``decoded`` (reward-grading input — verl-reference parity); ``content``/
    ``reasoning_content`` are the think-stripped split kept for other consumers.
    """
    if isinstance(response, list):
        candidates = response
    elif isinstance(response, dict):
        candidates = [response]
    else:
        raise RuntimeError(f"Unexpected sglang response type: {type(response)}")

    results: List[Dict[str, Any]] = []
    for candidate in candidates:
        meta = candidate.get("meta_info", {})

        raw_logprobs = meta.get("output_token_logprobs", [])
        token_logprobs: List[float] = []
        output_token_ids: List[int] = []
        for item in raw_logprobs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                token_logprobs.append(float(item[0]))
                output_token_ids.append(int(item[1]))
            elif isinstance(item, (int, float)):
                token_logprobs.append(float(item))

        if not output_token_ids:
            output_token_ids = list(meta.get("output_token_ids", []))

        raw_finish = meta.get("finish_reason", "unknown")
        if isinstance(raw_finish, dict):
            finish_reason = str(raw_finish.get("type", "unknown"))
        else:
            finish_reason = str(raw_finish)

        raw_text = str(candidate.get("text", ""))
        content, reasoning_content = _strip_thinking_tags(raw_text)

        n_tok = len(output_token_ids)
        n_lp = len(token_logprobs)
        if n_tok != n_lp:
            logger.warning(
                "SGLangLLMRolloutEngine: token_ids/logprobs length MISMATCH: token_ids=%d logprobs=%d text_len=%d",
                n_tok,
                n_lp,
                len(raw_text),
            )

        if known_prompt_token_ids is not None:
            prompt_token_ids = known_prompt_token_ids
        elif tokenizer is not None:
            prompt_token_ids = tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(meta.get("input_token_ids", []))

        results.append(
            {
                "text": raw_text,
                "content": content,
                "reasoning_content": reasoning_content,
                "token_ids": output_token_ids,
                "logprobs": token_logprobs,
                "prompt_token_ids": prompt_token_ids,
                "finish_reason": finish_reason,
                "prompt": prompt,
            }
        )

    return results


def build_rollout_resp(
    req: RolloutReq,
    prompts: List[str],
    raw_results: List[Dict[str, Any]],
    *,
    n_per_prompt: int,
    pad_token_id: int = 0,
    mm_encs: Optional[List["MMEncoding"]] = None,
) -> RolloutResp:
    """Pack a list of per-candidate dicts into a typed ``RolloutResp``.

    Pure function — no engine state. Tests feed canned ``raw_results`` and
    check the resulting ``RolloutResp`` shape.

    ``raw_results`` is in prompt-major order: candidate ``k`` of prompt
    ``i`` is at index ``i * n_per_prompt + k``. The output's
    ``tracks['ar'].decoded`` / ``tracks['ar'].segment`` rows are in the same
    order (segment rows 1:1 with samples, so downstream ``Segment.slice``
    reads the right tokens for sample ``j``).

    Per-sample ``prompt_token_ids`` (the chat-template-formatted input the
    LLM server saw — already produced by :meth:`_apply_chat_template` and
    threaded into ``_parse_one_response`` as ``known_prompt_token_ids``)
    are packed into a :class:`TextTokenCondition` on
    ``track.conditions['prompt']``. Right-padded to the in-batch max with
    ``pad_token_id``; attention_mask zeros out the pad positions. This is
    what :meth:`Qwen3ARStage.replay` consumes at train time to teacher-force
    over ``prompt + response``.
    """
    require(
        len(raw_results) == len(prompts) * n_per_prompt,
        f"build_rollout_resp: expected {len(prompts) * n_per_prompt} candidates "
        f"({len(prompts)} prompts × n={n_per_prompt}); got {len(raw_results)}",
    )

    decoded_texts: List[str] = []
    per_sample_tokens: List[torch.Tensor] = []
    per_sample_logprobs: List[torch.Tensor] = []
    per_sample_prompt_ids: List[List[int]] = []
    sample_ids: List[str] = []
    group_ids: List[str] = []
    # VLM: per-sample pixel_values / image_grid_thw, replicated from the
    # prompt-level processor encoding so each sibling sample carries the image
    # condition its rollout was generated under (replay reads these back).
    per_sample_pixel_values: List[Any] = []
    per_sample_image_grid_thw: List[Any] = []

    has_req_sids = bool(req.sample_ids)
    has_req_gids = bool(req.group_ids)

    for prompt_idx in range(len(prompts)):
        base = prompt_idx * n_per_prompt
        req_sid = req.sample_ids[prompt_idx] if has_req_sids else f"s{prompt_idx}"
        req_gid = req.group_ids[prompt_idx] if has_req_gids else req_sid
        enc = mm_encs[prompt_idx] if mm_encs is not None else None
        for k in range(n_per_prompt):
            r = raw_results[base + k]
            # decoded carries the RAW sampler text (<think> content included) so
            # reward grading matches the verl reference, which scores the full
            # decoded response. The previous `content or text` selection silently
            # flipped to raw whenever the think-strip left an empty string
            # (truncated mid-think); think-stripped text stays available as
            # r["content"] / r["reasoning_content"] for other consumers.
            text = r.get("text") or ""
            decoded_texts.append(text)
            tokens = r.get("token_ids") or []
            logprobs = r.get("logprobs") or []
            per_sample_tokens.append(torch.tensor(tokens, dtype=torch.long))
            per_sample_logprobs.append(torch.tensor(logprobs, dtype=torch.float32))
            per_sample_prompt_ids.append(list(r.get("prompt_token_ids") or []))
            sample_ids.append(f"{req_sid}#{k}" if n_per_prompt > 1 else req_sid)
            group_ids.append(req_gid)
            if enc is not None:
                per_sample_pixel_values.append(enc.pixel_values)
                per_sample_image_grid_thw.append(enc.image_grid_thw)

    text_segment = TextSegment.pack(
        tokens=per_sample_tokens,
        log_probs=per_sample_logprobs,
    )

    conditions: Dict[str, Any] = {}
    if any(per_sample_prompt_ids):
        max_plen = max(len(p) for p in per_sample_prompt_ids)
        batch = len(per_sample_prompt_ids)
        input_ids = torch.full((batch, max_plen), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros((batch, max_plen), dtype=torch.long)
        for i, p in enumerate(per_sample_prompt_ids):
            n_real = len(p)
            if n_real > 0:
                input_ids[i, :n_real] = torch.tensor(p, dtype=torch.long)
                attention_mask[i, :n_real] = 1
        conditions["prompt"] = TextTokenCondition(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # VLM: attach per-sample pixel_values / image_grid_thw (mirrors
    # QwenVLARConditions.to_dict()). Per-sample lists with FieldKind.CONCAT so
    # they survive the DP split/merge and reach the replay aligned with prompt.
    if per_sample_pixel_values and any(p is not None for p in per_sample_pixel_values):
        conditions["pixel_values"] = per_sample_pixel_values
        conditions["image_grid_thw"] = per_sample_image_grid_thw

    return RolloutResp(
        tracks={
            "ar": RolloutTrack(
                sample_ids=sample_ids,
                parent_ids=list(group_ids) if group_ids else None,
                conditions=conditions,
                segment=text_segment,
                decoded=Texts(texts=decoded_texts),
            ),
        }
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SGLangLLMRolloutEngine(BaseRolloutEngine):
    """LLM text generation engine backed by sglang SRT HTTP server.

    - Launches a local sglang SRT server for an LLM model (e.g. Qwen3).
    - Implements typed ``generate(req: RolloutReq) -> RolloutResp``.
    - Sends prompts in parallel via ``asyncio.gather()`` + ``httpx``.
    - Captures per-token log probs (for RL training).
    - Parses ``<think>`` tags from thinking models.
    - HTTP-based weight sync (init/destroy NCCL group, update from
      distributed, update from tensor) and HTTP-based memory management
      (sleep/wake via release_memory_occupation / resume_memory_occupation).
    """

    _component_name = "sglang_llm"

    def __init__(
        self,
        config: SGLangLLMEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, SGLangLLMEngineConfig),
            f"SGLangLLMRolloutEngine requires SGLangLLMEngineConfig; got {type(config).__name__}",
        )
        # LLM engine carries its own model path on the config; the diffusion
        # engine takes it from model_config. Log if a caller supplied one so
        # the divergence is visible.
        if model_config is not None:
            logger.debug(
                "SGLangLLMRolloutEngine: model_config provided but ignored — "
                "LLM engine uses config.pretrained_model_ckpt_path",
            )
        del strategy  # LLM rollout has no SDE strategy

        # CUDA-IPC tensor sync requires the non-expandable allocator on older
        # kernels (<5.10) that lack pidfd_getfd; matches PE rollout_actor.py.
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        except Exception:
            logger.debug("Failed to disable expandable CUDA segments for SGLang IPC.", exc_info=True)

        self.cfg = config
        self.rank = rank
        self._device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._server_process: Optional[multiprocessing.Process] = None
        self._router_process: Optional[multiprocessing.Process] = None
        self._router_url: Optional[str] = None
        self._http_client: Any = None
        self._tokenizer: Any = None
        self._chat_template_logged: bool = False
        self._parse_response_logged_first: bool = False
        self._is_offloaded = False

        engine_kwargs: Dict[str, Any] = dict(self.cfg.engine_kwargs or {})

        self._model_path: str = engine_kwargs.get("model_path") or self.cfg.pretrained_model_ckpt_path or ""
        require(
            bool(self._model_path),
            "SGLangLLMRolloutEngine requires a model path; provide via "
            "config.pretrained_model_ckpt_path or engine_kwargs['model_path'].",
        )

        tp_override = engine_kwargs.get("tp_size")
        if tp_override is not None:
            self._tp_size = int(tp_override)
        elif self.cfg.tp_size is not None:
            self._tp_size = int(self.cfg.tp_size)
        else:
            self._tp_size = 1

        port_override = engine_kwargs.get("port")
        if port_override is not None:
            self._port = int(port_override)
        elif self.cfg.port is not None:
            self._port = int(self.cfg.port)
        else:
            self._port = find_free_port()

        # Bind 0.0.0.0 so the SRT server accepts cross-node router connections.
        host = str(engine_kwargs.get("host") or self.cfg.host or "0.0.0.0")
        advertise_host = engine_kwargs.get("advertise_host")
        if not advertise_host:
            try:
                import ray

                advertise_host = ray.util.get_node_ip_address()
            except Exception:
                logger.debug("Failed to resolve Ray node IP for SGLang advertise_host.", exc_info=True)
                advertise_host = host if host not in ("0.0.0.0", "") else "127.0.0.1"
        self._base_url = f"http://{advertise_host}:{self._port}"
        self._concurrency = int(engine_kwargs.get("concurrency", self.cfg.concurrency))
        self._weights_onloaded_for_sync = False
        self._lora_loaded = False
        # Versioned LoRA pool: each sync loads a fresh ``{name}_v{N}`` so SRT
        # never serves a stale adapter; ``generate`` tags the latest version.
        self._lora_version = 0
        self._active_adapter: Optional[str] = None

        base_gpu_id = int(engine_kwargs.get("base_gpu_id", 0))
        force_set_cuda = bool(engine_kwargs.get("force_set_cuda_visible_devices", False))
        mem_fraction = float(engine_kwargs.get("mem_fraction_static", 0.88))

        # Colocate: N engines share a node, each initializing its own (tp=1)
        # torch.distributed env. SGLang leaves nccl_port=None → it calls
        # get_free_port() at model-init time, so the instances that finish
        # loading together race onto the *same* port → EADDRINUSE. Pin the port
        # here instead: grab a free one at construction (de-synchronized across
        # workers, exactly like self._port above, which doesn't collide across
        # the 8 colocated HTTP servers) and hand it to SGLang explicitly so it
        # never re-picks at the synchronized post-load moment.
        nccl_port_override = engine_kwargs.get("nccl_port")
        nccl_port = int(nccl_port_override) if nccl_port_override is not None else find_free_port()

        server_kwargs: Dict[str, Any] = {
            "model_path": self._model_path,
            "host": host,
            "port": self._port,
            "tp_size": self._tp_size,
            "mem_fraction_static": mem_fraction,
            "nccl_port": nccl_port,
        }

        current_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
        logger.info(
            "SGLangLLMRolloutEngine GPU config: base_gpu_id=%s, force_set_cuda=%s, current CUDA_VISIBLE_DEVICES=%s",
            base_gpu_id,
            force_set_cuda,
            current_cvd,
        )
        if force_set_cuda:
            gpu_ids = ",".join(str(base_gpu_id + i) for i in range(self._tp_size))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            logger.info(
                "SGLangLLMRolloutEngine: overriding CUDA_VISIBLE_DEVICES=%s",
                gpu_ids,
            )

        # PE-validated ServerArgs allowlist. ``disable_*_cuda_graph`` and
        # ``skip_server_warmup`` are the only knobs that keep TP=8 colocate
        # under the health timeout; ``enable_memory_saver`` +
        # ``enable_weights_cpu_backup`` enable sleep/wake to actually free
        # the KV pool back to the driver (without them sleep is no-op).
        for key in (
            "api_key",
            "context_length",
            "dp_size",
            "schedule_policy",
            "chunked_prefill_size",
            "reasoning_parser",
            "disable_cuda_graph",
            "disable_piecewise_cuda_graph",
            "cuda_graph_max_bs",
            "skip_server_warmup",
            "attention_backend",
            "prefill_attention_backend",
            "decode_attention_backend",
            "enable_memory_saver",
            "enable_weights_cpu_backup",
            "enable_lora",
            "lora_backend",
            "max_lora_rank",
            "lora_target_modules",
            "max_loras_per_batch",
            "max_loaded_loras",
            "enable_multimodal",
        ):
            if key in engine_kwargs:
                server_kwargs[key] = engine_kwargs[key]

        # SGLang's warmup self-check issues requests.get(...) against
        # http://{host}:{port}/model_info which honors HTTP(S)_PROXY env vars
        # and routes loopback through Squid (returns 503, kills SRT). Whitelist
        # the bind + advertise + loopback hosts.
        _extra_no_proxy = f"0.0.0.0,127.0.0.1,localhost,{advertise_host}"
        _cur_no_proxy = os.environ.get("no_proxy", "") or os.environ.get("NO_PROXY", "")
        os.environ["no_proxy"] = f"{_cur_no_proxy},{_extra_no_proxy}" if _cur_no_proxy else _extra_no_proxy
        os.environ["NO_PROXY"] = os.environ["no_proxy"]

        logger.info(
            "Launching SGLang SRT server: model=%s tp=%d port=%d",
            self._model_path,
            self._tp_size,
            self._port,
        )

        self._server_process = self._launch_server(server_kwargs)
        health_timeout = float(engine_kwargs.get("health_timeout_s", 300.0))
        wait_server_healthy(
            self._base_url,
            timeout_s=health_timeout,
            is_alive_fn=lambda: self._server_process is not None and self._server_process.is_alive(),
        )

        if httpx is not None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(None),
                trust_env=False,
            )

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, trust_remote_code=True)
        logger.info("SGLangLLMRolloutEngine: loaded tokenizer from %s", self._model_path)

        # VLM: load the AutoProcessor so multimodal prompts are encoded the SAME
        # way the trainside replay does (processor.apply_chat_template expands the
        # single <|image_pad|> to the per-image vision-token count and emits
        # pixel_values / image_grid_thw). The plain tokenizer emits only ONE
        # placeholder, which would leave both the SRT server and the replay blind
        # to the image. Encoding here (rollout) + storing pixel_values/grid_thw on
        # the response conditions keeps rollout and replay token-for-token aligned,
        # so the importance ratio stays ~1.0 in native-logprob mode.
        self._processor = None
        if self.cfg.image_token is not None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self._model_path, trust_remote_code=True)
            logger.info("SGLangLLMRolloutEngine: loaded AutoProcessor (VLM) from %s", self._model_path)
        logger.info("SGLang SRT server healthy at %s", self._base_url)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def _launch_server(self, server_kwargs: Dict[str, Any]) -> multiprocessing.Process:
        """Launch sglang SRT in a subprocess.

        ``multiprocessing.set_start_method("spawn", force=True)`` is
        process-global; PE-tested, Ray-compatible. Forcing matches the PE
        engine so torch CUDA init in the child happens cleanly.
        """
        from sglang.srt.entrypoints.http_server import launch_server
        from sglang.srt.server_args import ServerArgs

        # NCCL transport defaults — required for cross-process NCCL groups
        # used by weight sync to establish P2P/CUMEM channels. sglang's
        # _set_envs_and_config() defaults these to "0" when enable_symm_mem
        # is False, breaking broadcast with "Cuda failure 'invalid argument'".
        os.environ.setdefault("NCCL_CUMEM_ENABLE", "1")
        os.environ.setdefault("NCCL_NVLS_ENABLE", "1")

        multiprocessing.set_start_method("spawn", force=True)
        server_args = ServerArgs(**server_kwargs)
        p = multiprocessing.Process(target=launch_server, args=(server_args,))
        p.start()
        return p

    def shutdown(self) -> None:
        """Kill SRT server + router and close HTTP client."""
        if self._http_client is not None:
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(self._http_client.aclose())
                finally:
                    loop.close()
            except Exception:
                logger.debug("Failed to close SGLang HTTP client during shutdown.", exc_info=True)
            self._http_client = None
        if self._router_process is not None:
            logger.info("Shutting down sglang router (pid=%s)", self._router_process.pid)
            kill_process_tree(self._router_process.pid)
            self._router_process.join(timeout=5)
            self._router_process = None
            self._router_url = None
        if self._server_process is not None:
            logger.info("Shutting down SGLang SRT server (pid=%s)", self._server_process.pid)
            kill_process_tree(self._server_process.pid)
            self._server_process.join(timeout=10)
            self._server_process = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            logger.debug("Failed to shut down SGLangLLMRolloutEngine finalizer.", exc_info=True)

    # ------------------------------------------------------------------
    # Router lifecycle (multi-engine load balancing)
    # ------------------------------------------------------------------

    def start_router(
        self,
        worker_urls: List[str],
        port: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Start a sglang router daemon with pre-registered worker URLs.

        Only the first actor in a multi-actor LLM rollout group should call
        this. The router is created with all engine URLs known upfront, so
        no dynamic ``POST /workers`` calls are needed after startup.
        """
        router_port = port or find_free_port()
        node_ip = socket.gethostbyname(socket.gethostname())

        from sglang_router.launch_router import RouterArgs

        router_args = RouterArgs(
            worker_urls=list(worker_urls),
            host="0.0.0.0",
            port=router_port,
            disable_health_check=True,
        )
        router_args.log_level = "warn"

        self._router_process = multiprocessing.Process(
            target=run_router_process,
            args=(router_args,),
            daemon=True,
        )
        self._router_process.start()

        self._router_url = f"http://{node_ip}:{router_port}"
        wait_router_ready(self._router_url, timeout_s=30.0, process=self._router_process)

        logger.info(
            "Started sglang router at %s with %d workers (pid=%d)",
            self._router_url,
            len(worker_urls),
            self._router_process.pid,
        )
        return {"router_url": self._router_url}

    def get_server_info(self) -> Dict[str, Any]:
        """Return engine server metadata for router setup on the driver."""
        return {
            "host": self._base_url.split("//")[1].split(":")[0],
            "port": self._port,
            "url": self._base_url,
            "model_path": self._model_path,
            "concurrency": self._concurrency,
        }

    # ------------------------------------------------------------------
    # Generation — typed
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run text generation against the engine and return a typed response."""
        require(
            int(req.batch_size) > 0,
            "SGLangLLMRolloutEngine.generate requires non-empty req (batch_size > 0)",
        )
        text_primitive = req.primitives.get("text")
        require(
            text_primitive is not None,
            "SGLangLLMRolloutEngine.generate requires req.primitives['text']: Texts",
        )
        prompts = list(text_primitive.texts)
        require(
            len(prompts) == int(req.batch_size),
            f"SGLangLLMRolloutEngine.generate: prompt count {len(prompts)} != req.batch_size {int(req.batch_size)}",
        )

        # --- Image extraction (VLM) ---
        image_prim = req.primitives.get("image")
        pil_images: Optional[List[Any]] = None
        if image_prim is not None:
            require(
                self.cfg.image_token is not None,
                "SGLangLLMRolloutEngine.generate: req contains images but "
                "config.image_token is None (text-only mode). Set image_token "
                "in the engine config to enable VLM.",
            )
            require(
                isinstance(image_prim, Images),
                f"SGLangLLMRolloutEngine.generate: req.primitives['image'] must be "
                f"Images, got {type(image_prim).__name__}",
            )
            require(
                len(image_prim) == len(prompts),
                f"SGLangLLMRolloutEngine.generate: image batch {len(image_prim)} != prompt count {len(prompts)}",
            )
            pil_images = image_prim.to_pils()

        ar = get_ar_params(req.sampling_params)
        stage_ar: Dict[str, Any] = dict(req.stage_config.get("ar") or {})
        if self.cfg.samples_pre_expanded:
            # The caller (ARTrainer) already expanded the req to P*N entries,
            # one per GRPO sibling, so emit exactly one completion per entry.
            # samples_per_prompt was consumed by that expand; re-applying it
            # here would generate N completions per already-expanded entry.
            n = 1
        else:
            n = int(ar.samples_per_prompt if ar is not None else stage_ar.get("n", 1))
        sampling: Dict[str, Any] = {
            "n": n,
            "temperature": float(ar.temperature if ar is not None else self.cfg.temperature),
            "top_p": float(ar.top_p if ar is not None else self.cfg.top_p),
            # top_k MUST be threaded through: without it SGLang falls back to the
            # model generation_config default (top_k=20 for Qwen3), which peaks the
            # sampling vs the trainer's top_k=0 (unrestricted) → low intra-group
            # diversity → GRPO advantages collapse → the policy never learns.
            "top_k": int(ar.top_k) if ar is not None else 0,
            "max_new_tokens": int(ar.max_new_tokens if ar is not None else self.cfg.max_new_tokens),
            "return_logprob": bool(stage_ar.get("return_logprob", True)),
            "system_instruction": stage_ar.get("system_instruction") or self.cfg.system_instruction,
        }
        for key in ("stop", "stop_token_ids", "skip_special_tokens"):
            if key in stage_ar:
                sampling[key] = stage_ar[key]

        # VLM: processor-encode each (prompt, image) into an MMEncoding ONCE,
        # here. The chat-templated text (single placeholder) + image_data go to
        # SRT (which re-expands it, so the server sees the image); the expanded
        # input_ids + pixel_values + image_grid_thw are stored on the response
        # conditions so the replay teacher-forces over the IDENTICAL multimodal
        # input -> the importance ratio stays consistent (cf. SD3's
        # return_prompt_embeds). See MMEncoding for the field-level contract.
        mm_encs: Optional[List[MMEncoding]] = None
        if pil_images is not None:
            if self._processor is not None:
                mm_encs = [
                    self._encode_mm(prompts[i], pil_images[i], sampling.get("system_instruction"))
                    for i in range(len(prompts))
                ]
            else:
                # No processor: carry the raw image so the plain chat-template
                # path still attends it via image_data (no replay alignment).
                mm_encs = [MMEncoding(image=img) for img in pil_images]
        raw_results = self._run_async_gather(prompts, sampling, mm_encs=mm_encs)
        pad_id = getattr(self._tokenizer, "pad_token_id", None) or getattr(self._tokenizer, "eos_token_id", None) or 0
        return build_rollout_resp(
            req,
            prompts,
            raw_results,
            n_per_prompt=n,
            pad_token_id=int(pad_id),
            mm_encs=mm_encs,
        )

    def _run_async_gather(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        mm_encs: Optional[List["MMEncoding"]] = None,
    ) -> List[Dict[str, Any]]:
        """Drive ``_generate_text_async`` from a fresh event loop."""
        if self._http_client is None:
            raise RuntimeError(
                "httpx is required for SGLangLLMRolloutEngine.generate. Install httpx: pip install httpx"
            )
        t0 = time.perf_counter()
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(self._generate_text_async(prompts, sampling_params, mm_encs=mm_encs))
        finally:
            loop.close()
        elapsed = time.perf_counter() - t0
        logger.info(
            "SGLangLLMRolloutEngine.generate: %d prompts -> %d results in %.2fs",
            len(prompts),
            len(results),
            elapsed,
        )
        return results

    async def _generate_text_async(
        self,
        prompts: List[str],
        sampling_params: Dict[str, Any],
        mm_encs: Optional[List["MMEncoding"]] = None,
    ) -> List[Dict[str, Any]]:
        """All prompts sent in parallel via asyncio.gather + Semaphore."""
        params = dict(sampling_params)
        n = int(params.pop("n", 1))
        temperature = float(params.get("temperature", 0.7))
        max_new_tokens = int(params.get("max_new_tokens", 512))
        top_p = float(params.get("top_p", 0.9))
        # Map the trainer's top_k=0 (unrestricted, HF convention) to SGLang's
        # top_k=-1 (disabled); a positive value passes through. Omitting it lets
        # SGLang use the model default (top_k=20) → over-peaked sampling.
        _top_k = int(params.get("top_k", 0))
        top_k = _top_k if _top_k > 0 else -1
        return_logprob = bool(params.get("return_logprob", True))
        system_instruction = params.pop("system_instruction", None)

        sem = asyncio.Semaphore(self._concurrency)

        extra_sampling: Dict[str, Any] = {}
        for key in ("stop", "stop_token_ids", "skip_special_tokens"):
            if key in params:
                extra_sampling[key] = params[key]

        async def _generate_one(prompt: str, mm_enc: Optional["MMEncoding"] = None) -> List[Dict[str, Any]]:
            has_image = mm_enc is not None and mm_enc.image is not None
            sampling_block = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "top_k": top_k,
                "n": n,
                **extra_sampling,
            }
            if mm_enc is not None and mm_enc.text is not None:
                # VLM (processor ran): send the chat-templated TEXT (single
                # <|image_pad|>) + image_data so SRT's processor expands the
                # placeholder and the model actually attends the image. (Sending
                # the pre-expanded input_ids + image_data makes SRT return HTTP
                # 500.) ``prompt_token_ids`` carries the processor's EXPANDED ids
                # — used only for replay storage so the train-time teacher-forcing
                # matches the server's generation context.
                prompt_token_ids = mm_enc.input_ids
                payload: Dict[str, Any] = {
                    "text": mm_enc.text,
                    "sampling_params": sampling_block,
                    "return_logprob": return_logprob,
                    "logprob_start_len": 0,
                }
            else:
                prompt_token_ids = self._apply_chat_template(
                    prompt,
                    system_instruction,
                    has_image=has_image,
                )
                if prompt_token_ids is not None:
                    payload = {
                        "input_ids": prompt_token_ids,
                        "sampling_params": sampling_block,
                        "return_logprob": return_logprob,
                        "logprob_start_len": 0,
                    }
                else:
                    payload = {
                        "text": prompt,
                        "sampling_params": sampling_block,
                        "return_logprob": return_logprob,
                        "logprob_start_len": 0,
                    }
            # Activate the synced LoRA adapter for this request. Only set once
            # an adapter has been pushed (and not since invalidated by a weight
            # release); otherwise SRT serves the base model.
            if self._lora_loaded and self._active_adapter:
                payload["lora_path"] = self._active_adapter
            if has_image:
                payload["image_data"] = _pil_to_base64(mm_enc.image)
            async with sem:
                response = await self._apost("/generate", payload)

            parsed = _parse_one_response(response, prompt, prompt_token_ids, tokenizer=self._tokenizer)
            if not self._parse_response_logged_first and parsed:
                self._parse_response_logged_first = True
                first = parsed[0]
                logger.info(
                    "SGLangLLMRolloutEngine first response: "
                    "prompt_token_ids=%d token_ids=%d logprobs=%d raw_text[:200]=%r has_image=%s",
                    len(first.get("prompt_token_ids") or []),
                    len(first.get("token_ids") or []),
                    len(first.get("logprobs") or []),
                    str(first.get("text", ""))[:200],
                    has_image,
                )
            return parsed

        tasks = []
        for i, p in enumerate(prompts):
            enc = mm_encs[i] if mm_encs is not None else None
            tasks.append(_generate_one(p, enc))
        nested = await asyncio.gather(*tasks)
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
                response = await self._http_client.post(url, json=payload)
                response.raise_for_status()
                content = await response.aread()
                return json.loads(content) if content else {}
            except Exception as exc:
                if httpx is not None and isinstance(exc, httpx.HTTPStatusError):
                    status_code = exc.response.status_code
                    if status_code < 500 and status_code not in (408, 429):
                        error_detail = ""
                        if response is not None:
                            try:
                                error_detail = response.text[:500]
                            except Exception as body_exc:
                                error_detail = f"<failed to read response body: {body_exc!r}>"
                        raise RuntimeError(
                            f"SGLang SRT POST {url} failed with non-retryable HTTP {status_code}: "
                            f"{exc} | response={error_detail}"
                        ) from exc
                if attempt >= max_retries - 1:
                    error_detail = ""
                    if response is not None:
                        try:
                            error_detail = response.text[:500]
                        except Exception as body_exc:
                            error_detail = f"<failed to read response body: {body_exc!r}>"
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

    # ------------------------------------------------------------------
    # Multimodal encoding (processor — VLM)
    # ------------------------------------------------------------------

    def _encode_mm(
        self,
        user_prompt: str,
        image: Any,
        system_instruction: Optional[str] = None,
    ) -> "MMEncoding":
        """Processor-encode one (prompt, image) into the model's native layout.

        Returns a fully-populated :class:`MMEncoding`: ``input_ids`` already has
        the image placeholder expanded to the per-image vision-token count. This
        is the SAME encoding the trainside replay (``chat_template.embed``) uses,
        so the rollout (sent to SRT as ``text`` + ``image_data``) and the replay
        (teacher-forced over ``input_ids`` + ``pixel_values``) are token-for-token
        identical.
        """
        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]})
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        enc = self._processor(text=[text], images=[image], return_tensors="pt")
        return MMEncoding(
            image=image,
            text=text,
            input_ids=enc["input_ids"][0].tolist(),
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc["image_grid_thw"],
        )

    # ------------------------------------------------------------------
    # Chat template
    # ------------------------------------------------------------------

    def _apply_chat_template(
        self,
        user_prompt: str,
        system_instruction: Optional[str] = None,
        has_image: bool = False,
    ) -> Optional[List[int]]:
        """Build chat-formatted ``input_ids`` via the tokenizer's chat template.

        When ``has_image`` is True, builds multimodal content parts
        (``[{"type": "image"}, {"type": "text", ...}]``) for VLMs like
        Qwen2.5-VL whose HF chat templates handle structured content
        natively. Falls back to prepending ``image_token`` as a plain
        string for older templates that only accept flat text.

        Returns ``None`` when no chat template can be applied — the tokenizer
        is missing, lacks ``apply_chat_template``, or has no ``chat_template``
        set (a base / non-instruct model); the caller then falls back to the
        raw ``text`` payload variant. A chat template that *is* present but
        fails to render is surfaced (raised), except that VLMs whose template
        rejects structured content parts retry with ``image_token`` as plain
        text.
        """
        if self._tokenizer is None:
            return None
        if not hasattr(self._tokenizer, "apply_chat_template"):
            return None
        # No chat template set (base / non-instruct tokenizer) is a NORMAL
        # condition, not an error: fall back to the raw-text payload rather
        # than crashing the rollout. A template that EXISTS but fails to
        # render is surfaced in the ``try`` below.
        if getattr(self._tokenizer, "chat_template", None) is None:
            return None

        messages: List[Dict[str, Any]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        if has_image and self.cfg.image_token:
            content: List[Dict[str, str]] = [
                {"type": "image"},
                {"type": "text", "text": user_prompt},
            ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": user_prompt})

        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **(self.cfg.chat_template_kwargs or {}),
            )
        except Exception:
            # Fallback for tokenizers that don't support structured content
            # parts: prepend the image_token as plain text.
            if not (has_image and self.cfg.image_token):
                raise
            fallback_text = f"{self.cfg.image_token}\n{user_prompt}"
            messages_fb: List[Dict[str, Any]] = []
            if system_instruction:
                messages_fb.append({"role": "system", "content": system_instruction})
            messages_fb.append({"role": "user", "content": fallback_text})
            input_ids = self._tokenizer.apply_chat_template(
                messages_fb,
                add_generation_prompt=True,
                tokenize=True,
                **(self.cfg.chat_template_kwargs or {}),
            )

        if not self._chat_template_logged:
            self._chat_template_logged = True
            decoded_preview = self._tokenizer.decode(input_ids[:30], skip_special_tokens=False)
            logger.info(
                "Chat template applied: %d tokens, preview=%r, has_image=%s",
                len(input_ids),
                decoded_preview,
                has_image,
            )

        return input_ids

    # ------------------------------------------------------------------
    # Sync HTTP for non-generation endpoints (weight sync, memory)
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Any:
        """Synchronous POST JSON to sglang SRT server."""
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
            except Exception as body_exc:
                error_body = f"<failed to read error body: {body_exc!r}>"
            raise RuntimeError(f"SGLang SRT HTTP {exc.code} for {url}: {error_body}") from exc

    @staticmethod
    def _check_update_response(response: Any, operation: str) -> None:
        if isinstance(response, dict):
            if not response.get("success", True):
                detail = response.get("error_message") or response.get("message", "unknown")
                raise RuntimeError(f"SGLangLLMRolloutEngine.{operation} failed: {detail}")

    # ------------------------------------------------------------------
    # Weight sync — HTTP POST to sglang SRT
    # ------------------------------------------------------------------

    def update_weights_from_path(self, checkpoint_path: str) -> None:
        """Update weights from a checkpoint on disk."""
        resp = self._post(
            "/update_weights_from_disk",
            {"model_path": checkpoint_path, "flush_cache": True},
        )
        self._check_update_response(resp, "update_weights_from_path")

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Update weights from serialized tensors via HTTP.

        ``target_modules`` is intentionally NOT forwarded — the diffusion-side
        default ``["transformer"]`` doesn't match LLM module naming. Omitting
        the field lets the SRT server accept all incoming weights correctly.
        """
        del track_prefix
        payload: Dict[str, Any] = {
            "serialized_named_tensors": serialized_named_tensors,
            "flush_cache": flush_cache,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        resp = self._post("/update_weights_from_tensor", payload)
        self._check_update_response(resp, "update_weights_from_tensor")

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        resp = self._post(
            "/init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": int(master_port),
                "rank_offset": int(rank_offset),
                "world_size": int(world_size),
                "group_name": str(group_name),
                "backend": str(backend),
            },
        )
        self._check_update_response(resp, "init_weights_update_group")
        logger.info(
            "SGLangLLMRolloutEngine: NCCL group %r initialized (rank_offset=%d, world_size=%d)",
            group_name,
            rank_offset,
            world_size,
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        resp = self._post(
            "/destroy_weights_update_group",
            {"group_name": str(group_name)},
        )
        self._check_update_response(resp, "destroy_weights_update_group")

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Receive weights via NCCL broadcast from training actors.

        ``target_modules`` is intentionally NOT forwarded (see
        ``update_weights_from_tensor`` for rationale).
        """
        # sglang expects bare dtype strings like "bfloat16", not "torch.bfloat16".
        clean_dtypes = [d.replace("torch.", "") if isinstance(d, str) else d for d in dtypes]
        logger.info(
            "SGLangLLMRolloutEngine: update_weights_from_distributed group=%s, %d params, first=%s last=%s, flush=%s",
            group_name,
            len(names),
            names[0] if names else "<empty>",
            names[-1] if names else "<empty>",
            flush_cache,
        )
        resp = self._post(
            "/update_weights_from_distributed",
            {
                "names": list(names),
                "dtypes": clean_dtypes,
                "shapes": [list(s) for s in shapes],
                "group_name": str(group_name),
                "flush_cache": flush_cache,
            },
        )
        self._check_update_response(resp, "update_weights_from_distributed")

    # ------------------------------------------------------------------
    # Weight sync — LoRA tensor bag
    # ------------------------------------------------------------------

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Load a LoRA adapter from in-memory tensors via SGLang SRT HTTP API.

        SGLang SRT exposes ``POST /load_lora_adapter_from_tensors`` which
        accepts serialized LoRA tensors + a PEFT config dict and hot-loads the
        adapter on all TP workers internally. ``track_prefix`` routing is
        handled by the parent :class:`ComposedRolloutEngine`; this child sees
        only its own (prefix-stripped) tensors.
        """
        try:
            from sglang.srt.utils import MultiprocessingSerializer
        except ImportError:
            from sglang.srt.utils.utils import MultiprocessingSerializer

        serialized = MultiprocessingSerializer.serialize(lora_tensors, output_str=True)
        # Rotate to a fresh VERSIONED name each sync (spo_v2 lora-pool approach).
        # Re-loading the SAME ``lora_name`` is unreliable: SRT rejects a duplicate
        # name (HTTP 400 "already loaded"), an explicit /unload of the live adapter
        # can stall for minutes under colocate, and — the killer here — reusing the
        # name can serve STALE weights, so the rollout policy never actually updates
        # (reward stays flat while the FSDP model trains). Loading a NEW name forces
        # fresh weights; generation points at the latest via ``_active_adapter``.
        # Stale versions evict via SRT's LRU (``max_loaded_loras``).
        self._lora_version += 1
        versioned_name = f"{adapter_name}_v{self._lora_version}"
        resp = self._post(
            "/load_lora_adapter_from_tensors",
            {
                "lora_name": versioned_name,
                "config_dict": dict(peft_config or {}),
                "serialized_tensors": serialized,
            },
        )
        self._check_update_response(resp, "set_lora_from_tensors")
        self._active_adapter = versioned_name
        self._lora_loaded = True
        logger.info(
            "SGLangLLMRolloutEngine: LoRA adapter %r loaded as %r (%d tensor keys)",
            adapter_name,
            versioned_name,
            len(lora_tensors),
        )

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(
        self,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Release GPU memory (offload).

        Flushes the cache first; sglang's release only fully frees the KV
        pool when the scheduler has no pending references.

        ``tags`` selects which sglang SRT memory regions to release (e.g.
        ``["weights"]``). ``None`` releases everything.

        multi-stage routing kwarg used by :class:`ComposedRolloutEngine`
        when dispatching to a single-stage child. The composed parent
        handles the routing; this child sees only the calls that actually
        target it.
        """
        release_tags = None if tags is None or len(tags) == 0 else list(tags)
        if release_tags is None and self._is_offloaded:
            if not self._weights_onloaded_for_sync:
                return
            release_tags = ["weights"]
        if self._server_process is None or not self._server_process.is_alive():
            raise RuntimeError("Cannot sleep SGLangLLMRolloutEngine: SRT server is not alive.")
        if release_tags is None or "kv_cache" in release_tags:
            self._flush_cache()

        payload: Dict[str, Any] = {}
        if release_tags is not None:
            payload["tags"] = release_tags
        self._post("/release_memory_occupation", payload)
        self._is_offloaded = True
        self._weights_onloaded_for_sync = False
        # Releasing weights frees the SRT LoRA pool; the adapter must be
        # re-pushed (set_lora_from_tensors) before it can be referenced again.
        if release_tags is None or "weights" in release_tags:
            self._lora_loaded = False

    def onload_weights(self, *, track_prefix: str = "") -> None:
        """Resume only model weights so tensor/NCCL sync can update them."""
        del track_prefix
        if not self._is_offloaded:
            return
        if self._weights_onloaded_for_sync:
            return
        if self._server_process is None or not self._server_process.is_alive():
            raise RuntimeError("Cannot onload SGLangLLMRolloutEngine weights: SRT server is not alive.")
        self._post("/resume_memory_occupation", {"tags": ["weights"]})
        self._weights_onloaded_for_sync = True

    def _flush_cache(self) -> None:
        """Flush sglang scheduler cache; retry until 200.

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
            f"SGLangLLMRolloutEngine: /flush_cache did not return 200 after 60 attempts (last error: {last_err})"
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(
        self,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Resume GPU memory.

        Can be called multiple times with different tag subsets for a staged
        resume — e.g. ``wake_up(tags=["weights"])`` to allow weight sync, then
        ``wake_up(tags=["kv_cache", "cuda_graph"])`` before generation.

        """
        full_wake = tags is None or len(tags) == 0
        resume_tags = None if full_wake else list(tags)
        if resume_tags is None:
            if not self._is_offloaded:
                return
            if self._weights_onloaded_for_sync:
                resume_tags = ["kv_cache", "cuda_graph"]
        if self._server_process is None or not self._server_process.is_alive():
            raise RuntimeError("Cannot wake SGLangLLMRolloutEngine: SRT server is not alive.")
        payload: Dict[str, Any] = {}
        if resume_tags is not None:
            payload["tags"] = resume_tags
        self._post("/resume_memory_occupation", payload)
        if full_wake:
            self._is_offloaded = False
            self._weights_onloaded_for_sync = False
        elif "weights" in payload["tags"]:
            self._weights_onloaded_for_sync = True

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        """Check if sglang SRT server is responsive."""
        if self._is_offloaded:
            return True
        if self._server_process is None or not self._server_process.is_alive():
            return False
        try:
            wait_server_healthy(self._base_url, timeout_s=5, poll_interval_s=1)
            return True
        except (TimeoutError, RuntimeError):
            return False


__all__ = ["SGLangLLMRolloutEngine", "build_rollout_resp"]
