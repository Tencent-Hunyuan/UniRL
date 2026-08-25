"""EditScore scorer — VLM-judge reward for instruction-guided image editing.

Wraps the ``editscore`` package (VectorSpaceLab/EditScore). One class covers
the whole model family: the backbone (qwen25vl / qwen3vl, HF or vLLM engine)
and the base-checkpoint / LoRA pair are constructor params, so swapping
EditScore-7B for EditScore-Qwen3-VL-8B is a config change, not a code change.

Input convention (same as ``editreward``):
    history[0] = (prompt, source_image)   — source image before editing
    history[1] = (prompt, edited_image)   — edited image after editing

Package quirks this wrapper absorbs:

* LoRA is not applied at runtime: on first use the package merges the
  adapter into the base with peft (CPU) and saves a full merged checkpoint
  to ``cache_dir``; later runs load that cache directly. Point ``cache_dir``
  at shared storage so the one-time merge is reused across pods, and give
  the managed child a generous ``startup_timeout`` for the first run.
* The vLLM backbones hardcode ``LLM()`` kwargs (no ``gpu_memory_utilization``).
  We inject extra kwargs by wrapping the backbone module's ``LLM`` symbol
  during construction — required whenever the scorer shares a GPU with
  training (trainside) instead of owning the whole card.
* ``evaluate()`` is per-item (two generate calls per pass: semantic
  consistency + perceptual quality), so a batch of N items would be
  2*N*num_pass sequential vLLM calls. ``score()`` mirrors its retry and
  parsing semantics while batching each attempt through ``batch_inference``;
  rows that still fail fall back to upstream ``evaluate()`` individually.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from reward_service.logging_utils import get_logger
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = get_logger(__name__)

# backbone name → module whose ``LLM`` symbol we wrap to inject kwargs.
_VLLM_BACKBONE_MODULES = {
    "qwen3vl_vllm": "editscore.mllm_tools.qwen3vl_vllm",
    "qwen25vl_vllm": "editscore.mllm_tools.qwen25vl_vllm",
}


class EditScoreScorer(BaseScorer):
    name = "editscore"
    version = "1"
    input_kind = "image"
    # Class-level capability (checked by the server BEFORE construction so a
    # per_call misconfiguration does not pay a full engine load): the vLLM
    # backbones CAN offload — via sleep mode. Whether this instance actually
    # may is config-dependent (enable_sleep_mode); onload/offload raise a
    # pointed error otherwise instead of silently no-oping, which would report
    # state=offloaded while the engine still holds the GPU.
    supports_offload = True
    sub_metric_names = ("prompt_following", "consistency", "perceptual_quality", "overall")

    def __init__(
        self,
        model_name_or_path: str,
        backbone: str = "qwen3vl_vllm",
        lora_path: str | None = None,
        cache_dir: str | None = None,
        score_range: int = 25,
        num_pass: int = 1,
        temperature: float = 0.7,
        seed: int = 42,
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        max_num_batched_tokens: int = 8192,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float | None = None,
        enable_sleep_mode: bool = False,
        extra_llm_kwargs: dict[str, Any] | None = None,
        report_sub_metrics: list[str] | None = None,
        max_image_side: int | None = 1536,
        batched: bool = True,
    ) -> None:
        import importlib

        from editscore import EditScore

        if report_sub_metrics:
            unknown = set(report_sub_metrics) - set(self.sub_metric_names)
            if unknown:
                raise ValueError(
                    f"unknown report_sub_metrics {sorted(unknown)}; available: {list(self.sub_metric_names)}"
                )
            # Narrow what score() emits — e.g. ["overall"] gives RL a single
            # scalar instead of averaging four correlated sub-metrics.
            self.sub_metric_names = tuple(report_sub_metrics)

        inject: dict[str, Any] = dict(extra_llm_kwargs or {})
        if gpu_memory_utilization is not None:
            inject["gpu_memory_utilization"] = gpu_memory_utilization
        if enable_sleep_mode:
            # vLLM sleep mode backs onload()/offload() so per_call lifecycle
            # works even though the engine owns its own memory pools.
            inject["enable_sleep_mode"] = True
            # The frontend's multimodal processor cache survives sleep/wake but
            # the engine-core receiver cache does not: a post-wake request whose
            # image hash-hits the frontend cache is sent hash-only and trips
            # "Expected a cached item for mm_hash=..." in the engine, corrupting
            # scores. Repeat images are guaranteed here (each evaluate() sends
            # the edited image in both the SC and PQ requests), so the cache
            # must be off whenever the engine can sleep.
            inject["mm_processor_cache_gb"] = 0
            # Prefix-cache metadata has the same frontend/engine lifetime
            # mismatch across sleep. Other UniRL vLLM sleep-mode recipes
            # disable it for this reason; override the package's hardcoded
            # enable_prefix_caching=True rather than letting a caller opt back
            # into a cache that cannot survive this lifecycle safely.
            inject["enable_prefix_caching"] = False

        backbone_mod = None
        original_llm = None
        if inject:
            module_path = _VLLM_BACKBONE_MODULES.get(backbone)
            if module_path is None:
                raise ValueError(
                    f"gpu_memory_utilization/enable_sleep_mode/extra_llm_kwargs need a "
                    f"vLLM backbone, got {backbone!r} (supported: {sorted(_VLLM_BACKBONE_MODULES)})"
                )
            backbone_mod = importlib.import_module(module_path)
            original_llm = backbone_mod.LLM

            def _llm_with_injected_kwargs(**kwargs):
                return original_llm(**{**kwargs, **inject})

            backbone_mod.LLM = _llm_with_injected_kwargs
        try:
            self.es = EditScore(
                backbone=backbone,
                model_name_or_path=model_name_or_path,
                lora_path=lora_path,
                cache_dir=cache_dir,
                score_range=score_range,
                num_pass=num_pass,
                temperature=temperature,
                seed=seed,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=max_model_len,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
            )
        finally:
            if backbone_mod is not None:
                backbone_mod.LLM = original_llm

        self.supports_offload = bool(enable_sleep_mode)
        self._sleep_enabled = bool(enable_sleep_mode)
        self._max_image_side = max_image_side
        self._batched = bool(batched) and backbone in _VLLM_BACKBONE_MODULES
        self._num_pass = num_pass
        self._score_range = score_range
        self._seed = seed

    def _cap_image(self, img: Image.Image) -> Image.Image:
        """Bound vision tokens so requests always fit max_model_len.

        Source images arrive at dataset-native resolution; a ~3000px source
        alone is ~8.5k vision tokens ((w/16)*(h/16)/4) and overflows the
        context mid-run on whichever batch draws it. At 1536 the SC request
        (two images + rubric) stays near 5.5k tokens for an 8192 context.
        """
        side = self._max_image_side
        if side and max(img.size) > side:
            img = img.copy()
            img.thumbnail((side, side), Image.LANCZOS)
        return img

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        rows: list[tuple[int, str, Image.Image, Image.Image]] = []
        results: list[dict[str, float] | None] = [None] * len(items)
        for i, item in enumerate(items):
            try:
                if len(item.history) < 2:
                    raise ValueError(f"EditScore requires 2 history turns (source + edited), got {len(item.history)}")
                prompt, source_image = item.history[0]
                _, edited_image = item.history[1]
                if source_image is None or edited_image is None:
                    raise ValueError("Both source and edited images must be provided")
                rows.append((i, prompt, self._cap_image(source_image), self._cap_image(edited_image)))
            except Exception:
                logger.exception("EditScore failed to score item %d", i)
                results[i] = {k: float("nan") for k in self.sub_metric_names}

        if rows:
            try:
                scored = self._score_rows_batched(rows) if self._batched else None
            except Exception as exc:
                if self._is_oom_error(exc):
                    raise
                logger.exception("EditScore batched scoring failed; falling back to per-item")
                scored = None
            fallback_rows = []
            if scored is not None:
                for row, out in zip(rows, scored):
                    i = row[0]
                    if out is None:
                        fallback_rows.append(row)
                    else:
                        results[i] = out
            else:
                fallback_rows = rows
            for i, prompt, src, edited in fallback_rows:
                try:
                    out = self.es.evaluate([src, edited], prompt)
                    results[i] = {k: float(out[k]) for k in self.sub_metric_names}
                except Exception as exc:
                    if self._is_oom_error(exc):
                        raise
                    logger.exception("EditScore failed to score item %d", i)
                    results[i] = {k: float("nan") for k in self.sub_metric_names}
        return [r if r is not None else {k: float("nan") for k in self.sub_metric_names} for r in results]

    @staticmethod
    def _is_oom_error(exc: BaseException) -> bool:
        # Keep torch optional at module import time like the other scorer
        # dependencies; it is present in every environment that can construct
        # EditScore.
        import torch

        return isinstance(exc, torch.cuda.OutOfMemoryError)

    def _score_rows_batched(
        self,
        rows: list[tuple[int, str, Image.Image, Image.Image]],
    ) -> list[dict[str, float] | None]:
        """Batched twin of ``EditScore.evaluate``.

        The package scores one item at a time (two ``generate`` calls per
        pass), so a rollout's batch is 2*N*num_pass sequential engine round
        trips. Here all SC prompts and all PQ prompts of a pass go through
        ``batch_inference`` as two batched calls and vLLM schedules them
        concurrently. Like ``evaluate``, a parse failure retries generation
        twice before the third attempt enables the tolerant give-up parser.
        Parsing, the per-request seed (``seed + pass``), the min-over-heads
        scaling and the sqrt(SC*PQ) overall compose exactly as upstream.
        Persistent per-row failures return ``None`` so ``score()`` can retry
        them through upstream ``evaluate()`` without discarding good rows.
        """
        import numpy as np
        from editscore.utils import mllm_output_to_dict

        es = self.es
        scale = self._score_range / 10
        sc_msgs = [
            es.model.prepare_input([src, ed], es.SC_prompt.replace("<instruction>", p)) for _, p, src, ed in rows
        ]
        pq_msgs = [es.model.prepare_input(ed, es.PQ_prompt) for _, _, _, ed in rows]

        refusal = "I'm sorry, but I can't assist with that request."
        per_pass: list[list[dict[str, float] | None]] = []
        for pass_index in range(self._num_pass):
            pass_outs: list[dict[str, float] | None] = [None] * len(rows)
            pending = list(range(len(rows)))
            for attempt in range(3):
                if not pending:
                    break
                sc_texts = es.model.batch_inference(
                    [sc_msgs[idx] for idx in pending],
                    seed=self._seed + pass_index,
                )
                pq_texts = es.model.batch_inference(
                    [pq_msgs[idx] for idx in pending],
                    seed=self._seed + pass_index,
                )
                if len(sc_texts) != len(pending) or len(pq_texts) != len(pending):
                    raise RuntimeError(
                        "EditScore batch_inference returned an unexpected number of outputs: "
                        f"pending={len(pending)} SC={len(sc_texts)} PQ={len(pq_texts)}"
                    )

                retry: list[int] = []
                for row_index, sc_text, pq_text in zip(pending, sc_texts, pq_texts):
                    prompt = rows[row_index][1]
                    give_up = attempt == 2 or sc_text == refusal or pq_text == refusal
                    try:
                        sc = mllm_output_to_dict(
                            sc_text,
                            give_up_parsing=give_up,
                            text_prompt=prompt,
                            score_range=self._score_range,
                        )
                        pq = mllm_output_to_dict(
                            pq_text,
                            give_up_parsing=give_up,
                            text_prompt=prompt,
                            score_range=self._score_range,
                        )
                        if sc == "rate_limit_exceeded" or pq == "rate_limit_exceeded":
                            raise RuntimeError("EditScore rate_limit_exceeded")
                        if not isinstance(sc, dict) or not isinstance(pq, dict):
                            retry.append(row_index)
                            continue
                        sc_score = min(sc["score"]) / scale
                        pq_score = min(pq["score"]) / scale
                        pass_outs[row_index] = {
                            "prompt_following": sc["score"][0] / scale,
                            "consistency": sc["score"][1] / scale,
                            "perceptual_quality": pq_score,
                            "overall": float(np.sqrt(sc_score * pq_score)),
                        }
                    except Exception as exc:
                        if self._is_oom_error(exc):
                            raise
                        if attempt < 2:
                            retry.append(row_index)
                        else:
                            logger.exception(
                                "EditScore batched parse failed after retries for item %d",
                                rows[row_index][0],
                            )
                pending = retry
            per_pass.append(pass_outs)

        merged: list[dict[str, float] | None] = []
        for idx in range(len(rows)):
            outs = [p[idx] for p in per_pass]
            if all(out is not None for out in outs):
                valid_outs = [out for out in outs if out is not None]
                full = {k: float(np.mean([out[k] for out in valid_outs])) for k in valid_outs[0]}
                merged.append({k: full[k] for k in self.sub_metric_names})
            else:
                merged.append(None)
        return merged

    def _engine(self):
        return self.es.model.model  # EditScore -> Qwen3VL wrapper -> vllm.LLM

    def _require_sleep_mode(self, action: str) -> None:
        if not self._sleep_enabled:
            raise RuntimeError(
                f"EditScore {action} requires enable_sleep_mode: true in the scorer params — "
                "without it the vLLM engine cannot leave the GPU, and silently no-oping would "
                "report an offloaded state the memory does not reflect (per_call needs sleep mode)"
            )

    def onload(self) -> None:
        self._require_sleep_mode("onload")
        self._engine().wake_up()

    def offload(self) -> None:
        self._require_sleep_mode("offload")
        self._engine().sleep(level=1)

    def close(self) -> None:
        if hasattr(self, "es"):
            del self.es


register("editscore", EditScoreScorer)
