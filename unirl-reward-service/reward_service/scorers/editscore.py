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
  consistency + perceptual quality), so a batch of N items is 2*N*num_pass
  sequential vLLM calls. Fine for latency-focused trainside DP; batching
  across items via ``batch_inference`` is a follow-up.
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
        max_model_len: int = 1536,
        max_num_batched_tokens: int = 1536,
        max_num_seqs: int = 32,
        gpu_memory_utilization: float | None = None,
        enable_sleep_mode: bool = False,
        extra_llm_kwargs: dict[str, Any] | None = None,
        report_sub_metrics: list[str] | None = None,
        max_image_side: int | None = 1536,
    ) -> None:
        import importlib

        from editscore import EditScore

        if report_sub_metrics:
            unknown = set(report_sub_metrics) - set(self.sub_metric_names)
            if unknown:
                raise ValueError(
                    f"unknown report_sub_metrics {sorted(unknown)}; "
                    f"available: {list(self.sub_metric_names)}"
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
            inject.setdefault("mm_processor_cache_gb", 0)

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
        results: list[dict[str, float]] = []
        for item in items:
            try:
                if len(item.history) < 2:
                    raise ValueError(
                        f"EditScore requires 2 history turns (source + edited), got {len(item.history)}"
                    )
                prompt, source_image = item.history[0]
                _, edited_image = item.history[1]
                if source_image is None or edited_image is None:
                    raise ValueError("Both source and edited images must be provided")
                source_image = self._cap_image(source_image)
                edited_image = self._cap_image(edited_image)
                out = self.es.evaluate([source_image, edited_image], prompt)
                results.append({k: float(out[k]) for k in self.sub_metric_names})
            except Exception:
                logger.exception("EditScore failed to score item %d", len(results))
                results.append({k: float("nan") for k in self.sub_metric_names})
        return results

    def _engine(self):
        return self.es.model.model  # EditScore -> Qwen3VL wrapper -> vllm.LLM

    def onload(self) -> None:
        if self._sleep_enabled:
            self._engine().wake_up()

    def offload(self) -> None:
        if self._sleep_enabled:
            self._engine().sleep(level=1)

    def close(self) -> None:
        if hasattr(self, "es"):
            del self.es


register("editscore", EditScoreScorer)
