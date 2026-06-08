"""UnifiedReward-2.0 scorer via in-process vLLM.

Loads CodeGoat24/UnifiedReward-2.0-qwen3vl-{2b,4b} and exposes three
sub-metrics: alignment / coherence / style, each on a 1-5 scale.
We parse the model's free-form text output with a tolerant regex; if
parsing fails, that sub-metric is reported as NaN and a warning logged.

Prompt strategy: point-score template adapted from the official
inference_qwen/UnifiedReward-2.0-inference scripts. The model is asked
to rate the last (text, image) turn.
"""

from __future__ import annotations

import math
import re
from typing import Any

from reward_service.logging_utils import get_logger
from reward_service.scorers._common import (
    DEFAULT_VLLM_MM_LIMIT,
    build_vllm_llm_kwargs,
    image_to_data_url,
    resolve_model_path,
)
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.registry import register

logger = get_logger(__name__)

_POINT_SCORE_TEMPLATE = (
    "You are given a text prompt and one generated image. Rate the image on "
    "three dimensions from 1 to 5 (higher is better):\n"
    "- Alignment: how well the image matches the prompt.\n"
    "- Coherence: visual plausibility and absence of artifacts.\n"
    "- Style: aesthetic quality and stylistic consistency.\n\n"
    "Prompt: {prompt}\n\n"
    "Respond strictly in the format:\n"
    "Alignment: <score>\nCoherence: <score>\nStyle: <score>"
)

_SCORE_PATTERN = re.compile(
    r"(?P<name>alignment|coherence|style)\s*[::]\s*(?P<val>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


class UnifiedRewardScorer(BaseScorer):
    name = "unified_reward"
    sub_metric_names = ("alignment", "coherence", "style")

    def __init__(
        self,
        model_name: str = "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b",
        weights_path: str | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = 8192,
        max_tokens: int = 128,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        swap_space: int = 4,
        quantization: str | None = None,
        seed: int | None = None,
        max_num_seqs: int = 256,
        trust_remote_code: bool = True,
        limit_mm_per_prompt: dict[str, int] | None = None,
        extra_llm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        from vllm import LLM, SamplingParams

        model_path = resolve_model_path(model_name, weights_path)
        llm_kwargs = build_vllm_llm_kwargs(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=enforce_eager,
            swap_space=swap_space,
            quantization=quantization,
            seed=seed,
            max_num_seqs=max_num_seqs,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt=(
                limit_mm_per_prompt if limit_mm_per_prompt is not None else DEFAULT_VLLM_MM_LIMIT
            ),
            extra_llm_kwargs=extra_llm_kwargs,
        )
        self.llm = LLM(**llm_kwargs)
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        messages_batch = []
        for item in items:
            text, image = item.history[-1]
            data_url = image_to_data_url(image)
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": data_url}},
                            {
                                "type": "text",
                                "text": _POINT_SCORE_TEMPLATE.format(prompt=text),
                            },
                        ],
                    }
                ]
            )

        outputs = self.llm.chat(messages=messages_batch, sampling_params=self.sampling)
        return [self._parse(o.outputs[0].text) for o in outputs]

    @staticmethod
    def _parse(text: str) -> dict[str, float]:
        found: dict[str, float] = {}
        for m in _SCORE_PATTERN.finditer(text):
            found[m.group("name").lower()] = float(m.group("val"))
        result: dict[str, float] = {}
        for key in ("alignment", "coherence", "style"):
            if key in found:
                result[key] = found[key]
            else:
                logger.warning("UnifiedReward failed to parse %s from output: %r", key, text[:200])
                result[key] = math.nan
        return result


register("unified_reward", UnifiedRewardScorer)
