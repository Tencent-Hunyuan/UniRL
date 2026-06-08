"""WISE reward scorer — Qwen3.5-VL judge via in-process vLLM (registered as ``wise``).

Deploys a Qwen3.5 series VLM (dense or MoE) as a reward model. The default rubric
mirrors WISE (WISE_legacy/gpt_eval.py): the judge rates the image on Consistency /
Realism / Aesthetic Quality (each 0-2), and the scorer derives the WISE **WiScore**
= ``(0.7*Consistency + 0.2*Realism + 0.1*Aesthetic) / 2`` in ``[0, 1]`` (matching
WISE_legacy/Calculate.py), exposed as the first sub-metric so a consumer reducing
on the first sub-metric trains on the headline score rather than one component.

Works with Qwen3.5-35B-A3B (MoE, ~70GB bf16), Qwen3.5-122B-A10B, or any Qwen3.5
variant vLLM serves.

The scorer is generic — override ``score_prompt_template`` and ``metric_names`` in
the YAML params for a different rubric. WiScore is only derived for the default
WISE metrics; custom metric sets are returned as parsed.
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

_DEFAULT_SCORE_TEMPLATE = """Please evaluate strictly and return ONLY the three scores as requested.

# Text-to-Image Quality Evaluation Protocol

## System Instruction
You are an AI quality auditor for text-to-image generation. Apply these rules with ABSOLUTE RUTHLESSNESS. No assuming is allowed. You MUST strictly follow the criteria.
Only images meeting the HIGHEST standards should receive top scores. As long as the image doesn't satisfy the criteria, give lower scores.

**Input Parameters**
- PROMPT: [User's original prompt]
- EXPLANATION: [Further explanation of the original prompt]
---

## Scoring Criteria

**Consistency (0-2):**  How accurately and completely the image reflects the PROMPT.
* **0 (Rejected):**  Fails to capture key elements of the prompt, or contradicts the prompt.
* **1 (Conditional):** Partially captures the prompt. Some elements are present, but not all, or not accurately.  Noticeable deviations from the prompt's intent.
* **2 (Exemplary):**  Perfectly and completely aligns with the PROMPT.  Every single element and nuance of the prompt is flawlessly represented in the image. The image is an ideal, unambiguous visual realization of the given prompt.

**Realism (0-2):**  How realistically the image is rendered.
* **0 (Rejected):**  Physically implausible and clearly artificial. Breaks fundamental laws of physics or visual realism.
* **1 (Conditional):** Contains minor inconsistencies or unrealistic elements.  While somewhat believable, noticeable flaws detract from realism.
* **2 (Exemplary):**  Achieves photorealistic quality, indistinguishable from a real photograph.  Flawless adherence to physical laws, accurate material representation, and coherent spatial relationships. No visual cues betraying AI generation.

**Aesthetic Quality (0-2):**  The overall artistic appeal and visual quality of the image.
* **0 (Rejected):**  Poor aesthetic composition, visually unappealing, and lacks artistic merit.
* **1 (Conditional):**  Demonstrates basic visual appeal, acceptable composition, and color harmony, but lacks distinction or artistic flair.
* **2 (Exemplary):**  Possesses exceptional aesthetic quality, comparable to a masterpiece.  Strikingly beautiful, with perfect composition, a harmonious color palette, and a captivating artistic style. Demonstrates a high degree of artistic vision and execution.

---

## Output Format

**Do not include any other text, explanations, or labels.** You must return only three lines of text, each containing a metric and the corresponding score, for example:

**Example Output:**
Consistency: 2
Realism: 1
Aesthetic Quality: 0

---

**IMPORTANT Enforcement:**

Be EXTREMELY strict in your evaluation. A score of '2' should be exceedingly rare and reserved only for images that truly excel and meet the highest possible standards in each metric. If there is any doubt, downgrade the score.

For **Consistency**, a score of '2' requires complete and flawless adherence to every aspect of the prompt, leaving no room for misinterpretation or omission.

For **Realism**, a score of '2' means the image is virtually indistinguishable from a real photograph in terms of detail, lighting, physics, and material properties.

For **Aesthetic Quality**, a score of '2' demands exceptional artistic merit, not just pleasant visuals.

---
Here are the Prompt and EXPLANATION for this evaluation:
PROMPT: "{prompt}"
EXPLANATION: "{explanation}"
Please strictly adhere to the scoring criteria and follow the template format when providing your results."""

# Metric names mirror WISE: "Aesthetic Quality" (with space) appears in the
# VLM output, but we expose it as `aesthetic_quality` (underscore) in the
# returned dict — the regex below tolerates either spelling on input.
_DEFAULT_METRIC_NAMES = ("consistency", "realism", "aesthetic_quality")

# WiScore weights, from WISE_legacy/Calculate.py: a Consistency-dominated weighted
# average of the three 0-2 components, divided by 2 to land in [0, 1].
_WISCORE_WEIGHTS = {"consistency": 0.7, "realism": 0.2, "aesthetic_quality": 0.1}


def _wiscore(metrics: dict[str, float]) -> float:
    """WISE WiScore = ``(0.7*C + 0.2*R + 0.1*A) / 2`` in ``[0, 1]``.

    Returns ``NaN`` if any component is missing or non-finite, so a parse failure
    propagates as a per-item failure instead of a fabricated score.
    """
    total = 0.0
    for key, weight in _WISCORE_WEIGHTS.items():
        value = metrics.get(key, math.nan)
        if value is None or not math.isfinite(value):
            return math.nan
        total += weight * value
    return total / 2.0


def _build_score_pattern(metric_names: tuple[str, ...]) -> re.Pattern:
    """Build a regex that matches any of the configured metric names followed by a number.

    Underscores in the canonical metric name (e.g. ``aesthetic_quality``) are
    treated as ``[_\\s]`` so the regex also matches the space-separated form
    the VLM emits (e.g. "Aesthetic Quality: 2") — WISE prompt mirror.
    """
    def _to_alt(name: str) -> str:
        # ``aesthetic_quality`` -> ``aesthetic[_\s]quality`` so we match the
        # space-separated form the model writes back.
        return r"[_\s]".join(re.escape(part) for part in name.split("_"))

    names_alt = "|".join(_to_alt(n) for n in metric_names)
    return re.compile(
        rf"(?P<name>{names_alt})\s*[::]\s*(?P<val>-?\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )


class WiseScorer(BaseScorer):
    name = "wise"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-122B-A10B",
        weights_path: str | None = None,
        score_prompt_template: str | None = None,
        metric_names: list[str] | None = None,
        tensor_parallel_size: int = 4,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = 8192,
        max_tokens: int = 512,
        temperature: float = 0.2,
        dtype: str = "bfloat16",
        enforce_eager: bool = False,
        swap_space: int = 4,
        quantization: str | None = None,
        seed: int | None = None,
        max_num_seqs: int = 128,
        trust_remote_code: bool = True,
        limit_mm_per_prompt: dict[str, int] | None = None,
        extra_llm_kwargs: dict[str, Any] | None = None,
        enable_thinking: bool = False,
    ) -> None:
        from vllm import LLM, SamplingParams

        # Scoring prompt and metrics are fully configurable via YAML params.
        self._template = score_prompt_template or _DEFAULT_SCORE_TEMPLATE
        names = tuple(n.lower() for n in metric_names) if metric_names else _DEFAULT_METRIC_NAMES
        self._parse_metric_names = names
        self._score_pattern = _build_score_pattern(names)
        # WiScore is only defined for the default WISE rubric; for a custom metric
        # set we expose the parsed metrics as-is. Surface wiscore first so the
        # consumer's default sub_metric_reduce="first" trains on the headline score.
        self._compute_wiscore = names == _DEFAULT_METRIC_NAMES
        self.sub_metric_names = (("wiscore",) + names) if self._compute_wiscore else names
        self._enable_thinking = enable_thinking

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
        # temperature>0 on purpose: greedy decoding (temp=0) against this strict
        # rubric collapses to a low-scoring mode (most images score 0), giving a
        # poor RL reward signal; a small temperature restores score spread.
        # max_tokens is answer-sized — the model returns three short lines.
        self.sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        messages_batch = []
        for item in items:
            text, image = item.history[-1]
            data_url = image_to_data_url(image)
            # WISE-style: pull `explanation` from per-item metadata when
            # provided, else fall back to "None" so the template still
            # renders cleanly.
            explanation = "None"
            if item.metadata and item.metadata.get("explanation"):
                explanation = str(item.metadata["explanation"])
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            # Text first, then image — mirrors WISE's payload.
                            {
                                "type": "text",
                                "text": self._template.format(prompt=text, explanation=explanation),
                            },
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ]
            )

        outputs = self.llm.chat(
            messages=messages_batch,
            sampling_params=self.sampling,
            # Always pass enable_thinking explicitly — Qwen3.5's chat template
            # defaults to thinking=True when the kwarg is absent, so omitting it
            # would NOT disable thinking. We default it off: the rubric is
            # explicit, and a <think> chain would eat the token budget (risking a
            # truncated answer -> NaN) and slow every reward call.
            chat_template_kwargs={"enable_thinking": self._enable_thinking},
        )
        return [self._parse(o.outputs[0].text) for o in outputs]

    def _parse(self, text: str) -> dict[str, float]:
        """Extract metric scores from the VLM's text output (and derive WiScore)."""
        found: dict[str, float] = {}
        for m in self._score_pattern.finditer(text):
            # Normalise the captured name to canonical underscore form
            # ("Aesthetic Quality" -> "aesthetic_quality") so the lookup
            # below matches the metric names.
            key = re.sub(r"\s+", "_", m.group("name").strip().lower())
            found[key] = float(m.group("val"))
        metrics: dict[str, float] = {}
        for key in self._parse_metric_names:
            if key in found:
                metrics[key] = found[key]
            else:
                logger.warning("Wise scorer failed to parse %s from output: %r", key, text[:200])
                metrics[key] = math.nan
        if self._compute_wiscore:
            # wiscore first so it is the default reduction target.
            return {"wiscore": _wiscore(metrics), **metrics}
        return metrics


register("wise", WiseScorer)
