"""GenEval2 scorer (Soft-TIFA / VQAScore) via in-process vLLM.

Reproduces the scoring of **GenEval 2** (facebookresearch/GenEval2, arXiv
2512.16853), whose headline metric is **Soft-TIFA** with a Qwen3-VL VQA backbone.

Two modes:

1. **Soft-TIFA** (a per-prompt ``vqa_list`` is available — see "vqa_list source"
   below): for each ``(question, expected_answer)`` atom we ask the VQA model
   ``"{question} Answer in one word."`` and score the probability mass the model
   puts on the **expected answer** at the first generated token. A counting atom
   ("How many ...?" → e.g. "seven") scores the number word *and* its digit; every
   other atom is yes/no with expected "Yes". The per-prompt score is the
   arithmetic mean (``aggregate="am"``) or geometric mean (``aggregate="gm"``,
   which drives the score to 0 if any atom is unsatisfied) over atoms — matching
   GenEval2's ``soft_tifa_am`` / ``soft_tifa_gm``.

2. **VQAScore / degenerate** (no ``vqa_list``): ask
   ``Does this image show "<prompt>"? Answer ... Yes or No.`` and take the "Yes"
   probability. This is VQAScore (arXiv 2404.01291) and matches GenEval2's
   ``vqa_score``.

This deliberately scores the *expected* answer per atom. An earlier version asked
every atom as a yes/no question and scored P("Yes") regardless of the expected
answer, which turned counting atoms ("How many ...? -> seven") into noise.

vqa_list source (Soft-TIFA): ``ScoreItem.metadata["vqa_list"]`` is preferred (the
training side attaches each prompt's atoms, robust to prompt rewording); a
``dataset_path`` JSONL keyed by exact prompt text is the fallback. If a
``dataset_path`` is configured but neither source yields a list, the scorer fails
closed (``allow_fallback=False``) instead of silently scoring the weaker VQAScore
proxy. With no ``dataset_path`` and no metadata, it is pure VQAScore.

vqa_list entry format: ``[question, expected_answer]``, e.g.
``["How many backpacks are in the image?", "one"]``.

Fidelity note: GenEval2 reads the full-vocab softmax; for batched-rollout
throughput this scorer reads vLLM's top-``logprobs_topk`` logprobs, so an expected
answer outside the top-k contributes 0. The expected answer carries high
probability when the image is faithful and low probability otherwise, so this is
directionally faithful for a reward; raise ``logprobs_topk`` for closer agreement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
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

# VQAScore single-question template (no vqa_list); matches GenEval2 vqa_score().
_VQA_TEMPLATE = 'Does this image show "{prompt}"? Answer the question with Yes or No.'
# Soft-TIFA per-atom template; matches GenEval2 soft_tifa() ("Answer in one word.").
_VQA_QUESTION_TEMPLATE = "{question} Answer in one word."
_YES_SURFACE_FORMS = ("Yes", "yes", " Yes", " yes")
# Number word -> digit, mirroring GenEval2 return_numeric_string(): a counting
# atom scores the probability of either spelling of the expected count.
_WORD_TO_DIGIT = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def _answer_surface_forms(question: str, expected: str) -> tuple[str, ...]:
    """Surface forms of the *expected* answer to score at the first token.

    Mirrors GenEval2 ``soft_tifa()``: a "How many ...?" atom scores the expected
    number word and its digit (each plain, capitalised, and with a leading
    space); every other atom is yes/no with expected "Yes".
    """
    expected = (expected or "").strip()
    if question.startswith("How many"):
        forms = [expected, expected.capitalize(), " " + expected, " " + expected.capitalize()]
        digit = _WORD_TO_DIGIT.get(expected.lower())
        if digit is not None:
            forms += [digit, " " + digit]
        return tuple(forms)
    return _YES_SURFACE_FORMS


def _load_vqa_dataset(dataset_path: str) -> dict[str, list[list[str]]]:
    """Load a GenEval2 JSONL dataset and return a prompt→vqa_list mapping.

    Scans all ``*.jsonl`` files under *dataset_path* (non-recursive).
    Each line must contain at least ``prompt`` and ``vqa_list`` fields.

    Returns:
        Mapping from normalised prompt string to its VQA question list.
        Each entry in the list is ``[question, expected_answer]``.
    """
    dataset_dir = Path(dataset_path)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"GenEval2 dataset directory not found: {dataset_path}")

    prompt_to_vqa: dict[str, list[list[str]]] = {}
    jsonl_files = sorted(dataset_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {dataset_path}")

    for jsonl_file in jsonl_files:
        with open(jsonl_file, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skipping malformed JSON at %s:%d", jsonl_file.name, line_no)
                    continue
                prompt = entry.get("prompt", "").strip()
                vqa_list = entry.get("vqa_list")
                if prompt and vqa_list:
                    prompt_to_vqa[prompt] = vqa_list

    logger.info(
        "loaded GenEval2 dataset from %s: %d prompts across %d files",
        dataset_path,
        len(prompt_to_vqa),
        len(jsonl_files),
    )
    return prompt_to_vqa


class GenEval2Scorer(BaseScorer):
    name = "geneval2"
    sub_metric_names = ("vqascore",)

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        weights_path: str | None = None,
        dataset_path: str | None = None,
        aggregate: str = "am",
        allow_fallback: bool = False,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = 4096,
        logprobs_topk: int = 20,
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

        if aggregate not in ("am", "gm"):
            raise ValueError(f"geneval2 aggregate must be 'am' (arithmetic) or 'gm' (geometric), got {aggregate!r}")
        self._aggregate = aggregate
        self._allow_fallback = allow_fallback
        self._dataset_configured = dataset_path is not None

        # Load VQA dataset if provided (enables Soft-TIFA scoring via prompt lookup).
        if dataset_path is not None:
            self._prompt_to_vqa = _load_vqa_dataset(dataset_path)
        else:
            self._prompt_to_vqa: dict[str, list[list[str]]] = {}
            logger.info("no dataset_path provided; using degenerate VQAScore template unless metadata carries vqa_list")

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
        self.sampling = SamplingParams(temperature=0.0, max_tokens=1, logprobs=logprobs_topk)
        self._tokenizer = self.llm.get_tokenizer()
        self._yes_token_ids = self._resolve_answer_tokens(_YES_SURFACE_FORMS)
        if not self._yes_token_ids:
            raise RuntimeError("could not tokenize any 'Yes' surface form")

    def _resolve_answer_tokens(self, forms: tuple[str, ...]) -> frozenset[int]:
        """First-token id of each surface form (mirrors GenEval2 ``encode(form)[0]``)."""
        ids: set[int] = set()
        for form in forms:
            try:
                encoded = self._tokenizer.encode(form, add_special_tokens=False)
            except TypeError:
                encoded = self._tokenizer.encode(form)
            if encoded:
                ids.add(encoded[0])
        return frozenset(ids)

    def _lookup_vqa(self, prompt: str, item: ScoreItem) -> list | None:
        """Resolve the prompt's ``vqa_list``: metadata first, then dataset.

        Returns ``None`` (→ degenerate VQAScore) only when no dataset is
        configured. When a ``dataset_path`` is configured but no atoms are found,
        fail closed (unless ``allow_fallback``) rather than silently scoring a
        different metric.
        """
        meta = item.metadata or {}
        vqa_list = meta.get("vqa_list")
        if vqa_list:
            return vqa_list
        if self._prompt_to_vqa:
            found = self._prompt_to_vqa.get(prompt.strip())
            if found:
                return found
        if self._dataset_configured and not self._allow_fallback:
            raise ValueError(
                f"geneval2: no vqa_list for prompt {prompt[:80]!r}; pass it via "
                f"metadata['vqa_list'] or add the prompt to dataset_path. Set "
                f"allow_fallback=true to score the degenerate VQAScore template instead."
            )
        if self._prompt_to_vqa:
            logger.warning("geneval2: prompt not in dataset, using degenerate VQAScore template: %r", prompt[:80])
        return None

    @staticmethod
    def _message(data_url: str, text: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        # One VLM call per atom (Soft-TIFA) or one per item (degenerate). Each
        # message carries its own expected-answer token set so we can score
        # P(expected) per atom after a single batched generate.
        messages_batch: list[list[dict]] = []
        answer_ids_batch: list[frozenset[int]] = []
        item_question_counts: list[int] = []

        for item in items:
            text, image = item.history[-1]
            data_url = image_to_data_url(image)
            vqa_list = self._lookup_vqa(text, item)

            if vqa_list:
                for question, expected in vqa_list:
                    messages_batch.append(self._message(data_url, _VQA_QUESTION_TEMPLATE.format(question=question)))
                    answer_ids_batch.append(self._resolve_answer_tokens(_answer_surface_forms(question, expected)))
                item_question_counts.append(len(vqa_list))
            else:
                messages_batch.append(self._message(data_url, _VQA_TEMPLATE.format(prompt=text)))
                answer_ids_batch.append(self._yes_token_ids)
                item_question_counts.append(1)

        outputs = self.llm.chat(messages=messages_batch, sampling_params=self.sampling)

        results: list[dict[str, float]] = []
        idx = 0
        for count in item_question_counts:
            probs = [self._answer_prob(outputs[idx + j], answer_ids_batch[idx + j]) for j in range(count)]
            idx += count
            results.append({"vqascore": self._aggregate_probs(probs)})
        return results

    def _aggregate_probs(self, probs: list[float]) -> float:
        """Arithmetic (``am``) or geometric (``gm``) mean over atom soft-scores.

        ``gm`` mirrors ``scipy.gmean``: any non-positive atom drives the prompt
        score to 0, penalising a single unsatisfied atom harder than ``am`` does.
        """
        if not probs:
            return math.nan
        if self._aggregate == "gm":
            if min(probs) <= 0.0:
                return 0.0
            return math.exp(sum(math.log(p) for p in probs) / len(probs))
        return sum(probs) / len(probs)

    @staticmethod
    def _answer_prob(output, answer_token_ids: frozenset[int]) -> float:
        """Probability mass on the expected answer at the first generated token.

        Sums ``exp(logprob)`` over the expected-answer first-token ids that appear
        in vLLM's top-k logprobs (a top-k view of GenEval2's full-vocab softmax).
        """
        if not output.outputs:
            return math.nan
        first = output.outputs[0]
        if not first.logprobs:
            return math.nan
        token_lp = first.logprobs[0]
        prob = 0.0
        for tok_id, lp_obj in token_lp.items():
            if tok_id in answer_token_ids:
                lp_val = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
                prob += math.exp(lp_val)
        return float(prob)


register("geneval2", GenEval2Scorer)
