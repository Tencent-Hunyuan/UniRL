"""GenEval2 Soft-TIFA reward scorer.

Computes compositional generation quality using Qwen3-VL VQA scoring.
For each image, runs per-atom visual question answering and aggregates
answer probabilities via geometric mean (matching GenEval2 benchmark).

Reference: https://github.com/facebookresearch/GenEval2
Adapted from flow_factory/rewards/geneval2_soft_tifa.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from unirl.reward.base import BaseRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

try:
    from scipy.stats import gmean as _scipy_gmean

    def _gmean_scores(xs: List[float]) -> float:
        xs = [max(x, 1e-300) for x in xs]
        return float(_scipy_gmean(xs))

except ImportError:

    def _gmean_scores(xs: List[float]) -> float:
        xs = [max(x, 1e-300) for x in xs]
        return float(math.exp(sum(math.log(x) for x in xs) / len(xs)))


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

# Non-counting GenEval2 atoms expect "Yes" (matches remote reward-service scorer).
_YES_SURFACE_FORMS = ("Yes", "yes", " Yes", " yes")


def _answer_surface_forms(question: str, expected: str) -> tuple[str, ...]:
    """Surface forms of the expected answer to score at the first token.

    Mirrors GenEval2 ``soft_tifa()`` and ``unirl-reward-service`` geneval2 scorer:
    counting atoms ("How many ...?") score the expected number word and its digit;
    every other atom in the benchmark expects "Yes".
    """
    expected = (expected or "").strip()
    if question.startswith("How many"):
        forms = [expected, expected.capitalize(), " " + expected, " " + expected.capitalize()]
        digit = _WORD_TO_DIGIT.get(expected.lower())
        if digit is not None:
            forms += [digit, " " + digit]
        return tuple(forms)

    if expected.lower() != "yes":
        raise ValueError(f"GenEval2 non-counting atom expects 'Yes', got {expected!r} for question {question!r}")
    return _YES_SURFACE_FORMS


def _load_prompt_vqa_map(paths: List[Path]) -> Dict[str, List[Any]]:
    """Build prompt -> vqa_list mapping from GenEval2-style JSONL files."""
    out: Dict[str, List[Any]] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"GenEval2 benchmark JSONL not found: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompt = obj.get("prompt")
                vqa = obj.get("vqa_list")
                if prompt is not None and vqa is not None:
                    out[prompt] = vqa
    return out


def _resolve_data_paths(data_path: str) -> List[Path]:
    """Resolve data_path to a list of JSONL file paths."""
    if not data_path:
        return []
    p = Path(data_path).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        paths = []
        for name in ("train.jsonl", "test.jsonl"):
            child = p / name
            if child.is_file():
                paths.append(child)
        return paths
    return []


class GenEval2RewardScorer(LocalRewardBackend):
    """GenEval2 Soft-TIFA compositional reward using Qwen3-VL VQA.

    For each image, asks VQA questions from the GenEval2 benchmark and
    computes the soft probability of the correct answer. Scores are
    aggregated across atoms using geometric mean (default) or arithmetic mean.

    This scorer is slow (one VLM generation per VQA atom per image) but
    provides high-quality compositional evaluation.
    """

    canonical_model_name = "geneval2"

    def __init__(self, *, config: "GenEval2Spec", base_device: str) -> None:
        self._model_name_or_path = config.model_name
        self._aggregation = config.aggregation.lower()
        self._data_path = config.data_path
        if self._aggregation not in ("am", "gm"):
            raise ValueError(f"aggregation must be 'am' or 'gm', got {self._aggregation!r}")

        super().__init__(
            device=resolve_device(config.device, base_device),
            batch_size=config.batch_size,
        )

    def _load_model(self) -> None:
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError:
            raise ImportError("transformers>=4.40 is required for GenEval2 reward (Qwen3-VL support)")

        self.processor = AutoProcessor.from_pretrained(self._model_name_or_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self._model_name_or_path, torch_dtype="auto")
        self.model.to(self.device)
        self.model.eval()

        data_paths = _resolve_data_paths(self._data_path)
        self._prompt_to_vqa: Optional[Dict[str, List[Any]]] = None
        if data_paths:
            self._prompt_to_vqa = _load_prompt_vqa_map(data_paths)

    def _get_answer_probability(
        self,
        question: str,
        image: Image.Image,
        answer_token_ids: List[int],
    ) -> float:
        """Ask a VQA question and return the soft probability of the answer."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question} Answer in one word."},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        # Sum probabilities for all valid answer token IDs
        return sum(probs[0, tid].item() for tid in answer_token_ids)

    def _answer_token_ids(self, question: str, expected: str) -> List[int]:
        """Map a VQA atom to first-token IDs for its expected answer surface forms."""
        if self.processor is None:
            raise RuntimeError("GenEval2RewardScorer._answer_token_ids called before processor load")
        token_ids: List[int] = []
        for form in _answer_surface_forms(question, expected):
            encoded = self.processor.tokenizer.encode(form)
            if encoded:
                token_ids.append(encoded[0])
        return token_ids

    def _soft_tifa_score(
        self,
        vqa_list: List[List[str]],
        image: Image.Image,
    ) -> float:
        """Compute Soft-TIFA score for one image given its VQA list."""
        score_list: List[float] = []
        for vqa in vqa_list:
            question, answer = vqa[0], vqa[1]
            token_ids = self._answer_token_ids(question, answer)
            if not token_ids:
                continue
            prob = self._get_answer_probability(question, image, token_ids)
            score_list.append(prob)

        if not score_list:
            return 0.0

        if self._aggregation == "gm":
            return _gmean_scores(score_list)
        return sum(score_list) / len(score_list)

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = request.prompts
        if images is None:
            raise ValueError("GenEval2RewardScorer requires generated images in RewardRequest.generated['image']")
        if len(images) != len(prompts):
            raise ValueError(
                f"GenEval2RewardScorer expected equal image/prompt counts, "
                f"got images={len(images)} prompts={len(prompts)}"
            )

        rewards: List[float] = []
        for i in range(len(prompts)):
            vqa_list = None

            if request.metadata and i < len(request.metadata) and request.metadata[i]:
                vqa_list = request.metadata[i].get("vqa_list")

            if vqa_list is None and self._prompt_to_vqa is not None:
                vqa_list = self._prompt_to_vqa.get(prompts[i])

            if vqa_list is None:
                rewards.append(0.0)
                continue

            score = self._soft_tifa_score(vqa_list, images[i])
            rewards.append(score)

        return rewards


@dataclass
class GenEval2Spec(BaseRewardComponentSpec):
    """Typed config for the GenEval2 Soft-TIFA reward component.

    Args:
        batch_size: Batch size for reward computation (kept at 1 due to
            sequential VQA per image).
        device: Device for the Qwen3-VL model ("auto", "cuda", "cuda:0", etc).
        model_name: Hugging Face model ID for the VLM (default Qwen3-VL-8B).
        data_path: Path to GenEval2 JSONL file or directory containing
            train.jsonl/test.jsonl. Used for prompt -> vqa_list lookup.
        aggregation: Score aggregation across VQA atoms: "gm" (geometric mean,
            default, matching GenEval2 benchmark) or "am" (arithmetic mean).
    """

    batch_size: int = 1
    device: str = "auto"
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    data_path: str = ""
    aggregation: str = "gm"
