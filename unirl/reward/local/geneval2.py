"""GenEval2 Soft-TIFA reward scorer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

from unirl.reward.base import PromptRewardComponentSpec
from unirl.reward.local.device import resolve_device
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

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

_YES_SURFACE_FORMS = ("Yes", "yes", " Yes", " yes")


def _answer_surface_forms(question: str, expected: str) -> tuple[str, ...]:
    """Surface forms of the expected answer to score at the first token."""
    expected = ("" if expected is None else str(expected)).strip()
    if question.startswith("How many"):
        forms = [expected, expected.capitalize(), " " + expected, " " + expected.capitalize()]
        digit = _WORD_TO_DIGIT.get(expected.lower())
        if digit is not None:
            forms += [digit, " " + digit]
        return tuple(forms)

    if expected.lower() != "yes":
        raise ValueError(f"GenEval2 non-counting atom expects 'Yes', got {expected!r} for question {question!r}")
    return _YES_SURFACE_FORMS


class GenEval2RewardScorer(LocalRewardBackend):
    """GenEval2 Soft-TIFA compositional reward using Qwen3-VL VQA."""

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
            prompt_source=config.prompt_source,
        )

    def _load_model(self) -> None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self._model_name_or_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self._model_name_or_path, torch_dtype="auto")
        self.model.to(self.device)
        self.model.eval()

        self._prompt_to_vqa: Optional[Dict[str, List[Any]]] = None
        if self._data_path:
            p = Path(self._data_path).expanduser()
            if p.is_file():
                data_paths = [p]
            elif p.is_dir():
                data_paths = [p / name for name in ("train.jsonl", "test.jsonl") if (p / name).is_file()]
            else:
                data_paths = []
            if data_paths:
                out: Dict[str, List[Any]] = {}
                for path in data_paths:
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
                self._prompt_to_vqa = out

    def _soft_tifa_score(
        self,
        vqa_list: List[List[str]],
        image: Image.Image,
    ) -> float:
        """Compute Soft-TIFA score for one image given its VQA list."""
        score_list: List[float] = []
        for vqa in vqa_list:
            question, answer = vqa[0], vqa[1]
            token_ids: List[int] = []
            for form in _answer_surface_forms(question, answer):
                encoded = self.processor.tokenizer.encode(form)
                if encoded:
                    token_ids.append(encoded[0])
            if not token_ids:
                continue

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
            ).to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
            score_list.append(float(probs[0, token_ids].sum().item()))

        if not score_list:
            return 0.0

        if self._aggregation == "gm":
            return float(math.exp(sum(math.log(max(x, 1e-300)) for x in score_list) / len(score_list)))
        return sum(score_list) / len(score_list)

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        images = request.images
        prompts = self.prompts(request)
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
class GenEval2Spec(PromptRewardComponentSpec):
    """Typed config for the GenEval2 Soft-TIFA reward component."""

    batch_size: int = 1
    device: str = "auto"
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    data_path: str = ""
    aggregation: str = "gm"
