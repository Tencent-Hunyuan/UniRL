"""Scoring: a minimal reward-service HTTP client + local text graders."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests


class RewardServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.trust_env = False

    def check(self, rewards: Sequence[str]) -> None:
        try:
            served = self.session.get(f"{self.base_url}/rewards", timeout=30).json()["rewards"]
        except requests.RequestException as exc:
            raise SystemExit(f"reward service at {self.base_url} is unreachable: {exc}") from exc
        missing = [r for r in rewards if r not in served]
        if missing:
            raise SystemExit(f"reward service at {self.base_url} does not serve {missing} (served: {served})")

    def score_images(
        self,
        pairs: Sequence[Tuple[str, Path]],
        rewards: Sequence[str],
        chunk: int = 8,
        metadatas: Optional[Sequence[Optional[Dict]]] = None,
    ) -> Tuple[List[Optional[Dict[str, float]]], int]:
        """Score (prompt, image path) pairs."""
        rows: List[Optional[Dict[str, float]]] = []
        n_errors = 0
        for i in range(0, len(pairs), chunk):
            batch = pairs[i : i + chunk]
            reqs = [
                {
                    "history": [{"text": prompt, "image_b64": base64.b64encode(path.read_bytes()).decode("ascii")}],
                    "required_rewards": list(rewards),
                }
                for prompt, path in batch
            ]
            if metadatas is not None:
                for req, metadata in zip(reqs, metadatas[i : i + chunk]):
                    if metadata is not None:
                        req["metadata"] = metadata
            resp = self.session.post(f"{self.base_url}/score", json={"requests": reqs}, timeout=3600)
            resp.raise_for_status()
            data = resp.json()
            errors = data.get("errors") or [{} for _ in batch]
            for result, err in zip(data["results"], errors):
                n_errors += len(err)
                flat = {f"{reward}/{sub}": value for reward, subs in result.items() for sub, value in subs.items()}
                rows.append(flat or None)
            print(f"[score] {min(i + chunk, len(pairs))}/{len(pairs)}", flush=True)
        return rows, n_errors


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


def _geneval2_answer_forms(question: str, expected: str) -> Tuple[str, ...]:
    """Expected-answer surface forms scored at the first token (GenEval2 Soft-TIFA convention)."""
    expected = (expected or "").strip()
    if question.startswith("How many"):
        forms = [expected, expected.capitalize(), " " + expected, " " + expected.capitalize()]
        digit = _WORD_TO_DIGIT.get(expected.lower())
        if digit is not None:
            forms += [digit, " " + digit]
        return tuple(forms)
    return _YES_SURFACE_FORMS


def _geneval2_answer_token_ids(tokenizer, question: str, expected: str) -> List[int]:
    """First answer-token ids without BOS/EOS injection or duplicate probability mass."""
    token_ids: List[int] = []
    seen: set[int] = set()
    for form in _geneval2_answer_forms(question, expected):
        encoded = tokenizer.encode(form, add_special_tokens=False)
        if not encoded:
            continue
        token_id = int(encoded[0])
        if token_id not in seen:
            seen.add(token_id)
            token_ids.append(token_id)
    return token_ids


def score_images_local_geneval2(
    pairs: Sequence[Tuple[str, Path]],
    metadatas: Optional[Sequence[Optional[Dict]]] = None,
    *,
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    aggregation: str = "gm",
    device: str = "cuda",
) -> Tuple[List[Optional[Dict[str, float]]], int]:
    """Score (prompt, image) pairs with a self-contained transformers Qwen3-VL Soft-TIFA scorer."""
    import math

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dev = "cuda" if device in ("auto", "") else device
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto").to(dev)
    model.eval()

    def _answer_prob(question: str, image: "Image.Image", token_ids: List[int]) -> float:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"{question} Answer in one word."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(dev)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True
            )
        probs = torch.nn.functional.softmax(out.scores[0], dim=-1)
        return sum(probs[0, tid].item() for tid in token_ids)

    def _soft_tifa(vqa_list: List, image: "Image.Image") -> float:
        scores: List[float] = []
        for q, a in ((v[0], v[1]) for v in vqa_list):
            tids = _geneval2_answer_token_ids(processor.tokenizer, q, a)
            if tids:
                scores.append(_answer_prob(q, image, tids))
        if not scores:
            return 0.0
        if aggregation == "gm":
            scores = [max(s, 1e-300) for s in scores]
            return float(math.exp(sum(math.log(s) for s in scores) / len(scores)))
        return sum(scores) / len(scores)

    rows: List[Optional[Dict[str, float]]] = []
    n_errors = 0
    for i, (prompt, path) in enumerate(pairs):
        md = metadatas[i] if metadatas is not None and i < len(metadatas) else None
        vqa = md.get("vqa_list") if isinstance(md, dict) else None
        if not isinstance(vqa, list) or not vqa:
            rows.append(None)
            n_errors += 1
            continue
        rows.append({"geneval2/vqascore": float(_soft_tifa(vqa, Image.open(path).convert("RGB")))})
        if (i + 1) % 50 == 0:
            print(f"[score-local] {i + 1}/{len(pairs)}", flush=True)
    return rows, n_errors


_CANARY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAYElEQVR4nO3PQQ0AIBDAMMC/50MEj4ZkVbDtmVk/OzrgVQNaA1oDWgNa"
    "A1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgPaBXKqA31N0fbGAAAAAElFTkSuQmCC"
)
_CANARY_REQUEST = {
    "history": [{"text": "a plain white background", "image_b64": _CANARY_PNG_B64}],
    "required_rewards": ["geneval2"],
    "metadata": {"vqa_list": [["Is there a red double-decker bus in the image?", "Yes"]]},
}


def check_geneval2_metadata(client: RewardServiceClient) -> None:
    """SystemExit unless the service's geneval2 scorer consumes per-request ``metadata.vqa_list``."""
    resp = client.session.post(f"{client.base_url}/score", json={"requests": [_CANARY_REQUEST]}, timeout=600)
    resp.raise_for_status()
    score = (resp.json()["results"][0].get("geneval2") or {}).get("vqascore")
    if score is None or score > 0.5:
        raise SystemExit(
            f"the geneval2 scorer at {client.base_url} ignores request-metadata vqa_lists "
            f"(a deliberately false canary question scored {score!r}, expected ~0), so it would "
            "score this benchmark with its generic single-question template instead of GenEval2 "
            "Soft-TIFA. Deploy a geneval2 scorer with request-metadata support (RewardService "
            "geneval2 reading metadata['vqa_list'])."
        )


_BOXED_LETTER = re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}")
_ANSWER_IS = re.compile(r"(?i)answer\s*(?:is)?\s*:?\s*\(?([A-D])\)?")


def grade_math(answer: str, responses: List[str]) -> float:
    """avg@k exact-math correctness via HuggingFace math-verify, with the gold wrapped so it parses."""
    from math_verify import parse, verify  # lazy: pip install math-verify

    gold = parse("\\boxed{" + str(answer).strip() + "}")
    correct = 0
    for resp in responses:
        try:
            correct += bool(verify(gold, parse(resp)))
        except Exception:  # noqa: BLE001 — malformed generations grade as 0, like the trainer scorer
            pass
    return correct / len(responses)


def grade_mc(answer: str, responses: List[str]) -> float:
    """avg@k multiple-choice accuracy; the last stated letter wins."""
    correct = 0
    for resp in responses:
        found = _BOXED_LETTER.findall(resp) or _ANSWER_IS.findall(resp)
        correct += bool(found) and found[-1].upper() == answer.upper()
    return correct / len(responses)


GRADERS = {"math_verify": grade_math, "mc_letter": grade_mc}
