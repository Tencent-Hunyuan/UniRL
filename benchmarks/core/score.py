"""Scoring: a minimal reward-service HTTP client + local text graders.

Image/video rewards go through the ``unirl-reward-service`` ``POST /score``
endpoint (see ``unirl-reward-service/reward_service/schemas.py``):
``results[i][reward][sub_metric] -> float``; per-request failures land in
``errors[i][reward]``. Text grading is local (math-verify / letter match).
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import requests


class RewardServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        # Reward services live on the internal network; corporate proxy env vars
        # would 503 them (same rationale as unirl/reward/remote.py).
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
        """Score (prompt, image path) pairs. Returns per-pair flat dicts keyed
        ``"<reward>/<sub_metric>"`` (None when every reward errored) and the
        total error count. ``metadatas`` (aligned with ``pairs``) rides along as
        ``RewardRequest.metadata`` — the geneval scorers require it."""
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


# geneval2 metadata canary: a 64x64 white PNG plus one deliberately false VQA
# question about it. A scorer that honors request-metadata vqa_lists (Soft-TIFA)
# scores ~0; a pre-metadata scorer ignores it and answers its generic
# "does the image match the prompt" template on the trivially true prompt (~1).
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
    """SystemExit unless the service's geneval2 scorer consumes per-request
    ``metadata.vqa_list``. A scorer without metadata support still answers —
    with its degenerate single-question template — so a whole run would come
    back with plausible-looking numbers that are not GenEval2 Soft-TIFA."""
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


# GPQA's official baseline parses "The correct answer is (X)"; we also accept \boxed{X}.
_BOXED_LETTER = re.compile(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}")
_ANSWER_IS = re.compile(r"(?i)answer\s*(?:is)?\s*:?\s*\(?([A-D])\)?")


def grade_math(answer: str, responses: List[str]) -> float:
    """avg@k exact-math correctness via HuggingFace math-verify (same grader as
    ``unirl.reward.local.mathverify``, including the ``\\boxed{}``-wrapped gold —
    bare golds like ``\\left(3, \\frac{\\pi}{2}\\right)`` mis-parse otherwise)."""
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
