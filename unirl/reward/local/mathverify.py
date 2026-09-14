r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, List, Tuple

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

logger = logging.getLogger(__name__)

_VERIFY_TIMEOUT_S = int(float(os.environ.get("UNIRL_MATHVERIFY_TIMEOUT_S", "10")))


def _grade_all(conn: Any, jobs: List[Tuple[str, str]], seconds: int) -> None:
    """Grade every job and send the verdicts back; runs in the child, never the caller."""
    from math_verify import parse, verify

    verdicts: List[bool] = []
    for gold, prediction in jobs:
        try:
            verdicts.append(
                bool(
                    verify(
                        parse("\\boxed{" + gold + "}", parsing_timeout=seconds),
                        parse(prediction, parsing_timeout=seconds),
                        timeout_seconds=seconds,
                    )
                )
            )
        except Exception:
            verdicts.append(False)
    conn.send(verdicts)


def _grade_in_child(jobs: List[Tuple[str, str]], *, seconds: int) -> List[bool]:
    """Grade a batch in one child process, scoring all of it False if it never answers."""
    import multiprocessing
    from multiprocessing.connection import wait

    ctx = multiprocessing.get_context("forkserver")  # not fork — see ../README.md
    ctx.set_forkserver_preload(["math_verify"])
    receiver, sender = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_grade_all, args=(sender, jobs, seconds), daemon=True)
    try:
        proc.start()  # inside the try: a child dying here raises out of start()
        # sentinel as well as the pipe, so a child that dies costs a round-trip, not the budget.
        if receiver in wait([receiver, proc.sentinel], timeout=seconds * len(jobs) + 60):
            return list(receiver.recv())
        logger.warning("math-verify: %d grades did not return, scored 0.0", len(jobs))
    except Exception as exc:
        logger.warning("math-verify: grader failed (%r), scored 0.0", exc)
    finally:
        if proc.pid is not None:
            proc.kill()
            proc.join(timeout=1.0)
        receiver.close()
        sender.close()
    return [False] * len(jobs)


class MathVerifyRewardScorer(LocalRewardBackend):
    r"""Numeric/symbolic reward via HuggingFace ``math-verify`` (1.0 match / 0.0)."""

    canonical_model_name = "math_verify"
    input_kind = "text"

    def __init__(self, *, config: "MathVerifySpec", base_device: str) -> None:
        del base_device
        super().__init__()

    def _load_model(self) -> None:
        self.model = "math_verify"

    def _compute_model_rewards(self, request: RewardRequest) -> List[float]:
        generated = request.texts
        if generated is None:
            raise ValueError("MathVerifyRewardScorer requires request.texts (generated answers).")
        metadata_list = request.metadata or [None] * len(generated)
        rewards = [0.0] * len(generated)
        jobs: List[Tuple[str, str]] = []
        slots: List[int] = []
        for i, (text, meta) in enumerate(zip(generated, metadata_list)):
            if meta is not None and "answer" in meta:
                slots.append(i)
                jobs.append((str(meta["answer"]).strip(), text or ""))
        if not jobs:
            return rewards
        for i, ok in zip(slots, _grade_in_child(jobs, seconds=_VERIFY_TIMEOUT_S)):
            rewards[i] = 1.0 if ok else 0.0
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
