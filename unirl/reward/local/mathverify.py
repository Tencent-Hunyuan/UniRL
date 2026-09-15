r"""math-verify reward scorer — the paper's grader (HuggingFace Math-Verify)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Tuple

from unirl.reward.base import BaseRewardComponentSpec
from unirl.types.reward import RewardRequest

from .base import LocalRewardBackend

_VERIFY_TIMEOUT_S = int(os.environ.get("UNIRL_MATHVERIFY_TIMEOUT_S", "10"))
if _VERIFY_TIMEOUT_S <= 0:
    raise ValueError("UNIRL_MATHVERIFY_TIMEOUT_S must be a positive integer")


class MathVerifyRewardScorer(LocalRewardBackend):
    r"""Numeric/symbolic reward via HuggingFace ``math-verify`` (1.0 match / 0.0)."""

    canonical_model_name = "math_verify"
    input_kind = "text"

    @staticmethod
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

    @staticmethod
    def _grade_in_child(jobs: List[Tuple[str, str]], *, seconds: int) -> List[bool]:
        """Grade a batch in one child process, raising if the child never answers."""
        import multiprocessing
        from multiprocessing.connection import wait

        ctx = multiprocessing.get_context("forkserver")  # not fork — see ../README.md
        ctx.set_forkserver_preload(["math_verify", __name__])
        receiver, sender = ctx.Pipe(duplex=False)
        proc = ctx.Process(
            target=MathVerifyRewardScorer._grade_all,
            args=(sender, jobs, seconds),
            daemon=True,
        )
        try:
            proc.start()  # inside the try: a child dying here raises out of start()
            # sentinel as well as the pipe, so a child that dies costs a round-trip, not the budget.
            # Each job has two parse windows and at least one verification window.
            deadline_s = 3 * seconds * len(jobs) + 60
            ready = wait([receiver, proc.sentinel], timeout=deadline_s)
            if receiver in ready:
                verdicts = receiver.recv()
                if len(verdicts) != len(jobs):
                    raise RuntimeError(f"math-verify returned {len(verdicts)} verdicts for {len(jobs)} jobs")
                return verdicts
            if proc.sentinel in ready:
                proc.join(timeout=0)
                raise RuntimeError(f"math-verify child exited with code {proc.exitcode} before returning verdicts")
            raise TimeoutError(f"math-verify child did not return {len(jobs)} verdicts within {deadline_s}s")
        finally:
            if proc.pid is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=1.0)
            receiver.close()
            sender.close()

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
        for i, ok in zip(slots, self._grade_in_child(jobs, seconds=_VERIFY_TIMEOUT_S)):
            rewards[i] = 1.0 if ok else 0.0
        return rewards


@dataclass
class MathVerifySpec(BaseRewardComponentSpec):
    r"""Config for the math-verify scorer."""
