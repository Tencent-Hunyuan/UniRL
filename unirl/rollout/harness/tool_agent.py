"""ToolAgentHarness — the environment-driven multi-turn agent loop (LIN-492/531).

The task-semantics half of what ``AgenticRolloutEngine._run_one`` used to
inline (and the successor of the deleted ``AgentLoop`` prototype): each
turn forks a one-sample continuation, the ``"policy"`` engine fills it, and
the ENVIRONMENT decides what happens next — it parses the model's output
(e.g. a tool call), returns an observation that re-enters the chain as a
mask-0 input Part, and signals ``done``. The loop holds no other control
decision; queueing, concurrency, buffers, and abort belong to the hosting
runtime.

Resume-aware: ``turns_done = len(request.gen_parts())``, so a carried partial
continues from where it stopped (``env.reset`` is idempotent/turn-derived).
Suspension is checked at the top of each turn — the in-flight turn always
finishes naturally first (turn boundary). Whether a suspended trajectory can
actually RESUME is the env's property, not this loop's: stateless tool envs
re-derive state from the Sample; stateful envs (ALFWorld episodes, persistent
sessions) are torn down by ``close`` on suspension, so their recipes pair with
``tail_policy: drop``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from unirl.rollout.harness.protocol import HarnessContext, HarnessOutcome

if TYPE_CHECKING:
    from unirl.rollout.env.protocol import Environment
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class ToolAgentHarness:
    """``generate -> env.step -> observe`` until ``done`` / ``max_turns``, on the ``"policy"`` engine.

    ``env`` must be re-entrant (one shared instance serves concurrent
    trajectories on their own threads); ``sampling`` is only READ each turn
    (``fork`` builds a fresh gen Part per call), so sharing this harness
    across worker threads is safe as long as ``env.step`` is.
    """

    ENGINE = "policy"

    def __init__(self, *, env: "Environment", sampling: Any, max_turns: int) -> None:
        self.env = env
        self.sampling = sampling
        self.max_turns = int(max_turns)

    def run(self, request: "Sample", context: HarnessContext) -> HarnessOutcome:
        sample = request
        env_reward: Optional[float] = None
        try:
            sample = self.env.reset(request)
            turns_done = len(sample.gen_parts())
            for _ in range(self.max_turns - turns_done):
                if context.suspend_requested():
                    return HarnessOutcome(sample, "suspended", env_reward)
                sample = context.generate(self.ENGINE, sample.fork(1, sampling_params=self.sampling))
                observation, done, info = self.env.step(sample)
                if isinstance(info, dict) and info.get("reward") is not None:
                    env_reward = float(info["reward"])
                if done:
                    return HarnessOutcome(sample, "completed", env_reward)
                if observation is not None:
                    sample = sample.observe(observation)
            return HarnessOutcome(sample, "completed", env_reward)
        except Exception as exc:  # noqa: BLE001 — isolate: one bad trajectory must not sink the drain
            logger.warning("ToolAgentHarness: trajectory failed: %s", exc, exc_info=True)
            return HarnessOutcome(sample, "failed")
        finally:
            close = getattr(self.env, "close", None)
            if close is not None:
                try:
                    close(sample)
                except Exception:  # noqa: BLE001 — teardown must not sink the drain
                    logger.warning("ToolAgentHarness: env.close failed during teardown", exc_info=True)


__all__ = ["ToolAgentHarness"]
