"""The harness boundary: what a task-internal control flow may see and return.

Kept deliberately narrow. The vocabulary test is the guard: nothing here may
speak trainer words (``rollout_id`` / ``sync_interval`` / rewards scoring /
GRPO grouping / checkpoint cadence) or deployment words (wake/sleep/offload —
engine residency is the hosting runtime's job, applied at ``generate``
boundaries per deployment, so the same harness runs colocate, separate, or
trainside unchanged).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Protocol

if TYPE_CHECKING:
    from unirl.types.sample import Sample


@dataclass(frozen=True)
class HarnessOutcome:
    """What one task run produced.

    ``completed`` — the trajectory is terminal; ``sample`` is the full trace.
    ``suspended`` — cooperatively stopped at a harness-chosen safe point;
      ``sample`` is the checkpoint. Whether it can be RESUBMITTED depends on
      the environment, not this contract: a stateless env (tools derive state
      from the Sample) resumes; a stateful env (own episodes/sessions, torn
      down by ``close``) cannot, and its recipes must pair with
      ``tail_policy: drop``. Config-time validation of that pairing is
      deferred to the config-selected-harness step.
    ``failed`` — the task hit an infrastructure fault (backend outage, tool
      timeout, context overflow); ``sample`` is the partial trace. The hosting
      runtime marks it so a fault never enters advantage math as a legitimate
      low-scoring sibling.

    ``env_reward`` — an env-sourced per-trajectory return, when the task's
    environment emits one (``info["reward"]``; last value wins). Captured here
    because reading it is task semantics; ATTACHING it to the trace is tensor
    work the hosting runtime does, keeping harness modules torch-free.
    """

    sample: "Sample"
    status: Literal["completed", "suspended", "failed"]
    env_reward: Optional[float] = None


@dataclass(frozen=True)
class HarnessContext:
    """The runtime surface a harness runs against: named engines + a stop probe.

    Engine names ("policy", "ar", "diffusion", ...) are the hosting runtime's
    local dependency names from its config — not a global registry; the
    harness file states which names it uses, so the call graph stays readable.
    """

    engines: Mapping[str, Callable[["Sample"], "Sample"]] = field(default_factory=dict)
    suspend: Callable[[], bool] = lambda: False

    def generate(self, engine: str, sample: "Sample") -> "Sample":
        """One blocking model call on the named engine."""
        try:
            gen = self.engines[engine]
        except KeyError:
            raise KeyError(
                f"harness asked for engine {engine!r}; this runtime provides {sorted(self.engines)}"
            ) from None
        return gen(sample)

    def suspend_requested(self) -> bool:
        """True once the runtime wants a cooperative stop (quiesce / weight sync).

        The harness checks this at ITS safe points — between turns, between
        stages — and returns a ``suspended`` outcome; the runtime never
        interrupts mid-generation.
        """
        return self.suspend()


class RolloutHarness(Protocol):
    """One task's internal control flow, hosted by a rollout-worker runtime.

    ``run`` must not raise for task-level faults — catch and return a
    ``failed`` outcome so the partial trace survives; the hosting runtime
    keeps a last-resort net for harness bugs. Instances are shared across
    worker threads: keep all per-task state local to ``run``.
    """

    def run(self, request: "Sample", context: HarnessContext) -> HarnessOutcome: ...


__all__ = ["HarnessContext", "HarnessOutcome", "RolloutHarness"]
