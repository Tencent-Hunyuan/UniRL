"""AgentLoop — a synchronous, environment-driven multi-turn rollout driver (LIN-492).

See ``unirl/rollout/loop/README.md``. One generic loop runs multi-turn rollout over the existing
synchronous rollout engine, threading state through the ``Sample``/``Part`` model. Each turn
forks a continuation, the engine fills it, and the **environment** decides what happens next: it
consumes the model's output (e.g. parses a tool call), returns an observation that re-enters the
chain as a mask-0 input Part, and signals ``done`` when the trajectory is complete (e.g. the model
stopped emitting tool calls). The loop itself holds **no control decision** — it is purely
mechanical: ``generate -> env.step -> observe -> repeat``, bounded by ``max_turns``.

This matches verl / slime / areal / relax: the agent loop is driven by the model's tool call, not
a fixed plan. Fixed multi-stage pipelines (e.g. the AR→diffusion ``ComposedRolloutEngine``) are
*not* agent loops and do not use this class.

Prototype scope: synchronous only (the engine's ``generate`` is a blocking call).
"""

from __future__ import annotations

from unirl.rollout.loop.engine_port import RolloutEnginePort
from unirl.rollout.loop.environment import Environment
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt


class AgentLoop:
    """One generic, SYNCHRONOUS, environment-driven multi-turn rollout loop.

    The loop owns only mechanical control flow — not generation (engine), world dynamics or
    termination (environment), scoring, or training. Behaviour is set by:

    - ``environment`` — **the driver** (required). Each turn its :meth:`Environment.step` consumes
      the model's latest output and returns ``(observation, done, info)``; ``done`` ends the
      trajectory (the tool-call-driven stop condition). ``observation`` re-enters as a mask-0 Part.
    - ``sampling_params`` — one fixed config used every turn. The GRPO fan-out ``n`` is read from it
      for turn 0 (:func:`total_samples_per_prompt`); continuations fork one sample each.
    - ``max_turns`` — hard safety bound on turns.
    """

    def __init__(
        self,
        environment: Environment,
        sampling_params: BaseSamplingParams,
        max_turns: int = 8,
    ) -> None:
        self.environment = environment
        self.sampling_params = sampling_params
        self.max_turns = max_turns

    def run(self, engine: RolloutEnginePort, request: Sample) -> Sample:
        """Drive the episode: ``fork -> generate -> env.step -> observe`` until ``done`` / ``max_turns``."""
        sample = self.environment.reset(request)
        branch = total_samples_per_prompt(self.sampling_params)  # GRPO fan-out on the first turn
        for _ in range(self.max_turns):
            sample = engine.generate(sample.fork(branch, sampling_params=self.sampling_params))
            observation, done, _info = self.environment.step(sample)  # the environment decides termination
            if done:
                break
            if observation is not None:
                sample = sample.observe(observation)
            branch = 1  # continuations are one sample each
        return sample


__all__ = ["AgentLoop"]
