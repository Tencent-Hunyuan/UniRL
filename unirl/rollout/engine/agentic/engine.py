"""AgenticRolloutEngine — runs ONE agentic trajectory, in-process (LIN-522/531/693).

A ``Remote`` on a DP-replicated slab. Each worker builds its own local inner single-turn
engine and environment, and hosts a ``RolloutHarness`` over them.

Holds no scheduling decision: how many trajectories run, where, and what happens to an
interrupted one all belong to ``AgenticManager``, which calls ``run_trajectory`` once per
trajectory and ``set_stopping`` to request a turn-boundary stop. Both are un-decorated —
the caller is rank 0 with plain actor handles, not the driver with a Handle.

Lifecycle and weight-sync verbs below delegate to the inner engine.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.synchronous import SyncRolloutEngine
from unirl.rollout.harness.protocol import HarnessContext, RolloutHarness
from unirl.rollout.harness.tool_agent import ToolAgentHarness
from unirl.types.sample import Sample, _part_with_field

logger = logging.getLogger(__name__)


class AgenticRolloutEngine(Remote):
    """Runs one agentic trajectory over a local inner engine and environment."""

    _component_name = "agentic"

    def __init__(
        self,
        config: AgenticRolloutEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Optional[Any] = None,
    ) -> None:
        require(
            isinstance(config, AgenticRolloutEngineConfig),
            f"AgenticRolloutEngine requires AgenticRolloutEngineConfig; got {type(config).__name__}",
        )
        self.cfg = config
        self.rank = rank

        deps = dict(device=device, rank=rank, model_config=model_config)
        self._env = config.env  # an Environment (built per worker via its _target_); must be re-entrant
        require(self._env is not None, "AgenticRolloutEngine requires an env (config.env)")
        self._maybe_inject_tool_schemas(config.inner, self._env)
        inner = config.inner.make_engine(strategy=strategy, **deps)
        if not isinstance(inner, SyncRolloutEngine):
            shutdown = getattr(inner, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception as exc:
                    logger.warning("AgenticRolloutEngine invalid inner cleanup raised: %s", exc)
            raise ValueError(
                f"AgenticRolloutEngine inner must implement the single-turn engine contract; got {type(inner).__name__}"
            )
        self._inner: SyncRolloutEngine = inner

        self._sp = config.episode_sampling  # per-turn sampling params; carries n via samples_per_prompt
        self._max_turns = int(config.max_turns)
        _env_mt = getattr(self._env, "max_turns", None)
        require(
            _env_mt is None or int(_env_mt) == self._max_turns,
            f"env.max_turns ({_env_mt}) must equal config.max_turns ({self._max_turns}); "
            "they are set independently in the recipe and must agree.",
        )

        self._harness: RolloutHarness = ToolAgentHarness(env=self._env, sampling=self._sp, max_turns=self._max_turns)
        self._harness_ctx = HarnessContext(
            engines={"policy": self._inner.generate},
            suspend=lambda: self._stopping,
        )
        self._stopping = False

    @staticmethod
    def _maybe_inject_tool_schemas(inner_cfg: Any, env: Any) -> None:
        """Copy the env's tool JSON-schemas into the inner engine's chat-template
        kwargs so the model is told about the tools without the recipe restating
        them. No-op when the env exposes no ``tool_schemas`` or the inner config
        has no ``chat_template_kwargs``; an explicit ``tools`` entry is preserved."""
        get_schemas = getattr(env, "tool_schemas", None)
        if not callable(get_schemas) or not hasattr(inner_cfg, "chat_template_kwargs"):
            return
        ctk = dict(inner_cfg.chat_template_kwargs or {})
        ctk.setdefault("tools", get_schemas())
        inner_cfg.chat_template_kwargs = ctk

    def run_trajectory(self, task: Sample) -> Tuple[Sample, bool]:
        """One trajectory, on this drain thread: delegate to the harness. Returns ``(sample, done)``.

        ``done=True`` — terminal (``completed``/``failed`` outcome).
        ``done=False`` — **checkpointed** at a harness-chosen safe point
        (``suspended``, partial rollout): carried and resumed next drive.

        The harness owns the task-internal control flow (turn loop, stop
        conditions, teardown) and returns ``failed`` for task-level faults;
        this runtime keeps the last-resort net (a harness BUG must not sink
        the drain) and the tensor-side jobs: attaching an env-sourced reward,
        and NaN-marking failures so an infrastructure fault never enters GRPO
        as a legitimate low-scoring sibling (trainers give NaN zero advantage).
        Failure-isolated: never raises into the drain.
        """
        try:
            outcome = self._harness.run(task, self._harness_ctx)
            if outcome.status == "completed":
                return self._attach_env_reward(outcome.sample, outcome.env_reward), True
            if outcome.status == "suspended":
                return outcome.sample, False
            if outcome.status == "failed":
                return self._attach_env_reward(outcome.sample, float("nan")), True
            raise ValueError(f"unknown harness outcome status: {outcome.status!r}")
        except Exception as exc:  # noqa: BLE001 — harness bug; the partial trace is lost, the drain survives
            logger.warning("AgenticRolloutEngine: harness outcome failed, marking failed: %s", exc, exc_info=True)
            return self._attach_env_reward(task, float("nan")), True

    @staticmethod
    def _attach_env_reward(sample: Sample, reward: Optional[float]) -> Sample:
        """Attach an env-sourced trajectory return to the LAST generated Part so the
        trainer (:class:`~unirl.trainer.agentic_env.AgenticEnvTrainer`) can read it
        directly — env tasks bypass ``RewardService``. No-op when the env supplied no
        reward, so tool-only envs (calculator/search) are byte-identical."""
        if reward is None:
            return sample
        gens = sample.gen_parts()
        if not gens:
            return sample
        last = gens[-1]
        rewarded = _part_with_field(
            last, "rewards", torch.full((int(last.batch_size),), float(reward), dtype=torch.float32)
        )
        return sample.with_parts([rewarded if p is last else p for p in sample.parts])

    def set_stopping(self, stopping: bool) -> None:
        """Ask in-flight trajectories to suspend at their next turn boundary."""
        self._stopping = bool(stopping)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        self._inner.shutdown()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        self._inner.sleep()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        self._inner.wake_up()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload_weights(self, *, track_prefix: str = "") -> None:
        self._inner.onload_weights(track_prefix=track_prefix)

    @property
    def is_offloaded(self) -> bool:
        return self._inner.is_offloaded

    def health_check(self) -> bool:
        return self._inner.health_check()

    def get_memory_info(self) -> Dict[str, float]:
        return self._inner.get_memory_info()

    def pause(self) -> None:
        self._inner.pause()

    def resume(self) -> None:
        self._inner.resume()

    def init_weights_update_group(self, **kwargs: Any) -> None:
        self._inner.init_weights_update_group(**kwargs)

    def update_weights_from_distributed(self, **kwargs: Any) -> None:
        self._inner.update_weights_from_distributed(**kwargs)

    def destroy_weights_update_group(self, **kwargs: Any) -> None:
        self._inner.destroy_weights_update_group(**kwargs)

    def update_weights_from_ipc(self, **kwargs: Any) -> None:
        self._inner.update_weights_from_ipc(**kwargs)

    def update_weights_from_tensor(self, **kwargs: Any) -> None:
        self._inner.update_weights_from_tensor(**kwargs)

    def set_lora_from_tensors(self, adapter_name: str, lora_tensors: Dict[str, Any], **kwargs: Any) -> None:
        self._inner.set_lora_from_tensors(adapter_name, lora_tensors, **kwargs)
