"""One-trajectory agentic rollout engine."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.base import SyncRolloutEngine
from unirl.rollout.harness.protocol import HarnessContext, RolloutHarness
from unirl.rollout.harness.tool_agent import ToolAgentHarness
from unirl.types.sample import Sample, _part_with_field


class AgenticRolloutEngine(SyncRolloutEngine):
    """Run one environment-backed trajectory over a local inner engine."""

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
        self._env = config.env
        require(self._env is not None, "AgenticRolloutEngine requires an env (config.env)")
        self._maybe_inject_tool_schemas(config.inner, self._env)
        inner = config.inner.make_engine(strategy=strategy, **deps)
        if not isinstance(inner, SyncRolloutEngine):
            shutdown = getattr(inner, "shutdown", None)
            if callable(shutdown):
                shutdown()
            raise ValueError(
                f"AgenticRolloutEngine inner must implement the single-turn engine contract; got {type(inner).__name__}"
            )
        self._inner: SyncRolloutEngine = inner

        self._max_turns = int(config.max_turns)
        env_max_turns = getattr(self._env, "max_turns", None)
        require(
            env_max_turns is None or int(env_max_turns) == self._max_turns,
            f"env.max_turns ({env_max_turns}) must equal config.max_turns ({self._max_turns})",
        )
        self._stopping = False
        self._policy_version = 0
        self._harness: RolloutHarness = ToolAgentHarness(
            env=self._env,
            sampling=config.episode_sampling,
            max_turns=self._max_turns,
        )
        self._harness_ctx = HarnessContext(
            engines={"policy": self._inner.generate},
            suspend=lambda: self._stopping,
        )

    @staticmethod
    def _maybe_inject_tool_schemas(inner_cfg: Any, env: Any) -> None:
        get_schemas = getattr(env, "tool_schemas", None)
        if not callable(get_schemas) or not hasattr(inner_cfg, "chat_template_kwargs"):
            return
        chat_template_kwargs = dict(inner_cfg.chat_template_kwargs or {})
        chat_template_kwargs.setdefault("tools", get_schemas())
        inner_cfg.chat_template_kwargs = chat_template_kwargs

    def generate(self, sample: Sample) -> Sample:
        generated_before = len(sample.gen_parts())
        policy_version = self._policy_version
        outcome = self._harness.run(sample, self._harness_ctx)
        if outcome.status not in ("completed", "suspended", "failed"):
            raise ValueError(f"unknown harness outcome status: {outcome.status!r}")
        result = self._stamp_generated_versions(outcome.sample, generated_before, policy_version)
        return self._stamp_outcome(result, outcome.status)

    @staticmethod
    def _stamp_generated_versions(sample: Sample, generated_before: int, policy_version: int) -> Sample:
        generated_seen = 0
        parts = []
        for part in sample.parts:
            if part.is_gen:
                if generated_seen >= generated_before:
                    part = _part_with_field(part, "weight_version", policy_version)
                generated_seen += 1
            parts.append(part)
        return sample.with_parts(parts)

    @staticmethod
    def _stamp_outcome(sample: Sample, status: str) -> Sample:
        if not sample.parts:
            return sample
        last = _part_with_field(sample.parts[-1], "harness_status", status)
        return sample.with_parts([*sample.parts[:-1], last])

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_stopping(self, stopping: bool = True) -> None:
        self._stopping = bool(stopping)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_policy_version(self, policy_version: int) -> None:
        if policy_version < 0:
            raise ValueError(f"policy_version must be non-negative; got {policy_version}")
        self._policy_version = int(policy_version)

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

    @property
    def tensor_weight_sync_target(self) -> SyncRolloutEngine:
        """The concrete receiver whose serializer contract tensor sync follows."""
        return self._inner

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


__all__ = ["AgenticRolloutEngine"]
