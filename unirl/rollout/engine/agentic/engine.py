"""One-trajectory agentic rollout engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.harness.protocol import HarnessContext, RolloutHarness
from unirl.rollout.harness.tool_agent import ToolAgentHarness
from unirl.types.sample import Sample, _part_with_field

logger = logging.getLogger(__name__)


class AgenticRolloutEngine(BaseRolloutEngine):
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

        deps = dict(device=device, rank=rank, model_config=model_config)
        env = config.env
        require(env is not None, "AgenticRolloutEngine requires an env (config.env)")
        raw_kwargs = config.inner.chat_template_kwargs
        chat_template_kwargs = {} if raw_kwargs is None else dict(raw_kwargs)
        chat_template_kwargs.setdefault("tools", env.tool_schemas())
        config.inner.chat_template_kwargs = chat_template_kwargs
        self.inner: BaseRolloutEngine = config.inner.make_engine(strategy=strategy, **deps)

        require(
            env.max_turns == config.max_turns,
            f"env.max_turns ({env.max_turns}) must equal config.max_turns ({config.max_turns})",
        )
        self.stopping = False
        self.harness: RolloutHarness = ToolAgentHarness(
            env=env,
            sampling=config.episode_sampling,
            max_turns=config.max_turns,
        )
        self.harness_ctx = HarnessContext(
            engines={"policy": self.inner.generate},
            suspend=lambda: self.stopping,
        )

    def generate(self, sample: Sample) -> Sample:
        try:
            outcome = self.harness.run(sample, self.harness_ctx)
            if outcome.status not in ("completed", "suspended", "failed", "overflow"):
                raise ValueError(f"unknown harness outcome status: {outcome.status!r}")
            target, status = outcome.sample, outcome.status
        except Exception:  # noqa: BLE001 — last-resort net: a harness bug fails one trajectory, not the run
            logger.warning(
                "AgenticRolloutEngine: harness escaped its own net; marking trajectory failed", exc_info=True
            )
            target, status = sample, "failed"
        if not target.parts:
            return target
        last = _part_with_field(target.parts[-1], "harness_status", status)
        return target.with_parts([*target.parts[:-1], last])

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_stopping(self, stopping: bool = True) -> None:
        self.stopping = stopping

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        self.inner.shutdown()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        self.inner.sleep()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        self.inner.wake_up()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def onload_weights(self, *, track_prefix: str = "") -> None:
        self.inner.onload_weights(track_prefix=track_prefix)

    @property
    def is_offloaded(self) -> bool:
        return self.inner.is_offloaded

    @property
    def tensor_weight_sync_target(self) -> BaseRolloutEngine:
        """The concrete receiver whose serializer contract tensor sync follows."""
        return self.inner

    def health_check(self) -> bool:
        return self.inner.health_check()

    def get_memory_info(self) -> Dict[str, float]:
        return self.inner.get_memory_info()

    def pause(self) -> None:
        self.inner.pause()

    def resume(self) -> None:
        self.inner.resume()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_version(self, train_version: int) -> None:
        self.inner.set_version(train_version)

    def init_weights_update_group(self, **kwargs: Any) -> None:
        self.inner.init_weights_update_group(**kwargs)

    def update_weights_from_distributed(self, **kwargs: Any) -> None:
        self.inner.update_weights_from_distributed(**kwargs)

    def destroy_weights_update_group(self, **kwargs: Any) -> None:
        self.inner.destroy_weights_update_group(**kwargs)

    def update_weights_from_ipc(self, **kwargs: Any) -> None:
        self.inner.update_weights_from_ipc(**kwargs)

    def update_weights_from_tensor(self, **kwargs: Any) -> None:
        self.inner.update_weights_from_tensor(**kwargs)

    def set_lora_from_tensors(self, adapter_name: str, lora_tensors: Dict[str, Any], **kwargs: Any) -> None:
        self.inner.set_lora_from_tensors(adapter_name, lora_tensors, **kwargs)


__all__ = ["AgenticRolloutEngine"]
