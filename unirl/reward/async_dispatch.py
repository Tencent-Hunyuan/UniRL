"""Chain rollout completion to asynchronous reward scoring on the driver."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import fields as dc_fields
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Concurrent HTTP scores. Each worker uses its own requests.Session.
_MAX_INFLIGHT_SCORES = 8

RewardTiming = Tuple[float, float]


class RewardTimings:
    """Thread-safe generate/reward durations, consumed once per training step."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[RewardTiming] = []

    def append(self, item: RewardTiming) -> None:
        with self._lock:
            self._items.append(item)

    def consume_max(self) -> Optional[RewardTiming]:
        """Return the slowest generate and reward seconds since the last consume."""
        with self._lock:
            items = self._items
            self._items = []
        if not items:
            return None
        return (max(generate_s for generate_s, _reward_s in items), max(reward_s for _generate_s, reward_s in items))


class _DriverFutureCall:
    """Adapt a concurrent future to the async manager's call interface."""

    def __init__(self, future: "Future") -> None:
        self._future = future

    def ready(self) -> bool:
        return self._future.done()

    def result(self) -> Any:
        return self._future.result()


def _materialize_except_conditions(sample: Any) -> Any:
    """Fetch TensorRefs for scoring, leaving trajectory ``conditions`` as transport refs."""
    from unirl.distributed.tensor.ref import TensorRef, map_tree
    from unirl.types.sample import Part

    def leaf(value: Any) -> Any:
        if isinstance(value, Part):
            rebuilt = {
                field.name: (
                    getattr(value, field.name)
                    if field.name == "conditions"
                    else map_tree(getattr(value, field.name), leaf)
                )
                for field in dc_fields(value)
            }
            return type(value)(**rebuilt)
        if isinstance(value, TensorRef):
            return value.materialize(backend=None)
        return value

    return map_tree(sample, leaf)


class DriverRewardClient:
    """Run a remote HTTP reward client on the driver without a GPU worker."""

    # One scorer, so trainer DP-geometry sees dp_size=1.
    dp_size = 1

    def __init__(self, service: Any) -> None:
        from unirl.reward.remote import RemoteRewardBackend

        backend = service.backend
        if not isinstance(backend, RemoteRewardBackend):
            raise TypeError(
                "DriverRewardClient only hosts a remote HTTP reward backend; "
                f"got {type(backend).__name__}. A local scorer would load its weights on the driver."
            )
        self._service = service
        self._pool = ThreadPoolExecutor(max_workers=_MAX_INFLIGHT_SCORES, thread_name_prefix="driver-reward")

    def _score(self, sample: Any) -> Any:
        # Score a copy whose media is local, then attach the rewards to the
        # original so its trajectory TensorRefs stay on the rollout transport.
        from unirl.types.sample import _part_with_field

        scored = self._service.score_and_attach(_materialize_except_conditions(sample))
        frontier = _part_with_field(sample.parts[-1], "rewards", scored.parts[-1].rewards)
        frontier = _part_with_field(frontier, "component_rewards", scored.parts[-1].component_rewards)
        return sample.with_parts([*sample.parts[:-1], frontier])

    def launch_nowait(self, method_name: str, *args: Any, **kwargs: Any) -> _DriverFutureCall:
        if method_name != "score_and_attach":
            raise AttributeError(f"DriverRewardClient only serves score_and_attach, got {method_name!r}")
        return _DriverFutureCall(self._pool.submit(self._score, *args, **kwargs))

    def score_and_attach(self, sample: Any) -> Any:
        return self._score(sample)

    def offload(self) -> None:
        """No GPU weights to park; the residency planner still asks."""

    def onload(self) -> None:
        """No GPU weights to restore; the residency planner still asks."""

    def shutdown(self) -> None:
        """Finish submitted scores, then close the HTTP sessions."""
        self._pool.shutdown(wait=True, cancel_futures=False)
        self._service.dispose()


class ChainedRewardCall:
    """Release a rollout lane after generation while chained reward work continues."""

    def __init__(self, rollout_call: Any, reward: Any, timings: Optional[RewardTimings] = None) -> None:
        self._rollout_call = rollout_call
        self._reward = reward
        self._timings = timings
        self._reward_call: Optional[Any] = None
        self._lock = threading.Lock()
        self._started_at = time.perf_counter()
        self._released_at: Optional[float] = None
        self._recorded = False

    def _start_if_ready(self, *, block: bool) -> bool:
        with self._lock:
            if self._reward_call is not None:
                return True
            if not block and not self._rollout_call.ready():
                return False
            sample = self._rollout_call.result()
            self._released_at = time.perf_counter()
            self._reward_call = self._reward.launch_nowait("score_and_attach", sample)
            return True

    def _record(self) -> None:
        with self._lock:
            if self._recorded or self._timings is None or self._released_at is None:
                return
            self._recorded = True
            generate_s = self._released_at - self._started_at
            released_at = self._released_at
        self._timings.append((generate_s, time.perf_counter() - released_at))

    def is_capacity_released(self) -> bool:
        """Release rollout capacity and start reward once generation completes."""
        return self._start_if_ready(block=False)

    def ready(self) -> bool:
        """True only when the scored Sample can enter the completed queue."""
        if not self.is_capacity_released():
            return False
        if not self._reward_call.ready():
            return False
        # Record when scoring finishes, not when the trainer later pulls result().
        self._record()
        return True

    def result(self) -> Any:
        self._start_if_ready(block=True)
        value = self._reward_call.result()
        self._record()
        return value

    def discard_on_completion(self) -> None:
        """Drain the chain before its reward client is shut down."""
        try:
            self.result()
        except Exception:
            logger.debug("discarded chained reward call failed during shutdown", exc_info=True)


__all__ = ["ChainedRewardCall", "DriverRewardClient"]
