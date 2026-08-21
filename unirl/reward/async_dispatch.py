"""Chain rollout completion to asynchronous reward scoring on the driver."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _DriverFutureCall:
    """Future-like call matching the ``ready()``/``result()`` surface the
    async manager expects from a ``Handle`` ``launch_nowait`` result."""

    def __init__(self, future: "Future") -> None:
        self._future = future

    def ready(self) -> bool:
        return self._future.done()

    def result(self) -> Any:
        return self._future.result()


class DriverRewardClient:
    """Run a remote-HTTP reward client on the driver — no GPU worker, no slab.

    UniRL places every ``Remote`` role on a GPU worker, but a remote reward does
    no GPU work: it just POSTs the generated media to a scorer on another node.
    Hosting the client on the driver keeps the training node's GPUs entirely
    train+rollout. Exposes the ``launch_nowait`` / ``.ready()`` / ``.result()``
    surface the async per-lane manager uses, backed by a small thread pool so
    scoring overlaps rollout and training. ``score_and_attach`` runs the
    undecorated ``RewardService`` body (``@distributed`` only dispatches through
    a ``Handle``); the sample is hydrated first so CPU plasma tensors resolve on
    the driver.
    """

    # Handle-compatible layout: the driver reward client represents one scorer,
    # so trainer DP-geometry checks see dp_size=1 (batch % 1 == 0 always).
    dp_size = 1
    world_size = 1

    def __init__(self, service: Any, *, max_workers: int = 8) -> None:
        self._service = service
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="driver-reward")

    def _score(self, sample: Any) -> Any:
        # Materialize every TensorRef leaf on the driver (module-level hydrate()
        # only resolves a bare TensorRef, not one nested in the Sample tree).
        # Each span round-trips through its bound worker / plasma object_ref
        # (the launcher's explicit CPU export keeps generated media readable).
        from unirl.distributed.tensor.ref import TensorRef, map_tree

        resolved = map_tree(sample, lambda o: o.materialize(backend=None) if isinstance(o, TensorRef) else o)
        return self._service.score_and_attach(resolved)

    def launch_nowait(self, method_name: str, *args: Any, **kwargs: Any) -> _DriverFutureCall:
        if method_name != "score_and_attach":
            raise AttributeError(f"DriverRewardClient only serves score_and_attach, got {method_name!r}")
        return _DriverFutureCall(self._pool.submit(self._score, *args, **kwargs))

    def score_and_attach(self, sample: Any) -> Any:
        return self._score(sample)

    def is_available(self) -> bool:
        return self._service.is_available()

    def offload(self) -> None:
        pass

    def onload(self) -> None:
        pass

    def shutdown(self) -> None:
        # Every submitted score must finish before its backend/session is disposed.
        self._pool.shutdown(wait=True, cancel_futures=False)
        dispose = getattr(self._service, "dispose", None)
        if callable(dispose):
            dispose()


class ChainedRewardCall:
    """Future-like rollout call that releases its lane before reward completes.

    ``is_capacity_released`` becomes true as soon as generation finishes. At
    that transition the generated prompt group is submitted to ``RewardService``
    and the rollout lane can be refilled. ``ready`` becomes true only when reward
    finishes, so completed groups enter the manager queue in reward-completion
    order.
    """

    def __init__(self, rollout_call: Any, reward: Any) -> None:
        self._rollout_call = rollout_call
        self._reward = reward
        self._sample: Optional[Any] = None
        self._reward_call: Optional[Any] = None
        self._lock = threading.Lock()

    def _start_if_ready(self, *, block: bool) -> bool:
        with self._lock:
            if self._reward_call is not None:
                return True
            if not block and not self._rollout_call.ready():
                return False
            sample = self._rollout_call.result()
            parts = getattr(sample, "parts", None)
            status = parts[-1].harness_status if parts else None
            if status == "suspended":
                # Agentic quiesce owns partial-trajectory carry. Async batch
                # trainers do not normally produce this state, but preserving it
                # keeps the wrapper compatible with RolloutManager's contract.
                self._sample = sample
                return True
            self._sample = sample
            self._reward_call = self._reward.launch_nowait("score_and_attach", sample)
            return True

    def is_capacity_released(self) -> bool:
        """Release rollout capacity and start reward once generation completes."""
        return self._start_if_ready(block=False)

    def ready(self) -> bool:
        """True only when the scored Sample can enter the completed queue."""
        if not self.is_capacity_released():
            return False
        if self._reward_call is None:
            return True
        return self._reward_call.ready()

    def result(self) -> Any:
        self._start_if_ready(block=True)
        if self._reward_call is None:
            if self._sample is None:
                raise RuntimeError("async reward call has neither a reward future nor a passthrough Sample")
            return self._sample

        return self._reward_call.result()

    def discard_on_completion(self) -> None:
        """Drain the chain before its reward client is shut down."""
        try:
            self.result()
        except Exception:
            logger.debug("discarded chained reward call failed during shutdown", exc_info=True)


def chain_reward(rollout_call: Any, reward: Any) -> ChainedRewardCall:
    """Return a future that starts reward as soon as ``rollout_call`` completes."""
    return ChainedRewardCall(rollout_call, reward)


__all__ = ["ChainedRewardCall", "DriverRewardClient", "chain_reward"]
