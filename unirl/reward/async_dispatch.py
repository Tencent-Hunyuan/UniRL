"""Chain rollout completion to asynchronous reward scoring on the driver."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

logger = logging.getLogger(__name__)


class _DriverCall:
    """Future-like call matching the ``ready()``/``result()`` surface the
    async manager expects from a ``Handle`` ``launch_nowait`` result."""

    def __init__(self, future: "Future") -> None:
        self._future = future

    def ready(self) -> bool:
        return self._future.done()

    def result(self) -> Any:
        return self._future.result()


class DriverLocalReward:
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

    # Handle-compatible layout: a driver-local reward is a single scorer, so the
    # trainer's dp-geometry checks see dp_size=1 (batch % 1 == 0 always).
    dp_size = 1
    world_size = 1

    def __init__(self, service: Any, *, max_workers: int = 8) -> None:
        self._service = service
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="driver-reward")

    def _score(self, sample: Any) -> Any:
        # Materialize every TensorRef leaf on the driver (module-level hydrate()
        # only resolves a bare TensorRef, not one nested in the Sample tree).
        # Each span round-trips through its bound worker / plasma object_ref
        # (the lane CPU-map keeps generated media globally readable).
        from unirl.distributed.tensor.ref import TensorRef, map_tree

        resolved = map_tree(sample, lambda o: o.materialize(backend=None) if isinstance(o, TensorRef) else o)
        return self._service.score_and_attach(resolved)

    def launch_nowait(self, method_name: str, *args: Any, **kwargs: Any) -> _DriverCall:
        if method_name != "score_and_attach":
            raise AttributeError(f"DriverLocalReward only serves score_and_attach, got {method_name!r}")
        return _DriverCall(self._pool.submit(self._score, *args, **kwargs))

    def score_and_attach(self, sample: Any) -> Any:
        return self._score(sample)

    def set_request_batch_size(self, n: int) -> None:
        """Chunk each HTTP /score call to ``n`` sample rows (backend-side)."""
        backend = getattr(self._service, "backend", None)
        if backend is not None and hasattr(backend, "request_batch_size"):
            backend.request_batch_size = int(n)

    def is_available(self) -> bool:
        return self._service.is_available()

    def offload(self) -> None:
        pass

    def onload(self) -> None:
        pass

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
        dispose = getattr(self._service, "dispose", None)
        if callable(dispose):
            dispose()


class AsyncRewardCall:
    """Future-like rollout call that releases its lane before reward completes.

    ``capacity_ready`` becomes true as soon as generation finishes. At that
    transition the generated prompt group is immediately submitted to
    ``RewardService`` and the rollout lane can be refilled. ``ready`` becomes true
    only when reward finishes, so completed groups enter the manager queue in
    reward-completion order.
    """

    def __init__(self, rollout_call: Any, reward: Any, *, max_attempts: int = 2) -> None:
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        self._rollout_call = rollout_call
        self._reward = reward
        self._max_attempts = int(max_attempts)
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

    def capacity_ready(self) -> bool:
        """Release rollout capacity and start reward once generation completes."""
        return self._start_if_ready(block=False)

    def ready(self) -> bool:
        """True only when the scored Sample can enter the completed queue."""
        if not self.capacity_ready():
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

        for attempt in range(self._max_attempts):
            try:
                return self._reward_call.result()
            except Exception:
                if attempt + 1 >= self._max_attempts:
                    raise
                logger.warning("per-prompt reward scoring failed; retrying the same generated group")
                if self._sample is None:
                    raise RuntimeError("cannot retry async reward without the generated Sample")
                self._reward_call = self._reward.launch_nowait("score_and_attach", self._sample)
        raise AssertionError("unreachable")

    def discard_on_completion(self) -> None:
        """Drain the chained calls in the background and release their outputs."""

        def discard() -> None:
            try:
                self.result()
            except Exception:
                logger.debug("discarded async reward chain failed during shutdown", exc_info=True)

        threading.Thread(target=discard, name="async-reward-discard", daemon=True).start()


def chain_reward(rollout_call: Any, reward: Any, *, max_attempts: int = 2) -> AsyncRewardCall:
    """Return a future that starts reward as soon as ``rollout_call`` completes."""
    return AsyncRewardCall(rollout_call, reward, max_attempts=max_attempts)


__all__ = ["AsyncRewardCall", "DriverLocalReward", "chain_reward"]
