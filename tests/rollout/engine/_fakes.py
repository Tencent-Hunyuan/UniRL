"""CPU-only fakes for exercising the sync rollout contract under threads.

No GPU, no sglang/vllm. ``FakeBackend`` mimics the sync ``Backend`` shape (one
shared ``threading.Semaphore`` bound; ``generate`` records the wire and fans a
multi-payload batch out on a pool, like the HTTP impl; sync ``abort``/
``pause``/``resume`` flag-setters), and ``FakeEngine(BaseSingleTurnRolloutEngine)``
implements the sync path like the real ``SGLangRolloutEngine`` (build per-prompt
wire -> backend.generate -> flatten -> fill the frontier gen ``Part`` ->
``_stamp_weight_version``) — safe for CONCURRENT callers, the property the
agentic drain threads rely on. Observability for the threading tests: lock-
guarded ``inflight``/``peak`` counters, an ``entered`` event, and an optional
hold-gate (``block_until_released`` + ``release``) that parks every in-flight
request until a control (or the test) releases it.

To keep ``Sample``/``Part`` equality (dataclass ``__eq__``) usable in assertions,
the fake fills only the gen ``Part``'s ``primitive`` (a tensor-free ``Texts``)
and leaves the tensor-bearing ``segment`` ``None`` — a raw ``torch.Tensor`` field
would make ``==`` raise "ambiguous truth value".
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseSingleTurnRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

# --------------------------------------------------------------------------- #
# Deterministic "generation"
# --------------------------------------------------------------------------- #


def raw_text_for(prompt_text: str, k: int) -> str:
    """The decoded text for candidate ``k`` of ``prompt_text`` — a pure function,
    so a reference can be reconstructed independently of the engine run."""
    return f"{prompt_text}::cand{k}"


@dataclass
class FakeRaw:
    """Structural stand-in for the seam's ``RawResult`` (the wire fields the
    adapter consumes). Only ``text`` is read by the fake's response builder; the
    aligned ``token_ids``/``logprobs`` are carried to mirror the real shape."""

    text: str
    token_ids: List[int]
    logprobs: List[float]
    finish_reason: str = "stop"


def _raw_for(prompt_text: str, k: int) -> FakeRaw:
    return FakeRaw(
        text=raw_text_for(prompt_text, k),
        token_ids=[len(prompt_text), k],
        logprobs=[-0.1 * k, -0.2 * k],
    )


# --------------------------------------------------------------------------- #
# Fake backend — faithful to the sync Backend shape (semaphore + thread fan-out)
# --------------------------------------------------------------------------- #


class FakeBackend:
    """The sync ``Backend`` shape, in-memory: shared semaphore, per-request hold
    (``yields`` × 1ms — enough to overlap threads without slowing the suite),
    sync control flags, and lock-guarded concurrency counters."""

    def __init__(self, *, concurrency: int, yields: int = 4) -> None:
        # One bound across ALL callers of this backend (batch fan-out threads
        # AND concurrent engine.generate callers) — the load-bearing "shared".
        self._sem = threading.Semaphore(int(concurrency))
        self.concurrency = int(concurrency)
        self._hold_s = int(yields) * 0.001

        # Observability for the wire-order/overlap assertions. ``calls`` is the
        # wire the backend was handed, recorded at ``generate`` entry (so batch
        # order is deterministic — per-thread execution order is not).
        self._lock = threading.Lock()
        self.calls: List[Dict[str, Any]] = []
        self._inflight = 0
        self.peak = 0
        # Optional extra per-prompt hold (seconds), keyed by payload["text"], to
        # stagger completion order for the continuous-batching overlap test.
        self.delay_for: Dict[str, float] = {}

        # Control-plane flags the sync verbs set.
        self.aborted = False
        self.paused = False

        # In-flight gating for the control/threading tests: when set,
        # generate parks until abort() (or the test) sets ``release``.
        self.block_until_released = False
        self.entered = threading.Event()  # set once a request body is running
        self.release = threading.Event()

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    # ---- generation: thread-safe, semaphore-bounded ----
    def generate(self, requests: List[Dict[str, Any]]) -> List[FakeRaw]:
        with self._lock:
            self.calls.extend(requests)  # deterministic wire order
        if not requests:
            return []
        if len(requests) == 1:
            return self._generate_one(requests[0])
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(requests))) as pool:
            nested = list(pool.map(self._generate_one, requests))
        return [item for sublist in nested for item in sublist]

    def _generate_one(self, payload: Dict[str, Any]) -> List[FakeRaw]:
        n = int(payload["sampling_params"]["n"])
        with self._sem:
            with self._lock:
                self._inflight += 1
                self.peak = max(self.peak, self._inflight)
            self.entered.set()
            try:
                if self.block_until_released:
                    if not self.release.wait(timeout=5.0):
                        raise TimeoutError("FakeBackend hold-gate never released")
                else:
                    time.sleep(self._hold_s + self.delay_for.get(payload.get("text"), 0.0))
            finally:
                with self._lock:
                    self._inflight -= 1
        return [_raw_for(payload["text"], k) for k in range(n)]

    # ---- control plane: plain sync verbs (always reach the backend) ----
    def abort(self, *, abort_all: bool = True, rid: Optional[str] = None) -> None:
        del abort_all, rid
        self.aborted = True
        self.release.set()  # release any parked request (partials return)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def shutdown(self) -> None:
        self.release.set()  # never leave a parked request behind


# --------------------------------------------------------------------------- #
# Fake engine — the sync single-turn contract, safe for concurrent callers
# --------------------------------------------------------------------------- #


class FakeEngine(BaseSingleTurnRolloutEngine):
    """Minimal single-turn engine over the sync fake backend."""

    def __init__(self, *, concurrency: int = 8, yields: int = 4) -> None:
        self._backend = FakeBackend(concurrency=concurrency, yields=yields)
        self._weight_version = 0

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Sync whole-Sample path through ``backend.generate`` (thread-safe:
        prepare/finish are pure per-call, the backend bounds concurrency)."""
        wire, _ = self._build_inputs(sample)
        raw = self._backend.generate(wire)
        return self._stamp_weight_version(self._build_response(sample, raw))

    @staticmethod
    def _build_inputs(sample: Sample) -> Tuple[List[Dict[str, Any]], int]:
        """Adapter-like build_inputs: one ``/generate``-shaped payload per prompt,
        carrying ``n`` = the per-prompt fan-out (gen rows / #prompts)."""
        prompts = list(sample.parts[0].primitives["text"].texts)
        n = sample.parts[-1].batch_size // len(prompts)
        wire = [{"text": p, "sampling_params": {"n": n}} for p in prompts]
        return wire, n

    @staticmethod
    def _build_response(sample: Sample, raw: List[FakeRaw]) -> Sample:
        """Adapter-like build_response: row ``j`` of the gen Part <- ``raw[j]``
        (prompt-major / group-by-parent), filling only the tensor-free primitive."""
        gen_part = sample.parts[-1]
        filled = gen_part.fill(primitives={"text": Texts(texts=[r.text for r in raw])})
        return Sample(parts=[*sample.parts[:-1], filled])

    # ---- control plane: sync verbs forwarded to the backend ----
    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        del ids
        self._backend.abort(abort_all=True)
        return []

    def pause(self) -> None:
        self._backend.pause()

    def resume(self) -> None:
        self._backend.resume()

    def shutdown(self) -> None:
        self._backend.shutdown()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def build_request_batch(*, P: int, n: int) -> Sample:
    """A multi-prompt request forked ``n`` ways: parts = [input(P), gen-shell(P*n)].

    ``Part.input`` + ``Sample.fork`` lay the lineage down group-by-parent, the
    shape ``generate``'s split -> gather -> concat round-trips over.
    """
    ids = [f"p{i}" for i in range(P)]
    prompts = [f"prompt-{i}" for i in range(P)]
    head = Part.input(ids, primitives={"text": Texts(texts=prompts)})
    request = Sample.request(head)
    return request.fork(n, sampling_params=ARSamplingParams(samples_per_prompt=n))


__all__ = [
    "FakeBackend",
    "FakeEngine",
    "FakeRaw",
    "build_request_batch",
    "raw_text_for",
]
