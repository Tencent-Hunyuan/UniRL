"""AgenticRolloutEngine — multi-turn (agentic) rollout over a rank-0 coordinator (LIN-522, LIN-531).

The engine is a ``BaseRolloutEngine`` on a DP-replicated slab. Each worker builds its **own local**
inner single-turn engine + environment (the ``ComposedRolloutEngine`` build-inner pattern) and runs
a **pool of drain threads** (``per_worker_concurrency`` of them) that pull single-trajectory tasks
from a central queue and run each through the configured harness (the multi-turn agent loop,
:class:`~unirl.rollout.harness.tool_agent.ToolAgentHarness`) on its own thread:
per-turn ``inner.generate`` (the inner engine is safe for concurrent callers — its backend keeps
the in-flight requests batching together) and ``env.step`` (the env is re-entrant across threads).

**Partial rollout interface (LIN-531).** The engine is the *mechanism*; the trainer owns the *policy*
(how many to over-sample, staleness, when to sync). The coordinator exposes a
**submit / poll / finalize / abort** interface over a **background** drain the trainer reaps and
interrupts — consumed driver-side through
:class:`~unirl.rollout.engine.asynchronous.AsyncAgenticRolloutEngine` (rank-0 unwrap + group assembly +
versioned buffering):

- ``submit(request)`` — enqueue a pool (fresh prompts and/or carried partials) and fire the drain
  **non-blocking**; the trainer reaps completions with ``poll`` and cuts the tail with ``abort``.
- ``poll()`` — the trajectories **completed since the last poll** (aggregated across workers).
- ``finalize_if_drained()`` — when a drive has naturally drained, join its worker refs and reap the
  final completions in one coordinator call before the next ``submit`` resets worker buffers.
- ``abort()`` — a **turn-boundary** stop: workers stop pulling and checkpoint their in-flight
  trajectories at the next turn boundary; returns the carried set (in-flight partials + not-started
  tasks). After it returns the inner engine is **decode-idle** → safe for the trainer to weight-sync.
- ``generate(request)`` — a thin *barrier* convenience (submit + drain-to-completion) kept for the
  colocate/``partial_rollout=false`` path, byte-identical to the pre-partial-rollout behaviour.

Two roles, one class:

- **Coordinator = rank 0** (``set_workers`` + ``submit``/``poll``/``abort``/``generate``,
  ``BROADCAST + RANK_ZERO``): rank 0 fans the batch into ``n × P`` single-trajectory tasks, fires one
  ``run_drain`` per worker (raw ``Worker.call``), serves ``next_task`` pulls, and aggregates the
  per-worker completed/carried buffers.
- **Worker = every instance** (``run_drain`` / ``_drain_worker`` / ``_run_one`` / ``_pull`` + the
  un-decorated control methods ``drain_completed``/``collect_carried``/``set_stopping``/
  ``reset_round``): ``run_drain`` blocks in its Worker-actor slot for the whole drive while the
  drain threads work; the control methods ride the threaded Worker
  (``worker_max_concurrency>1``) alongside it.

Weight sync: the trainer quiesces the rollout (``abort`` — turn-boundary checkpoint, or the
``generate`` barrier) and syncs *between* drives. The quiesce is structural: ``abort``/``generate``
join the drain threads (``ray.get(drain_refs)``), and every generate call runs on one of those
threads — after the join, nothing is in flight and the inner engine is decode-idle. Delegated
verbs below forward to the inner engine.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Deque, Dict, List, Optional, Tuple

import ray
import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.synchronous import BaseRolloutEngine, SyncRolloutEngine
from unirl.rollout.harness.protocol import HarnessContext, RolloutHarness
from unirl.rollout.harness.tool_agent import ToolAgentHarness
from unirl.types.sample import Sample, _part_with_field
from unirl.types.sampling import total_samples_per_prompt

logger = logging.getLogger(__name__)


class AgenticRolloutEngine(BaseRolloutEngine):
    """Multi-turn rollout engine: rank-0 coordinator over per-worker pull-drain loops."""

    _component_name = "agentic"

    # ------------------------------------------------------------------
    # Construction (mirrors ComposedRolloutEngine: build the inner engine + env)
    # ------------------------------------------------------------------

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
        # Each worker builds its OWN local inner engine + environment.
        self._env = config.env  # an Environment (built per worker via its _target_); must be re-entrant
        require(self._env is not None, "AgenticRolloutEngine requires an env (config.env)")
        # Single source of truth for tool schemas (LIN-519): advertise the env's
        # tools to the model through the inner engine's chat template, so a recipe
        # never restates the schema JSON (which could silently drift from the env's
        # actual tools). Mutates the inner CONFIG before it is built; an explicit
        # ``inner.chat_template_kwargs.tools`` in the recipe still wins.
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
        self._n = total_samples_per_prompt(self._sp)  # GRPO group size
        self._max_turns = int(config.max_turns)
        self._partial_rollout = bool(getattr(config, "partial_rollout", False))
        # Guard: the env's own turn bound (set independently in the recipe under
        # ``env.max_turns``) must agree with the engine's ``config.max_turns``, else
        # the drain loop and the env disagree on when a trajectory is done.
        _env_mt = getattr(self._env, "max_turns", None)
        require(
            _env_mt is None or int(_env_mt) == self._max_turns,
            f"env.max_turns ({_env_mt}) must equal config.max_turns ({self._max_turns}); "
            "they are set independently in the recipe and must agree.",
        )
        # Per-worker trajectory cap = the drain pool size (distinct from the inner
        # backend's per-request semaphore — two independent bounds). A trajectory
        # holds its thread across its whole life, incl. tool-wait between turns —
        # so siblings keep the GPU busy while one waits on a slow tool.
        self._concurrency = int(config.per_worker_concurrency)

        # Task-internal control flow is the harness's (worker-side plugin);
        # this runtime keeps queueing/concurrency/buffers/abort. Step 1 of the
        # harness migration constructs the one existing harness here from the
        # same config fields; config-selected harnesses come later.
        self._harness: RolloutHarness = ToolAgentHarness(env=self._env, sampling=self._sp, max_turns=self._max_turns)
        self._harness_ctx = HarnessContext(
            engines={"policy": self._inner.generate},
            suspend=lambda: self._stopping,
        )

        # Coordinator state (populated on rank 0 only, by set_workers).
        self._workers: List[Any] = []
        self._role: str = ""
        self._queue: Deque[Sample] = deque()
        self._qlock = threading.Lock()
        self._drain_refs: List[Any] = []  # rank-0: outstanding run_drain ObjectRefs (joined by abort/generate)

        # Worker-side partial-rollout buffers (LIN-531). The drain's done-callback
        # routes each finished trajectory into ``_completed`` (terminal) or
        # ``_checkpointed`` (turn-boundary-interrupted); ``_buf_lock`` guards them
        # because the loop thread appends while a control-RPC thread reads.
        self._completed: List[Sample] = []
        self._checkpointed: List[Sample] = []
        self._buf_lock = threading.Lock()
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

    # ------------------------------------------------------------------
    # Coordinator (rank 0) — the NCCLWeightSync pattern
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def set_workers(self, actor_handles: List[Any], role_name: str) -> None:
        """Rank 0 caches the slab's Worker actor handles (incl. its own) + role name.

        Wired once by the builder: ``handle.set_workers(handle.workers, handle.role_name)``
        (the ``NCCLWeightSync.set_rollout_targets`` shape). The handles are plain
        picklable Ray actor handles, so they survive the ``Worker.call`` arg path.
        """
        self._workers = list(actor_handles)
        self._role = str(role_name)

    def _split_request(self, request: Sample) -> List[Sample]:
        """Fan a batched request into ``n × P`` single-trajectory tasks (the ``n``
        siblings of a prompt share its slash-free root id — design §3.1)."""
        return [prompt for prompt in request.split() for _ in range(self._n)]

    def _enqueue_tasks(self, tasks: List[Sample]) -> None:
        """Install a ready task list (fresh single-trajectory tasks and/or carried
        partials) as the queue. Each element is one trajectory the drain pulls."""
        with self._qlock:
            self._queue = deque(tasks)

    def _fan(self, method: str) -> List[Sample]:
        """Fan an un-decorated worker control method via raw ``Worker.call`` and
        flatten the per-worker ``List[Sample]`` returns (FIFO worker order)."""
        refs = [w.call.remote(self._role, method, (), {}) for w in self._workers]
        return [s for shard in ray.get(refs) for s in shard]

    def _fire_drain(self) -> None:
        """Fire one ``run_drain`` per worker **non-blocking**; keep the refs so
        ``abort``/``generate`` can join them once the drives quiesce/finish."""
        coordinator = self._workers[0]  # rank 0's own Worker actor handle (workers pull from it)
        self._drain_refs = [
            w.call.remote(self._role, "run_drain", (coordinator, self._role), {}) for w in self._workers
        ]

    def _reset_all(self) -> None:
        """Clear every worker's per-drive buffers + stop flag before a new drive."""
        ray.get([w.call.remote(self._role, "reset_round", (), {}) for w in self._workers])

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def submit(self, tasks: List[Sample]) -> None:
        """Rank-0: enqueue a task pool and fire the **background** drain (non-blocking).

        ``tasks`` is a **flat list of single-trajectory tasks** — fresh prompt
        siblings (the trainer pre-splits its request into ``n × P`` via
        ``Sample.split()`` × ``n``) and/or carried partial trajectories (each resumed
        by ``_run_one`` from ``len(gen_parts())``). A list (not one batched ``Sample``)
        because carried partials have heterogeneous part-depths and cannot share one
        ``Sample``. The trainer reaps completions with :meth:`poll` and cuts the tail
        with :meth:`abort`. Safe to call only when the previous drive was atomically
        :meth:`finalize_if_drained` or :meth:`abort`ed (else two drains would double-pull).
        """
        require(bool(self._workers), "AgenticRolloutEngine.submit: call set_workers() first (rank 0)")
        self._reset_all()
        self._enqueue_tasks(list(tasks))
        self._fire_drain()

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def poll(self) -> List[Sample]:
        """Rank-0: the trajectories **completed since the last poll**, across workers.

        Non-blocking (reads each worker's completed buffer). The trainer loops on it,
        groups by root id, and stops once it has enough complete GRPO groups.
        """
        if not self._workers:
            return []
        return self._fan("drain_completed")

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def finalize_if_drained(self) -> Optional[List[Sample]]:
        """Finalize a naturally drained drive and return its last completions.

        ``None`` means at least one ``run_drain`` ref is still active. Once every ref
        is ready, ``ray.get`` joins them (and surfaces worker failures), the refs are
        cleared, and the now-stable completed buffers are drained exactly once. An
        empty ref set is already finalized and returns ``[]`` without polling workers.

        The readiness check and final reap must remain ONE coordinator operation
        when the next action is ``submit()``: ``submit`` resets per-drive worker
        buffers, so a separate check-then-poll would race it.
        """
        if not self._drain_refs:
            return []
        ready, _ = ray.wait(self._drain_refs, num_returns=len(self._drain_refs), timeout=0)
        if len(ready) != len(self._drain_refs):
            return None
        refs = self._drain_refs
        try:
            ray.get(refs)  # join and surface a failed worker drain before polling its buffers
        finally:
            # All refs are ready, so this drive is terminal even when one worker
            # reports a failure. Do not make later callers re-observe stale refs.
            self._drain_refs = []
        return self._fan("drain_completed")

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Rank-0: **turn-boundary** stop; return the carried (unfinished) set.

        Signals every worker to stop pulling and checkpoint its in-flight
        trajectories at the next turn boundary (``set_stopping``), joins the drives
        (``ray.get(drain_refs)`` — the quiesce), then collects the checkpointed
        partials + the not-yet-started tasks left in the queue. After it returns the
        inner engine is **decode-idle**, so the trainer can weight-sync safely; the
        carried set is re-submitted next drive to resume under the new weights.

        Contract: trajectories that *complete during the quiesce* (their in-flight
        turn was terminal) land in the **completed** buffer, not the carried set — the
        caller should ``poll()`` once more after ``abort()`` to collect them before the
        next ``submit`` (which resets the per-drive buffers).
        """
        del ids  # Agentic task ids do not map to backend request ids; abort the round.
        if not self._workers:
            return []
        ray.get([w.call.remote(self._role, "set_stopping", (), {}) for w in self._workers])
        if self._drain_refs:
            ray.get(self._drain_refs)  # wait for every drive to quiesce (in-flight checkpointed)
            self._drain_refs = []
        carried = self._fan("collect_carried")
        with self._qlock:
            carried.extend(self._queue)  # not-yet-started tasks carry too
            self._queue = deque()
        return carried

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def generate(self, sample: Sample) -> List[Sample]:
        """Thin **barrier** convenience: run the whole batch to completion.

        ``submit`` + drain-to-completion + collect — byte-identical trajectory set
        to the pre-partial-rollout behaviour (the ``partial_rollout=false`` /
        colocate path). Reached via ``handle.generate(batch)`` → ``[List[Sample]]``
        (BROADCAST passthrough collect, rank-0 only); the caller unwraps ``[0]``.
        """
        require(bool(self._workers), "AgenticRolloutEngine.generate: call set_workers() first (rank 0)")
        self._reset_all()
        self._enqueue_tasks(self._split_request(sample))
        self._fire_drain()
        ray.get(self._drain_refs)  # drives finish naturally (queue drained, no abort)
        self._drain_refs = []
        return self._fan("drain_completed")

    def next_task(self, worker_rank: int) -> Optional[Sample]:
        """Hand out the next trajectory task, or ``None`` when the queue is drained.

        Reached by raw ``Worker.call`` from pulling workers (un-decorated). FIFO for
        v1; this is the seam where sticky-prefix / staleness-aware routing later
        lives (design §11). ``worker_rank`` is an (unused) hint for that future.
        """
        del worker_rank
        with self._qlock:
            return self._queue.popleft() if self._queue else None

    # ------------------------------------------------------------------
    # Per-worker execution — the drain thread pool
    # ------------------------------------------------------------------

    def run_drain(self, coordinator: Any, role_name: str) -> None:
        """Drive this worker's drain for the whole drive (background).

        Reached by raw ``Worker.call`` (rank 0 fires one per worker, non-blocking).
        Runs ``per_worker_concurrency`` drain threads — one whole trajectory per
        thread at a time, so the pool size IS the trajectory cap. Results are
        written to the worker's ``_completed`` / ``_checkpointed`` buffers (read
        by ``drain_completed`` / ``collect_carried``), so this returns nothing.

        Joins every thread and re-raises the first failure — ``shutdown(wait=True)``
        alone would swallow a dead thread's error and silently strand queued tasks
        where today's contract is a loud drain failure.
        """
        with ThreadPoolExecutor(max_workers=self._concurrency, thread_name_prefix="agentic-drain") as pool:
            futures = [pool.submit(self._drain_worker, coordinator, role_name) for _ in range(self._concurrency)]
        for future in futures:
            future.result()

    def _drain_worker(self, coordinator: Any, role_name: str) -> None:
        """One drain thread: pull → run a whole trajectory → buffer, until the
        queue drains or ``set_stopping``. An unexpected failure (the pull RPC —
        ``_run_one`` itself never raises) sets ``_stopping`` so sibling threads
        checkpoint at their next turn boundary instead of grinding through the
        queue behind a doomed drive, then re-raises into ``run_drain``'s join."""
        try:
            while not self._stopping:
                task = self._pull(coordinator, role_name)
                if task is None:  # sentinel: queue drained
                    break
                sample, done = self._run_one(task)  # never raises (failure-isolated)
                with self._buf_lock:
                    (self._completed if done else self._checkpointed).append(sample)
        except BaseException:
            self._stopping = True
            raise

    def _run_one(self, task: Sample) -> Tuple[Sample, bool]:
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
            # Exhaustive over the outcome contract: a plugin returning an unknown
            # status (or a malformed outcome object) must not slip into training
            # as "completed" — it falls through to the except and is NaN-marked.
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

    def _pull(self, coordinator: Any, role_name: str) -> Optional[Sample]:
        """Pull the next task from the coordinator — a blocking Ray RPC on this
        drain thread. Kept as a method so no-Ray CPU tests can monkeypatch the
        seam with an in-process queue."""
        ref = coordinator.call.remote(role_name, "next_task", (self.rank or 0,), {})
        return ray.get(ref)

    # ------------------------------------------------------------------
    # Per-worker partial-rollout control methods (un-decorated, raw Worker.call).
    # They ride the threaded Worker (worker_max_concurrency>1) alongside the
    # running drain: the drain threads append to the buffers; these read/clear
    # them on a control thread — hence _buf_lock.
    # ------------------------------------------------------------------

    def drain_completed(self) -> List[Sample]:
        """Return + clear the trajectories completed since the last call (for ``poll``)."""
        with self._buf_lock:
            out, self._completed = self._completed, []
        return out

    def collect_carried(self) -> List[Sample]:
        """Return + clear the checkpointed (turn-boundary-interrupted) trajectories (for ``abort``)."""
        with self._buf_lock:
            out, self._checkpointed = self._checkpointed, []
        return out

    def set_stopping(self) -> None:
        """Signal the drain + in-flight ``_run_one``s to checkpoint at the next turn boundary."""
        self._stopping = True

    def reset_round(self) -> None:
        """Clear the per-drive buffers + stop flag before a new ``submit``/``generate``."""
        self._stopping = False
        with self._buf_lock:
            self._completed = []
            self._checkpointed = []

    # ------------------------------------------------------------------
    # Lifecycle — delegated to the inner engine
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Weight sync — delegated to the inner engine (single inner, no prefix routing).
    # Reached via raw Worker.call by the weight-sync driver (e.g. NCCLWeightSync).
    # ------------------------------------------------------------------

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
