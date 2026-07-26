"""Async diffusion RL trainer — disaggregated train/rollout slabs for DiT.

Diffusion sibling of :class:`~unirl.trainer.async_ar.AsyncARTrainer`. It subclasses
:class:`~unirl.trainer.diffusion.DiffusionTrainer` with ``layout="separate"`` to
REUSE its two-slab build (train slab + dedicated rollout engine slab), the
cross-slab weight-sync wiring (``RemoteLoraWeightSync`` for the BAGEL recipe;
``NCCLWeightSync`` is also supported by ``_connect_separate``), and the diffusion
plumbing (``_build_req`` / ``_drop_decoded`` / ``evaluate`` / checkpoint /
FlowGRPO ``stack.train_track``). On top of that it overlays the SAME
single-threaded async rollout buffer loop as ``AsyncARTrainer``:

* Generation is launched as **non-blocking Ray futures** on the rollout slab
  (``_generate_async``) and reaped on the driver thread (``_reap_ready``); no
  producer thread, no locks.
* Reward is scored synchronously at reap time (``_score_into_buffer``) before
  groups enter the buffer. Generation overlaps training; reward scoring itself
  does not.
* Training consumes the freshest ``batch_size`` groups per step
  (``_advantage_and_train``: advantage + FlowGRPO optimizer step); it never calls
  the reward.

Two numeric knobs (identical semantics to AsyncARTrainer):
  * ``max_inflight`` — must be ``1`` so reap-time transfer never competes with
    a queued generation on the rollout workers.
  * ``buffer_max_staleness`` — regular rollout-weight syncs a buffered group
    may cross. ``0`` (default) never crosses a sync; ``>0`` enables a bounded
    policy-lag buffer.

Draining all in-flight generations before each weight sync is MANDATORY (a
weight + KV update corrupts an in-flight generation); that is the single-threaded
``_drain_all`` quiesce.

NOTE: the async buffer/generate-seam machinery below is intentionally a faithful
copy of ``AsyncARTrainer`` (it is engine- and modality-agnostic); a future refactor
could lift it into a shared mixin. Kept self-contained here to leave the validated
AR path untouched.
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import ray
import torch

from unirl.distributed.group.dispatch import DISPATCH_MODE_REGISTRY, Dispatch
from unirl.distributed.tensor import WorkerLocalTransport, hydrate
from unirl.distributed.tensor.pytree import infer_batch_size
from unirl.train.stack import TrainStepResult
from unirl.trainer.diffusion import DiffusionTrainer
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

logger = logging.getLogger(__name__)


class _RolloutBuffer:
    """Group-keyed rollout buffer (single-threaded; no lock needed).

    Each entry is one prompt's GRPO group — a ``RolloutTrack`` of
    ``samples_per_prompt`` already-scored samples — stamped with the
    ``weight_version`` it was generated under and a monotonic ``gen_id`` for
    freshness ordering. Groups are always complete (the whole ``generate``
    finished before they are ``put``), so there is no partial-group bookkeeping.
    """

    def __init__(self) -> None:
        self._items: List[Tuple[RolloutTrack, int, int]] = []  # (group, weight_version, gen_id)

    def put(self, track: RolloutTrack, *, weight_version: int, gen_id: int) -> None:
        self._items.append((track, int(weight_version), int(gen_id)))

    def size(self) -> int:
        return len(self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
    ) -> Optional[List[Tuple[RolloutTrack, int, int]]]:
        """Pop the ``n`` freshest complete groups, carrying leftovers forward.

        Returns ``None`` if fewer than ``n`` groups remain after eviction. When
        ``max_staleness`` is set, groups older than ``current_version -
        max_staleness`` weight versions are evicted first (bounded off-policy).
        """
        if max_staleness is not None and current_version is not None:
            self._items = [it for it in self._items if current_version - it[1] <= max_staleness]
        if len(self._items) < n:
            return None
        self._items.sort(key=lambda it: it[2], reverse=True)  # freshest gen_id first
        picked, self._items = self._items[:n], self._items[n:]
        return picked


class AsyncDiffusionTrainer(DiffusionTrainer):
    """Disaggregated async diffusion trainer (two slabs, resident engine, cross-slab sync)."""

    def __init__(
        self,
        *,
        max_inflight: int = 1,
        buffer_max_staleness: Optional[int] = None,
        **diffusion_kwargs: Any,
    ) -> None:
        # Async needs disjoint train/rollout slabs; force the separate layout.
        layout = diffusion_kwargs.setdefault("layout", "separate")
        if layout != "separate":
            raise ValueError(f"AsyncDiffusionTrainer requires layout='separate', got {layout!r}.")
        max_inflight = int(max_inflight)
        if max_inflight != 1:
            raise ValueError(
                "AsyncDiffusionTrainer requires max_inflight=1: multiple queued generations "
                "block the reap-time cross-slab transfer on the rollout workers; "
                f"got {max_inflight}."
            )
        super().__init__(**diffusion_kwargs)

        if self.weight_sync is None:
            raise ValueError(
                "AsyncDiffusionTrainer requires a cross-slab weight sync; add a `sync:` block to the recipe."
            )

        # ---- async state ----
        self._max_inflight = max_inflight
        self._buffer_max_staleness = buffer_max_staleness
        self._weight_version = 0  # driver-tracked policy version (# of weight syncs issued)
        # The rollout resp's single track key (e.g. "diffusion"), captured from the
        # first reaped generation so the reassembled resp keeps the same key.
        self._track_key: str = "diffusion"

    # ------------------------------------------------------------------
    # Non-blocking generate seam (split of the rollout Handle dispatch at ray.get;
    # mirrors AsyncARTrainer._generate_async — engine-agnostic).
    # ------------------------------------------------------------------

    def _generate_async(self, req: RolloutReq):
        """Launch ``generate`` non-blocking; return (refs, worker_local)."""
        r = self.rollout
        dispatch_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["dispatch_fn"]
        bs = infer_batch_size((req,), {})
        if bs is not None and bs % r.dp_size != 0:
            raise ValueError(f"req batch_size={bs} not divisible by rollout dp_size={r.dp_size}")
        shards = dispatch_fn(r, (req,), {}, bs)
        worker_local = issubclass(r.pool.transport_cls, WorkerLocalTransport)
        shards = r.pool.transport_cls.localize(shards, r.pool, r.device_ids, r.worker_ids)
        refs = r._execute_all("generate", shards, grad_mode=False, call_id=None)
        return refs, worker_local

    def _collect_resp(self, refs, worker_local) -> RolloutResp:
        """Join a completed generate → full RolloutResp (blocks in ray.get)."""
        r = self.rollout
        collect_fn = DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["collect_fn"]
        results = ray.get(refs)
        results = [r._rebind_tree(x, r.workers[i], worker_local=worker_local) for i, x in enumerate(results)]
        return collect_fn(r, results)

    @staticmethod
    def _is_ready(refs) -> bool:
        """True iff every worker's generate ref is resolved (non-blocking reap)."""
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=0)
        return len(ready) == len(refs)

    # ------------------------------------------------------------------
    # In-flight bookkeeping
    # ------------------------------------------------------------------

    def _launch(self, gen_id: int) -> None:
        """Build a request and launch one non-blocking generation."""
        req = self._build_req(self.data_source.get_samples(self.batch_size), gen_id)
        refs, worker_local = self._generate_async(req)
        self._inflight.append(
            {
                "refs": refs,
                "worker_local": worker_local,
                "req": req,
                "gen_id": gen_id,
                "weight_version": self._weight_version,
            }
        )

    def _score_into_buffer(self, rec: Dict[str, Any], resp: RolloutResp) -> None:
        """Score a completed generation and split its groups into the buffer.

        Scoring (``reward.score_and_attach``) is synchronous at reap time,
        before the next launch and training-batch consumption. It must precede
        ``_drop_decoded`` because the reward reads ``decoded``.
        """
        req = rec["req"]
        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)
        self._track_key = next(iter(resp.tracks))
        self._drop_decoded(req, resp, rollout_id=rec["gen_id"])
        (track,) = resp.tracks.values()
        for group in track.split():
            self._buffer.put(group, weight_version=rec["weight_version"], gen_id=rec["gen_id"])

    def _reap_ready(self) -> None:
        """Move every completed in-flight generation into the buffer (scored)."""
        still: List[Dict[str, Any]] = []
        for rec in self._inflight:
            if self._is_ready(rec["refs"]):
                self._score_into_buffer(rec, self._collect_resp(rec["refs"], rec["worker_local"]))
            else:
                still.append(rec)
        self._inflight = still

    def _drain_all(self) -> None:
        """Finish + buffer EVERY in-flight generation (single-threaded quiesce).

        Mandatory before a weight sync (a weight + KV update corrupts an in-flight
        generate), before eval/checkpoint (shared engine), and in ``finally``.
        """
        for rec in self._inflight:
            self._score_into_buffer(rec, self._collect_resp(rec["refs"], rec["worker_local"]))
        self._inflight = []

    def _next_batch(self, rollout_id: int, interval: int, M: int, stale: int, num_rollouts: int):
        """Top up launches, reap completed generations, and return the freshest
        ``batch_size`` groups (blocking on the oldest in-flight generation if the
        buffer is short).

        The launch clamp guarantees that ``stale=0`` never launches into a
        future sync window, so no generation crosses a regular rollout-weight
        sync boundary.
        """
        while True:
            # Reap (and cross-slab-transfer the completed generation's segment)
            # BEFORE launching the next one. The transfer runs on the rollout
            # worker as an NCCL send; if a fresh generation were already queued on
            # that worker (launch-first), the send would block behind it (~150s).
            # Reaping first gives the transfer an idle-worker window; the launch
            # below then starts the NEXT generation, which overlaps the caller's
            # train step. Contention-free as long as at most one generation is in
            # flight at the transfer instant (max_inflight=1).
            self._reap_ready()
            staleness_window = ((rollout_id // interval) + 1 + stale) * interval
            ceiling = min(num_rollouts, staleness_window)
            while self._launch_id < ceiling and len(self._inflight) < M:
                self._launch(self._launch_id)
                self._launch_id += 1

            picked = self._buffer.drain_freshest(
                self.batch_size, current_version=self._weight_version, max_staleness=stale
            )
            if picked is not None:
                return picked
            if self._inflight:
                ray.get(self._inflight[0]["refs"])  # block on oldest; next _reap_ready harvests it
            else:
                raise RuntimeError("async-diffusion: buffer underflow with no in-flight generations")

    # ------------------------------------------------------------------
    # Train tail (mirrors DiffusionTrainer.train_step's post-generate half:
    # advantage → FlowGRPO stack step; reward already attached at reap time).
    # ------------------------------------------------------------------

    def _advantage_and_train(
        self,
        track: RolloutTrack,
        resp: RolloutResp,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer step for a SCORED track (rewards already attached)."""
        if t0 is None:
            t0 = time.perf_counter()
        mean_reward = 0.0
        if track.rewards is not None:
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
        track = track.compute_advantages(normalize=True, use_global_std=self._adv_use_global_std)
        (name,) = resp.tracks.keys()  # single-track diffusion
        resp.tracks[name] = track
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(rollout_id, result, resp, step_time_s=time.perf_counter() - t0)
        # train_step is bypassed, so BaseTrainer's per-step reset hook never fires;
        # reclaim transport buffers here (no-op for colocate_store/gpu).
        self._reset_transport_buffers()
        return result, mean_reward

    # ------------------------------------------------------------------
    # Train loop
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "auto",
    ) -> None:
        interval = max(1, weight_sync_interval)
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        M = self._max_inflight

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Single-threaded: exactly one get_samples(batch_size) per launch and
        # launches are 1:1 with gen_id, so replaying start_rollout times restores
        # the exact stream position (deterministic resume).
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "max_inflight": M,
                "buffer_max_staleness": stale,
                "weight_sync_interval": interval,
                "train_fraction": self._train_fraction,
            },
        )

        self._buffer = _RolloutBuffer()
        self._inflight: List[Dict[str, Any]] = []
        self._launch_id = start_rollout

        if resumed and self.weight_sync is not None:
            self.weight_sync.sync()  # push restored weights into the fresh engine
        if self.eval_interval > 0:
            # Evaluate the policy already resident on the rollout slab. Eval must
            # neither advance the async weight version nor offload this engine.
            self.evaluate(start_rollout, sync_weights=False, sleep_after=False)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                picked = self._next_batch(rollout_id, interval, M, stale, num_rollouts)
                track = RolloutTrack.concat([p[0] for p in picked])
                resp = RolloutResp(tracks={self._track_key: track})
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    track, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    self._drain_all()  # eval shares the engine
                    self.evaluate(step, sync_weights=False, sleep_after=False)
                if save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts):
                    self._drain_all()  # consistent engine + deterministic resume
                    self.maybe_save_checkpoint(
                        rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                    )
                if step % interval == 0 and self.weight_sync is not None:
                    self._drain_all()  # MANDATORY: weight/KV update corrupts in-flight generations
                    self.weight_sync.sync()
                    self._weight_version += 1
        finally:
            # Match BaseTrainer._finish_wandb: cleanup failures must not mask
            # the exception that caused teardown.
            active_exception = sys.exc_info()[0] is not None
            try:
                self._drain_all()
            except Exception:
                if not active_exception:
                    raise
                logger.exception("Failed to drain in-flight generations during trainer teardown")
            finally:
                self._finish_wandb()
