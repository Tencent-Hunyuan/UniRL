"""Async autoregressive RL trainer — disaggregated train/rollout slabs.

Sibling of :class:`~unirl.trainer.ar.ARTrainer` (which is synchronous and
*colocated*: rollout engine and FSDP train shard time-share each GPU via
``sleep()/wake_up()``, and every step runs ``generate → reward → train`` in
series). ``AsyncARTrainer`` instead places training and rollout on **disjoint
GPU slabs** and overlaps generation with training, porting slime's two async
modes:

* ``rollout_mode="pipeline"`` (v0, slime ``train_async.py``): keep ONE
  generation in flight — launch ``generate(N+1)`` before ``train(N)`` and drain
  it before each weight sync. Overlap appears only when ``weight_sync_interval >
  1`` (with ``=1`` a sync sits between every pair, so it degenerates to the
  on-policy ``collect → train → sync → launch`` sequence; that is the
  reward-parity baseline vs ``ARTrainer``).
* ``rollout_mode="buffer"`` (v1, slime ``fully_async_rollout``): a background
  producer thread continuously generates + scores into a group-keyed buffer;
  training drains the freshest ``batch_size`` groups each step. Weights are
  pushed every ``weight_sync_interval`` via a quiesce-drain swap (the producer's
  in-flight generation finishes, then the engine receives weights).

This subclasses ``ARTrainer`` to reuse ``_build_req`` / ``evaluate`` and all the
``BaseTrainer`` plumbing, but its ``__init__`` calls ``BaseTrainer.__init__``
**directly** (NOT ``ARTrainer.__init__``) because the parent opens the colocate
``placement(fraction=1.0)`` block we must replace with two disjoint slabs. The
engine stays **resident** (no ``sleep()/wake_up()``), and weights cross the slab
boundary via ``NCCLWeightSync`` instead of the colocate ``TensorWeightSync``.

Thread-safety: the off-thread ``generate`` is safe because the autograd context
is thread-local (``grad_context._tls``) so the generating thread always sees no
``enable_grad`` (and never touches the rollout Handle's grad call-counter), and
the producer is the SOLE caller of the rollout Handle. Engine access (generate
vs. ``weight_sync.sync()`` — which also RPCs the rollout workers) is serialized
by ``self._engine_lock``, never by sharing a Handle.
"""

import inspect
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer.base import BaseTrainer
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class _RolloutBuffer:
    """Thread-safe group-keyed rollout buffer (slime ``BufferQueue`` analogue).

    Each entry is one prompt's GRPO group — a ``RolloutTrack`` of
    ``samples_per_prompt`` already-scored samples — stamped with the
    ``weight_version`` it was generated under and a monotonic ``gen_id`` for
    freshness ordering. Groups are always complete (the whole ``generate``
    finished before they are ``put``), so there is no partial-group bookkeeping.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[Tuple[RolloutTrack, int, int]] = []  # (group, weight_version, gen_id)

    def put(self, track: RolloutTrack, *, weight_version: int, gen_id: int) -> None:
        with self._lock:
            self._items.append((track, int(weight_version), int(gen_id)))

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def drain_freshest(
        self,
        n: int,
        *,
        current_version: Optional[int] = None,
        max_staleness: Optional[int] = None,
    ) -> Optional[List[Tuple[RolloutTrack, int, int]]]:
        """Pop the ``n`` freshest complete groups, carrying leftovers forward.

        Returns ``None`` if fewer than ``n`` groups are available. When
        ``max_staleness`` is set, groups older than ``current_version -
        max_staleness`` weight versions are evicted first (bounded off-policy).
        """
        with self._lock:
            if max_staleness is not None and current_version is not None:
                self._items = [it for it in self._items if current_version - it[1] <= max_staleness]
            if len(self._items) < n:
                return None
            self._items.sort(key=lambda it: it[2], reverse=True)  # freshest gen_id first
            picked, self._items = self._items[:n], self._items[n:]
            return picked


class AsyncARTrainer(ARTrainer):
    """Disaggregated async AR trainer (two slabs, resident engine, NCCL sync)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        adv_normalization_scope: str = "group",
        normalize_adv_by_std: bool = True,
        balance_shards: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = 60,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        # ---- async-specific knobs ----
        train_fraction: float = 0.5,
        rollout_mode: str = "pipeline",
        buffer_max_staleness: Optional[int] = None,
    ) -> None:
        # Call BaseTrainer.__init__ directly: ARTrainer.__init__ opens the
        # colocate ``placement(fraction=1.0)`` block, which is exactly what we
        # must NOT run. (ARTrainer itself just calls BaseTrainer.__init__ here,
        # so this honors the base contract.)
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        # ---- scalar/config fields (mirrors ar.py:62-88 verbatim) ----
        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)
        self.weight_sync = None

        # ---- async-specific state ----
        self._rollout_mode = str(rollout_mode)
        self._train_fraction = float(train_fraction)
        self._buffer_max_staleness = buffer_max_staleness
        # Driver-tracked policy version (# of weight syncs the driver issued) and
        # producer generation counter (# of generate-batches consumed from the
        # data stream). Used by buffer mode; initialized here so they always exist.
        self._weight_version = 0
        self._gen_count = 0
        # DP size of the TRAIN slab — the divisor for balance_shards (the parent
        # uses self.num_devices because colocate training spans the whole pool;
        # here training only spans the train slab).
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        # DP_SCATTER divisibility: the per-rollout sample count must split evenly
        # across BOTH slabs (training dispatches over the train slab, generation
        # over the rollout slab). Fail early with a clear message rather than mid
        # run inside Handle dispatch.
        self._rollout_devices = self.num_devices - self._train_devices
        total = int(self.batch_size) * int(self.sampling_params.samples_per_prompt)
        for slab_name, slab in (("train", self._train_devices), ("rollout", self._rollout_devices)):
            if total % slab != 0:
                raise ValueError(
                    f"batch_size * samples_per_prompt = {total} is not divisible by the "
                    f"{slab_name} slab size {slab}; adjust batch_size / samples_per_prompt / train_fraction."
                )

        # ---- two disjoint top-level slabs (diffusion.py:115-129 template) ----
        # The train scope must FULLY EXIT before the rollout scope opens, else a
        # nested placement would carve a sub-slab instead of a disjoint slab.
        with placement(self.pool, fraction=self._train_fraction, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline)
            self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)
            if sync_cfg is not None:
                # NCCL handler: rollout is cross-slab and wired via the handshake
                # below — it takes only ``backend`` (no rollout sibling).
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
        # Rollout slab = the rest (fraction is relative to the WHOLE pool).
        with placement(self.pool, fraction=1.0 - self._train_fraction, shared_workers=True):
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
                raise ValueError(
                    "AsyncARTrainer needs a dedicated-rollout engine (vllm/sglang) on the "
                    "separate slab; the trainside direct-sampling engine needs the pipeline "
                    "as a local sibling and cannot live cross-slab."
                )
            self.rollout = remote(**rollout_parsed)

        if self.weight_sync is not None:
            self._connect_separate(sync_cfg)

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake (NCCL branch of diffusion.py:191-208).

        Rank 0 picks a rendezvous addr/port, is handed the rollout slab's Worker
        actor handles, then ``connect`` fires each rollout worker's
        ``init_weights_update_group`` non-blocking and joins the broadcast group
        itself. Only ``NCCLWeightSync`` is supported here (the AR async path is
        always cross-slab full-weight); a non-NCCL target is a config error.
        """
        target = str(sync_cfg.get("_target_", ""))
        if not target.endswith("NCCLWeightSync"):
            raise ValueError(
                f"AsyncARTrainer (separate slabs) requires a cross-slab weight sync "
                f"(NCCLWeightSync); got sync._target_={target!r}."
            )
        addr, port = self.weight_sync.pick_master()[0]
        self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
        self.weight_sync.connect(
            master_addr=addr,
            master_port=port,
            num_rollout_gpus=len(self.rollout.workers),
        )

    # ------------------------------------------------------------------
    # Shared post-generate path (mirrors ar.py:148-182, minus wake/sleep)
    # ------------------------------------------------------------------

    def _gen(self, req: RolloutReq) -> RolloutResp:
        """Blocking generation on the resident engine (run off the main thread).

        NO ``wake_up()/sleep()`` — disaggregated, the engine owns its GPUs and
        must stay hot (and a continuous producer needs it resident).
        """
        return self.rollout.generate(req)

    def _advantage_and_train(
        self,
        track: RolloutTrack,
        resp: RolloutResp,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
    ) -> Tuple[TrainStepResult, float]:
        """Advantage + optimizer step for a SCORED track (rewards already attached).

        Shared tail of both async modes and the parity-critical half of
        ``ARTrainer.train_step`` (ar.py:152-182): hydrate the mean reward, GRPO
        group-normalize advantages, (optionally) balance shards, then one
        ``train_track`` optimizer step. ``resp`` carries the same single track,
        for the wandb panels.
        """
        if t0 is None:
            t0 = time.perf_counter()
        mean_reward = 0.0
        if track.rewards is not None:
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
        track = track.compute_advantages(
            normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope
        )
        (name,) = resp.tracks.keys()  # single-track for now; revisit if multi-track lands
        resp.tracks[name] = track
        if self.balance_shards:
            # Balance over the TRAIN slab's DP size (not the whole pool).
            track = track.balance_shards(self._train_devices)
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            resp,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params, "max_new_tokens", None),
        )
        # train_step is bypassed, so BaseTrainer's per-step reset hook never
        # fires; reclaim transport buffers here (no-op for colocate_store/gpu).
        self._reset_transport_buffers()
        return result, mean_reward

    def _score_and_train(
        self,
        req: RolloutReq,
        resp: RolloutResp,
        *,
        training_progress: float,
        rollout_id: int,
        t0: Optional[float] = None,
    ) -> Tuple[TrainStepResult, float]:
        """``reward → advantage → optimizer step`` for an already-generated resp.

        Identical call sequence to ``ARTrainer.train_step`` (ar.py:148-182) so
        the learning signal is parity by construction; only the generation half
        (wake/sync/generate/sleep) is replaced by the async driver. Used by v0
        (the producer scores inline in v1).
        """
        if t0 is None:
            t0 = time.perf_counter()
        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)
        self._drop_decoded(req, resp, rollout_id=rollout_id)
        (track,) = resp.tracks.values()
        return self._advantage_and_train(
            track, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
        )

    # ------------------------------------------------------------------
    # Train entrypoint
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        save_interval: int = 0,
        save_dir: Optional[str] = None,
        load_dir: Optional[str] = None,
        save_mode: str = "full",
    ) -> None:
        kwargs = dict(
            num_rollouts=num_rollouts,
            weight_sync_interval=weight_sync_interval,
            save_interval=save_interval,
            save_dir=save_dir,
            load_dir=load_dir,
            save_mode=save_mode,
        )
        if self._rollout_mode == "pipeline":
            self._train_pipeline(**kwargs)
        elif self._rollout_mode == "buffer":
            self._train_buffer(**kwargs)
        else:
            raise ValueError(f"unknown rollout_mode={self._rollout_mode!r} (expected 'pipeline' or 'buffer')")

    # ------------------------------------------------------------------
    # v0 — one-step pipeline (slime train_async.py)
    # ------------------------------------------------------------------

    def _train_pipeline(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int,
        save_interval: int,
        save_dir: Optional[str],
        load_dir: Optional[str],
        save_mode: str,
    ) -> None:
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope, "async_mode": "pipeline"},
        )

        def build(rid: int) -> RolloutReq:
            return self._build_req(self.data_source.get_samples(self.batch_size), rid)

        ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="async-ar-gen")
        try:
            req = build(start_rollout)
            if resumed and self.weight_sync is not None:
                # Engine booted fresh; push the restored weights before generate.
                self.weight_sync.sync()
            if self.eval_interval > 0:
                self.evaluate(rollout_id=-1)  # baseline, engine quiescent (no in-flight gen yet)
            fut = ex.submit(self._gen, req)

            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                next_id = rollout_id + 1
                has_next = next_id < num_rollouts
                sync_due = has_next and next_id % interval == 0
                eval_this = self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0
                # Defer the next launch when a sync (stale weights) or an eval
                # (engine contention) sits between this train and the next gen —
                # so generation never overlaps a weight swap or eval.
                defer_launch = sync_due or eval_this
                next_req = next_fut = None
                if has_next and not defer_launch:
                    next_req = build(next_id)
                    next_fut = ex.submit(self._gen, next_req)

                resp = fut.result()  # drain gen(rollout_id)
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._score_and_train(
                    req, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                if eval_this:
                    self.evaluate(rollout_id=rollout_id)  # engine quiescent (launch deferred)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )

                if has_next and defer_launch:
                    if sync_due and self.weight_sync is not None:
                        self.weight_sync.sync()
                    next_req = build(next_id)
                    next_fut = ex.submit(self._gen, next_req)
                if next_fut is not None:
                    fut, req = next_fut, next_req
        finally:
            ex.shutdown(wait=True)
            self._finish_wandb()

    # ------------------------------------------------------------------
    # v1 — continuous rollout buffer (slime fully_async_rollout)
    # ------------------------------------------------------------------

    def _producer_loop(self) -> None:
        """Background producer: continuously generate + score into the buffer.

        Sole caller of the rollout Handle. Holds ``_engine_lock`` only around
        ``generate`` so a consumer-side ``weight_sync.sync()`` (the swap) is
        serialized against in-flight generation. Backpressure: idle when the
        buffer is full. Any exception is recorded and stops the loop so the
        consumer can re-raise instead of waiting forever.
        """
        try:
            while not self._stop.is_set():
                if self._buffer.size() >= self._max_buffer_groups:
                    time.sleep(0.02)
                    continue
                with self._data_lock:
                    if self._stop.is_set():
                        break
                    inputs = self.data_source.get_samples(self.batch_size)
                    self._gen_count += 1
                    gid = self._gen_count
                req = self._build_req(inputs, gid)
                with self._engine_lock:
                    if self._stop.is_set():
                        break
                    wv = self._weight_version  # the policy version these samples are drawn from
                    resp = self._gen(req)
                track = resp.tracks["ar"]
                track = self.reward.score_and_attach(req=req, track=track)  # off-lock; reward actor
                # Free the per-rollout generated payloads (consumed by scoring;
                # training reads only segment/advantages). No media logging in
                # buffer mode.
                track.decoded = None
                track.media_preview = None
                for group in track.split():
                    self._buffer.put(group, weight_version=wv, gen_id=gid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("async-ar producer thread crashed")
            self._producer_exc = exc
            self._stop.set()

    def _swap_weights(self) -> None:
        """Quiesce-drain weight swap: push new weights into the resident engine.

        Acquiring ``_engine_lock`` drains any in-flight ``generate`` (the
        producer releases the lock when its current batch completes); ``sync``
        then broadcasts into a quiescent engine and the driver policy version
        advances so newly produced groups are stamped fresh.
        """
        with self._engine_lock:
            self.weight_sync.sync()
            self._weight_version += 1

    def _async_state_path(self, save_dir: Optional[str], step: int) -> str:
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        return os.path.join(base_dir, f"checkpoint-{step}", "async_state.json")

    def _resume_gen_count(self, load_dir: Optional[str]) -> int:
        if not load_dir:
            return 0
        path = os.path.join(os.path.abspath(load_dir), "async_state.json")
        if not os.path.exists(path):
            return 0
        with open(path) as f:
            return int(json.load(f).get("gen_count", 0))

    def _train_buffer(
        self,
        *,
        num_rollouts: int,
        weight_sync_interval: int,
        save_interval: int,
        save_dir: Optional[str],
        load_dir: Optional[str],
        save_mode: str,
    ) -> None:
        if self.weight_sync is None:
            raise ValueError("buffer mode requires a cross-slab weight sync (set the `sync` block).")
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        # Resume: the producer consumes the data stream at its OWN pace (decoupled
        # from rollout_id), so fast-forward by the recorded producer gen_count
        # rather than by start_rollout (which counts train steps).
        gen_count0 = self._resume_gen_count(load_dir)
        for _ in range(gen_count0):
            self.data_source.get_samples(self.batch_size)
        self._gen_count = gen_count0
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={"adv_normalization_scope": self.adv_normalization_scope, "async_mode": "buffer"},
        )

        # Concurrency state (consumed by _producer_loop / _swap_weights).
        self._buffer = _RolloutBuffer()
        self._engine_lock = threading.Lock()
        self._data_lock = threading.Lock()
        self._stop = threading.Event()
        self._producer_exc: Optional[BaseException] = None
        self._max_buffer_groups = 2 * self.batch_size

        # Push current train weights into the fresh engine before producing
        # (required on resume; redundant-but-harmless on a fresh run).
        self.weight_sync.sync()
        if self.eval_interval > 0:
            self.evaluate(rollout_id=-1)  # baseline, producer not started yet

        producer = threading.Thread(target=self._producer_loop, name="async-ar-producer", daemon=True)
        producer.start()
        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                # Wait for a full batch of freshest complete groups.
                picked = None
                while picked is None:
                    if self._producer_exc is not None:
                        raise RuntimeError("async-ar rollout producer died") from self._producer_exc
                    picked = self._buffer.drain_freshest(
                        self.batch_size,
                        current_version=self._weight_version,
                        max_staleness=self._buffer_max_staleness,
                    )
                    if picked is None:
                        time.sleep(0.05)

                track = RolloutTrack.concat([p[0] for p in picked])
                resp = RolloutResp(tracks={"ar": track})
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    track, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    with self._engine_lock:  # quiesce producer; eval shares the engine
                        self.evaluate(rollout_id=rollout_id)

                step = rollout_id + 1
                if save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts):
                    # Quiesce the producer so the recorded gen_count + engine are
                    # consistent; buffered-but-unconsumed groups are dropped on
                    # resume (matches slime — the buffer is not checkpointed).
                    with self._engine_lock, self._data_lock:
                        self.maybe_save_checkpoint(
                            rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                        )
                        path = self._async_state_path(save_dir, step)
                        if os.path.isdir(os.path.dirname(path)):
                            with open(path, "w") as f:
                                json.dump({"gen_count": self._gen_count}, f)

                if step % interval == 0:
                    self._swap_weights()
        finally:
            self._stop.set()
            producer.join(timeout=60)
            self._finish_wandb()
