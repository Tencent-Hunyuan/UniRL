"""Async autoregressive RL trainer — disaggregated train/rollout slabs.

Sibling of :class:`~unirl.trainer.ar.ARTrainer` (synchronous + *colocated*:
rollout engine and FSDP train shard time-share each GPU via ``sleep()/wake_up()``,
and every step runs ``generate → reward → train`` in series). ``AsyncARTrainer``
instead places training and rollout on **disjoint GPU slabs**, keeps the engine
**resident**, pushes weights cross-slab via ``NCCLWeightSync``, and overlaps
generation with training.

ONE single-threaded loop (slime's "one trainer loop; async-depth is a knob"
principle, implemented with UniRL-native non-blocking Ray dispatch instead of
slime's thread+asyncio). The async behavior is set by **two numeric knobs**:

* ``max_inflight`` — how many generations run concurrently (overlap/parallelism
  depth). ``1`` ≈ the classic one-step pipeline; higher fans out more.
* ``buffer_max_staleness`` — how many weight-syncs a buffered group may cross
  before it is evicted. ``0`` (default) = **on-policy**: the launch clamp never
  lets a generation cross a weight sync, so ``ratio≈1`` (the colocate-parity
  regime). ``>0`` = **off-policy continuous buffer**: generations may run ahead
  across syncs, bounded by eviction; the rollout-anchored DRPO ratio absorbs it.

Generation is launched as **non-blocking Ray futures** by
``RayGenerationDispatcher`` and reaped by ``AsyncRolloutScheduler`` on the
single driver thread — no producer thread, no locks. Draining all in-flight
generations before each weight sync is **mandatory** (the engine corrupts an
in-flight generation when weights + KV cache update mid-flight); this is the
single-threaded ``_drain_all`` quiesce.

Subclasses ``ARTrainer`` to reuse ``_build_req``/``evaluate`` and ``BaseTrainer``
plumbing, but ``__init__`` calls ``BaseTrainer.__init__`` **directly** (the parent
opens the colocate ``placement(fraction=1.0)`` block we replace with two slabs).
"""

import inspect
import logging
import sys
import time
from typing import Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.rollout.async_runtime import (
    AsyncRolloutScheduler,
    BufferedRolloutGroup,
    InflightGeneration,
    RayGenerationDispatcher,
)
from unirl.train.stack import TrainStepResult
from unirl.trainer.ar import ARTrainer
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


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
        eval_num_prompts: int = -1,
        eval_batch_size: int = 8,
        eval_samples_per_prompt: int = 16,
        eval_temperature: float = 1.0,
        # ---- async knobs ----
        train_fraction: float = 0.5,
        max_inflight: int = 1,
        buffer_max_staleness: Optional[int] = None,
    ) -> None:
        # Call BaseTrainer.__init__ directly: ARTrainer.__init__ opens the
        # colocate ``placement(fraction=1.0)`` block, which is exactly what we
        # must NOT run. (ARTrainer itself just calls BaseTrainer.__init__ here.)
        BaseTrainer.__init__(self, cfg=cfg, logging_cfg=logging_cfg)

        # ---- scalar/config fields (mirrors ar.py:62-88) ----
        self.batch_size = batch_size
        self.adv_normalization_scope = adv_normalization_scope
        self.normalize_adv_by_std = normalize_adv_by_std
        self.balance_shards = bool(balance_shards)
        self.eval_interval = int(eval_interval)
        _num = int(eval_num_prompts)
        self.eval_num_prompts = -1 if _num < 0 else _num
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_temperature = float(eval_temperature)
        self.data_source = instantiate(data_source_cfg)
        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self.weight_sync = None

        # ---- async state ----
        self._train_fraction = float(train_fraction)
        self._max_inflight = max(1, int(max_inflight))
        self._buffer_max_staleness = buffer_max_staleness
        self._weight_version = 0  # driver-tracked policy version (# of weight syncs issued)
        # DP size of the TRAIN slab — the divisor for balance_shards (the parent
        # uses self.num_devices because colocate training spans the whole pool;
        # here training only spans the train slab).
        self._train_devices = int(round(self.num_devices * self._train_fraction))
        if self._train_devices <= 0 or self._train_devices >= self.num_devices:
            raise ValueError(
                f"train_fraction={train_fraction} yields {self._train_devices} train "
                f"devices of {self.num_devices}; must leave a non-empty rollout slab."
            )
        # DP_SCATTER divisibility: per-rollout sample count must split evenly over
        # BOTH slabs (training over the train slab, generation over the rollout
        # slab). Fail early with a clear message rather than mid-run in dispatch.
        self._rollout_devices = self.num_devices - self._train_devices
        total = int(self.batch_size) * total_samples_per_prompt(self.sampling_params)
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
        itself. Only ``NCCLWeightSync`` is supported here (always cross-slab
        full-weight); a non-NCCL target is a config error.
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
    # Generic async-runtime hooks
    # ------------------------------------------------------------------

    def _build_async_req(self, gen_id: int) -> RolloutReq:
        """Consume one data batch and build the request for ``gen_id``."""
        return self._build_req(self.data_source.get_samples(self.batch_size), gen_id)

    def _score_completed(
        self,
        job: InflightGeneration,
        resp: RolloutResp,
    ) -> List[RolloutResp]:
        """Score a completed generation and split it into tree-complete groups.

        Scoring must precede ``_drop_decoded`` (the reward reads ``decoded``).
        Keyed by ``gen_id`` so media panels behave like the old pipeline path.
        """
        req = job.req
        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)
        self._drop_decoded(req, resp, rollout_id=job.gen_id)
        return resp.split()

    def _drain_all(self) -> None:
        """Finish + buffer EVERY in-flight generation (the single-threaded quiesce).

        Mandatory before a weight sync (the engine corrupts an in-flight generate
        when weights + KV cache update mid-flight), before eval/checkpoint (shared
        engine), and in ``finally`` (no leaked ObjectRefs).
        """
        self._async_scheduler.drain_all(self._score_completed)

    # ------------------------------------------------------------------
    # Train tail (mirrors ar.py:152-182, minus wake/sleep) — reward parity
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
        track = track.compute_advantages(normalize=self.normalize_adv_by_std, scope=self.adv_normalization_scope)
        (name,) = resp.tracks.keys()  # single-track for now; revisit if multi-track lands
        resp.tracks[name] = track
        if self.balance_shards:
            track = track.balance_shards(self._train_devices)  # over the TRAIN slab DP size
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(
            rollout_id,
            result,
            resp,
            step_time_s=time.perf_counter() - t0,
            trunc_len=getattr(self.sampling_params.get("ar"), "max_new_tokens", None),
        )
        # train_step is bypassed, so BaseTrainer's per-step reset hook never
        # fires; reclaim transport buffers here (no-op for colocate_store/gpu).
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
        save_mode: str = "full",
    ) -> None:
        interval = max(1, weight_sync_interval)
        # Staleness budget: how many weight-syncs a generation may cross before it
        # is evicted. 0 (default) = on-policy (no generation crosses a sync).
        stale = self._buffer_max_staleness if self._buffer_max_staleness is not None else 0
        M = self._max_inflight

        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Single-threaded: exactly one get_samples(batch_size) per launch and
        # launches are 1:1 with rollout_id, so replaying start_rollout times
        # restores the exact stream position (deterministic resume).
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "adv_normalization_scope": self.adv_normalization_scope,
                "max_inflight": M,
                "buffer_max_staleness": stale,
                "weight_sync_interval": interval,
            },
        )

        self._async_scheduler = AsyncRolloutScheduler(
            RayGenerationDispatcher(self.rollout),
            groups_per_step=self.batch_size,
        )
        self._async_scheduler.reset(start_rollout)

        if resumed and self.weight_sync is not None:
            self.weight_sync.sync()  # push restored weights into the fresh engine
        if self.eval_interval > 0:
            self.evaluate(rollout_id=-1)  # baseline; engine quiescent

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                picked = self._next_step(rollout_id, interval, M, stale, num_rollouts)
                group_tracks = []
                for item in picked:
                    (group_track,) = item.resp.tracks.values()
                    group_tracks.append(group_track)
                track = RolloutTrack.concat(group_tracks)
                resp = RolloutResp(tracks={"ar": track})
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    track, resp, training_progress=training_progress, rollout_id=rollout_id, t0=t0
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                if self.eval_interval > 0 and step % self.eval_interval == 0:
                    self._drain_all()  # eval shares the engine
                    self.evaluate(rollout_id=rollout_id)
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

    def _next_step(
        self,
        rollout_id: int,
        interval: int,
        M: int,
        stale: int,
        num_rollouts: int,
    ) -> List[BufferedRolloutGroup]:
        """Top up launches, reap completed generations, and return the freshest
        ``groups_per_step`` (``batch_size``) groups for ``rollout_id`` (blocking
        on the oldest in-flight generation if the buffer is short).

        The launch clamp is the load-bearing on-policy guarantee: a generation
        launched now is consumed later, so bound how far ahead we launch to
        ``stale`` weight-syncs. ``stale=0`` ⇒ never launch into a future
        sync-window ⇒ no generation crosses a sync ⇒ ``ratio≈1`` (on-policy).
        """
        return self._async_scheduler.next_step(
            rollout_id=rollout_id,
            sync_interval=interval,
            max_inflight=M,
            max_staleness=stale,
            num_rollouts=num_rollouts,
            current_version=self._weight_version,
            build_req=self._build_async_req,
            on_complete=self._score_completed,
        )
