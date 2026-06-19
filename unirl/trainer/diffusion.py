import dataclasses
import inspect
import logging
import os
import time
from typing import Dict, Optional, Tuple

import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


class DiffusionTrainer(BaseTrainer):
    """Reference trainer: train + rollout colocated on the whole pool.

    For separate slabs, open two sibling ``placement`` blocks with
    ``fraction<1.0``. For real-colocate (distinct worker processes on the
    same GPU), nest a ``placement(..., shared_workers=False)`` inside.
    """

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
        layout: str = "colocate",
        train_fraction: float = 0.5,
        enable_fsdp_offload: bool = False,
        adv_use_global_std: bool = False,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._layout = str(layout)
        # Colocate memory dance: offload the FSDP train state (params + grads +
        # optimizer) to CPU during the rollout's generate so a colocate
        # vLLM/SGLang engine fits, onload before the train backward. Off by
        # default; only safe (and only set true) for layout=="colocate" with a
        # SEPARATE engine rollout under GRPO — gated again in train_step.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        # FlowDPPO advantage parity: when True, RolloutTrack.compute_advantages
        # keeps the per-group mean but divides by ONE batch-wide std (the v1
        # ``use_global_std=True`` scale) instead of each prompt's own std. Off by
        # default → unchanged per-group GRPO normalization for every other recipe.
        self._adv_use_global_std = bool(adv_use_global_std)
        # Set in _build_rollout: True when the rollout is the trainside
        # direct-sampling engine (it reuses the train model → must NOT offload).
        self._rollout_is_trainside = False
        # Set in _build_train_side: True only for the DiffusionNFT algorithm, which
        # needs the EMA dual-adapter swap around rollout. Stays False for GRPO
        # so its hot path is untouched.
        self._uses_ema = False

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Per-sample latent shape for the driver-authored x_T recipe (see
        # _build_req), resolved ONCE here via the pipeline's framework-level
        # ``latent_shape`` classmethod — each model contributes its OWN geometry
        # instead of a hardcoded SD3 shape.
        #
        # SCOPE CAVEAT: a recipe is AUTHORED for every pipeline that implements
        # ``latent_shape`` (all of them today), but recipe CONSUMPTION is wired
        # end-to-end only for SD3 (``SD3Pipeline.diffuse`` trainside + the SD3
        # vllm path + the model-agnostic sglang engine). A non-SD3 model gets a
        # correctly-shaped recipe that today only the sglang engine consumes — no
        # crash and no within-run divergence (trainside replays the rollout's
        # returned trajectory), but it does NOT yet get the full cross-engine x_T
        # guarantee. To extend a model: wire recipe consumption into its trainside
        # pipeline + vllm translator (mirroring SD3). ``None`` ⇒ no recipe
        # (DISABLE_DRIVER_XT, or ``latent_shape`` raised ``NotImplementedError``)
        # and every engine falls back to its own RNG.
        self._noise_latent_shape: Optional[list] = (
            None
            if os.environ.get("DISABLE_DRIVER_XT")
            else self._resolve_noise_latent_shape(pipeline_cfg=pipeline_cfg, model_cfg=bundle_cfg)
        )

        # Set below from the `sync` block; None trainside (shares the module).
        self.weight_sync = None

        # Construction (_build_train_side / _build_rollout) is shared; only the
        # placement topology and the train→rollout sync wiring differ per layout.
        train_cfgs = dict(
            bundle_cfg=bundle_cfg,
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            reward_cfg=reward_cfg,
            algorithm_cfg=algorithm_cfg,
            stack_cfg=stack_cfg,
        )
        if self._layout == "separate":
            # Two disjoint top-level slabs. A nested placement would carve a
            # sub-slab of the parent (not a disjoint slab), so the train scope
            # must fully exit before the rollout scope opens.
            with placement(self.pool, fraction=train_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                if sync_cfg is not None:
                    # NCCL handler: rollout is cross-slab, wired via the handshake below.
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
            # Rollout slab = the rest. Top-level ``fraction`` is relative to the
            # WHOLE pool (placement.py), so the remainder is ``1 - train_fraction``.
            with placement(self.pool, fraction=1.0 - train_fraction, shared_workers=True):
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=False)
            if self.weight_sync is not None:
                self._connect_separate(sync_cfg)
        else:
            # Single slab: train + rollout are siblings on one Worker.
            with placement(self.pool, fraction=1.0, shared_workers=True):
                self._build_train_side(**train_cfgs)
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=True)
                if sync_cfg is not None:
                    # Colocated handlers (tensor/ipc) take the engine as a local sibling.
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

    def _build_train_side(
        self,
        *,
        bundle_cfg,
        pipeline_cfg,
        backend_cfg,
        reward_cfg,
        algorithm_cfg,
        stack_cfg,
    ) -> None:
        """Build the train-side remotes in the *currently active* placement scope.

        Scope-agnostic: ``remote_hydra`` lands each remote in whatever
        ``placement(...)`` block is open, so both layouts reuse this.
        """
        self.bundle = remote_hydra(bundle_cfg)
        self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
        self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
        self.reward = remote_hydra(reward_cfg)
        # DiffusionNFT resolves its frozen reference adapter off ``backend.ema`` (the
        # FSDPBackend owns the dual-adapter EMA), so it needs the backend sibling
        # injected alongside ``pipeline``. GRPO takes neither and would reject the
        # extra kwarg, so gate on the algorithm's declared ``requires_ema_rollout``
        # (off-policy algorithms set it True). The same flag drives the eval-EMA
        # swap around ``generate`` in ``train_step``: on-policy algorithms MUST
        # sample with the trainable weights so the first-step importance ratio is 1.
        algo_cls = get_class(str(algorithm_cfg.get("_target_", "")))
        self._uses_ema = getattr(algo_cls, "requires_ema_rollout", False)
        algo_extra = {"backend": self.backend} if self._uses_ema else {}
        self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline, **algo_extra)
        self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

    def _build_rollout(self, rollout_cfg, *, allow_pipeline: bool):
        """Build the rollout remote in the currently active placement scope.

        The trainside direct-sampling engine takes ``pipeline`` as a local
        sibling and is only valid colocated (``allow_pipeline=True``); vllm /
        sglang engines take no pipeline and work in either layout.
        """
        rollout_parsed = parse_hydra_cfg(rollout_cfg)
        if "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters:
            if not allow_pipeline:
                raise ValueError(
                    "layout='separate' requires a dedicated-rollout engine "
                    "(vllm/sglang); the trainside direct-sampling engine needs "
                    "the pipeline as a local sibling and cannot live on a "
                    "separate slab."
                )
            self._rollout_is_trainside = True
            # The trainside engine samples through the SAME SP-parallelized model
            # the backend wraps, so its DP_SCATTER must shard over the model's
            # dp_size (not all world ranks) — else the two ranks of an SP pair get
            # different prompts and the sampling Ulysses all-to-all hangs. Inherit
            # the backend's SP degree (sp_size is a handle-layout hint, stripped
            # before the engine __init__; the pipeline sibling alone is flat).
            return remote(**rollout_parsed, pipeline=self.pipeline, sp_size=self.backend.sp_size)  # direct sampling
        return remote(**rollout_parsed)  # vllm / sglang

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake: hand rank 0 the rollout Worker handles.

        Driver-orchestrated because the rollout slab is cross-slab (not a
        sibling). The LoRA-over-Ray handler (``RemoteLoraWeightSync``) only needs
        the rollout engine's ``(role, workers)`` to push adapters by Ray RPC.
        ``NCCLWeightSync`` additionally rendezvous a broadcast group: ``pick_master``
        on rank 0, hand it the rollout Worker handles, then ``connect`` (rank 0
        fires the rollout joins non-blocking, then joins the group itself).
        """
        if str(sync_cfg.get("_target_", "")).endswith("NCCLWeightSync"):
            addr, port = self.weight_sync.pick_master()[0]
            self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
            self.weight_sync.connect(
                master_addr=addr,
                master_port=port,
                num_rollout_gpus=len(self.rollout.workers),
            )
        else:
            self.weight_sync.set_rollout_targets([(self.rollout.role_name, self.rollout.workers)])

    def _resolve_noise_latent_shape(self, *, pipeline_cfg: DictConfig, model_cfg: DictConfig) -> Optional[list]:
        """Per-sample latent shape for the driver-authored x_T recipe, or ``None``.

        Delegates to the pipeline's ``latent_shape`` classmethod — the framework's
        driver-side :class:`~unirl.models.types.pipeline.LatentShapeProvider`
        contract — so each model returns its OWN geometry (SD3 ``(16, H/8, W/8)``,
        WAN a 5D video shape, Flux a 128-ch packed shape, …) and no model-specific
        shape is baked into this generic trainer. A pipeline opts out of
        driver-authored noise by raising ``NotImplementedError`` (→ ``None`` →
        engines draw their own x_T). Any OTHER exception (e.g. an invalid frame
        count) propagates — that is a real config error, not an opt-out.

        In practice every shipped pipeline returns a shape, so a recipe is
        authored for all models; recipe *consumption* is currently SD3-only (see
        the scope caveat in ``__init__``).
        """
        target = getattr(pipeline_cfg, "_target_", None)
        if not isinstance(target, str):
            return None
        pipeline_cls = get_class(target)
        latent_shape_fn = getattr(pipeline_cls, "latent_shape", None)
        if latent_shape_fn is None:
            return None
        try:
            shape = latent_shape_fn(model_config=model_cfg, sampling_spec=self.sampling_params.get("diffusion"))
        except NotImplementedError:
            return None
        return [int(x) for x in shape]

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data source batch into a typed :class:`RolloutReq`.

        Expands ``inputs`` by ``total_samples_per_prompt(sampling_params)`` so
        each prompt produces an N-sample GRPO group (sibling samples consecutive,
        sample IDs ``prompt:<gid>:sample:<j>``).

        ``rollout_id`` keys the SDE step scheduler (``get_sde_indices``): the
        resolved indices are stamped onto a per-request copy of the diffusion
        sampling params, and the schedule config itself is nulled so only the
        resolved ``sde_indices`` ride to the engine.
        """
        inputs = inputs.expand(total_samples_per_prompt(self.sampling_params))
        diffusion = self.sampling_params.get("diffusion")
        sde_indices = diffusion.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diffusion, sde_indices=sde_indices, scheduler=None)
        sampling_params = {**self.sampling_params, "diffusion": diffusion}
        # Driver-authoritative x_T, shipped as a deterministic RECIPE. The driver
        # is the single source of initial noise: it authors per-sample noise group
        # ids keyed on (rollout_id, STABLE sample/group id); base_seed rides on
        # sampling_params.seed and the latent shape is the pipeline's own geometry
        # (self._noise_latent_shape, resolved once in __init__). Each engine
        # regenerates the BYTE-IDENTICAL x_T from this recipe via regen_initial_noise
        # (generate_shared_noise pinned to CPU-fp32, then moved to the engine device
        # — CPU randn is bit-stable across machines for a fixed torch version, which
        # is what makes trainside / vllm / sglang agree to the byte; verified across
        # nodes+clusters on torch 2.11.0).
        # So x_T is:
        #   - per-rollout-VARYING (rollout_id in the key) → genuine exploration,
        #   - per-sample-UNIQUE   (stable sample id in the key) → diverse GRPO groups,
        #   - IDENTICAL across engines for a given (seed, rollout) → curves align,
        #   - reproducible under resume / re-shard / re-batch (ids are STABLE, not a
        #     positional batch index, so a sample keeps its x_T wherever it lands).
        # ``init_same_noise=True`` keys by prompt group instead (siblings share).
        # Root cause this fixes: each engine used to draw its OWN x_T from independent
        # RNG → divergent reward curves; a single driver-authored x_T removes that.
        # Opt out with DISABLE_DRIVER_XT=1 (resolved in __init__ → shape None here).
        init_noise_group_ids: list = []
        init_noise_latent_shape = self._noise_latent_shape
        if init_noise_latent_shape is not None:
            if bool(getattr(self.sampling_params.get("diffusion"), "init_same_noise", False)):
                init_noise_group_ids = [f"r{rollout_id}:{g}" for g in inputs.group_ids]
            else:
                init_noise_group_ids = [f"r{rollout_id}:{s}" for s in inputs.sample_ids]
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
            init_noise_group_ids=init_noise_group_ids,
            init_noise_latent_shape=init_noise_latent_shape,
        )

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[TrainStepResult, float]:
        """One ``rollout → reward → advantage → optimizer step`` pass.

        ``training_progress`` in ``[0, 1]`` drives clip-range / LR schedules
        inside the algorithm. The reference trainer is stateless — the
        outer training loop owns step counting; ``rollout_id`` only keys the
        wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).

        ``sync_weights`` pushes the latest LoRA into the engine between
        ``wake_up`` and ``generate`` — one wake/sleep instead of two, with this
        ``generate`` already using the fresh adapter.

        Returns ``(train_result, mean_reward)`` — the mean unnormalized
        per-sample reward of the single track (0.0 if none), for the log line.
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        # Colocate FSDP offload: free the train state (params+grads+optimizer)
        # for the memory-heavy generate when a SEPARATE engine does the rollout.
        # Gated off for the trainside rollout (reuses the train model → can't be
        # offloaded) and for DiffusionNFT (``_uses_ema``; its EMA adapter swap touches the
        # backend around generate). Off by default. ``sync`` above needs the base
        # onloaded, so offload only AFTER it.
        _do_fsdp_offload = (
            self._enable_fsdp_offload
            and self._layout != "separate"
            and not self._rollout_is_trainside
            and not self._uses_ema  # _uses_ema == "is DiffusionNFT"
        )
        if _do_fsdp_offload:
            self.backend.offload()
        # DiffusionNFT samples under the EMA-smoothed ("old") adapter. HOW "old"
        # reaches the rollout depends on topology, so each mechanism fires only in
        # its own regime (never both):
        #   - trainside engine: it reuses THIS process's model, so swap the adapter
        #     in place around generate and restore "default" before the loss.
        #   - separate engine (sglang/vllm): runs in its own process and receives
        #     "old" via the weight sync's merged push (backend.rollout_adapter_name);
        #     the in-process swap cannot reach it, so skip the wasted swap + RPC.
        # No-op for GRPO (gated on _uses_ema).
        _inproc_ema_swap = self._uses_ema and self._rollout_is_trainside
        if _inproc_ema_swap:
            self.backend.apply_eval_ema()
        resp = self.rollout.generate(req)
        if _inproc_ema_swap:
            self.backend.restore_from_eval()
        self.rollout.sleep()
        if _do_fsdp_offload:
            self.backend.onload()

        for name, track in list(resp.tracks.items()):
            if track.segment is not None:
                resp.tracks[name] = self.reward.score_and_attach(req=req, track=track)

        mean_reward = 0.0
        for track in resp.tracks.values():
            if track.rewards is None:
                continue
            # Hydrate in place so the wandb reward/advantage stats reuse this
            # fetch instead of re-pulling the TensorRef from the worker.
            track.rewards = hydrate(track.rewards)
            mean_reward = float(track.rewards.to(torch.float32).mean().item())
            break  # single-track for now; revisit if multi-track lands

        for name, track in list(resp.tracks.items()):
            if track.rewards is not None:
                resp.tracks[name] = track.compute_advantages(normalize=True, use_global_std=self._adv_use_global_std)

        self._drop_decoded(req, resp, rollout_id=rollout_id)
        (track,) = resp.tracks.values()
        result = self.stack.train_track(track, training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(rollout_id, result, resp, step_time_s=time.perf_counter() - t0)
        return result, mean_reward

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
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into ``train_step``'s generate; no-op trainside).

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` is the output folder (defaults to
        ``./checkpoints``); ``save_mode="auto"`` writes LoRA-only checkpoints
        when LoRA is active and full checkpoints otherwise.
        ``load_dir``: restore from a checkpoint directory and RESUME from its
        saved step — ``num_rollouts`` is the TOTAL budget, so resuming
        checkpoint-500 with ``num_rollouts=600`` runs rollouts 500..599.
        """
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate.
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )
                result, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
