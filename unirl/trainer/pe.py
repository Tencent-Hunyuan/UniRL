"""UniRL v2 PE (Prompt Enhancement) joint trainer.

Two :class:`~unirl.train.stack.TrainStack` siblings — one for the
diffusion model, one for the AR LLM — colocated on the whole pool, sharing the
composed :class:`~unirl.models.pe.pipeline.PEPipeline` as a *trainside*
rollout (the rollout reads the live FSDP modules, so no weight sync).

One ``train_step``::

    rollout.generate(req)           → 2-track RolloutResp {"ar", "diffusion"}
    reward.score_and_attach(image)  → score the "diffusion" (image) track only
    resp.propagate_rewards("mean")  → credit-assign image reward up to "ar"
    track.compute_advantages()      → per-track GRPO (ar by prompt, diff by rewrite)
    {name}.stack.train_track(track) → route each track to its own model

Mirrors :class:`~unirl.trainer.diffusion.DiffusionTrainer` but wires two
of everything and a composed rollout. Deferred (same as the reference trainer):
multi-epoch replay, checkpoint / eval cadence, structured logging.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.models.pe.pipeline import PEPipeline
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.sampling import BaseSamplingParams, get_diffusion_params
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names match PEPipeline's output and the per-side attributes on the
# trainer (``self.ar`` / ``self.diffusion``); also the algorithms' stage_attr.
TRACK_NAMES: Tuple[str, ...] = ("ar", "diffusion")


@dataclass
class _Side:
    """The sibling Remotes that make up one trained track."""

    bundle: Any
    pipeline: Any
    backend: Any
    algorithm: Any
    stack: Any


class PETrainer(BaseTrainer):
    """PE joint trainer: two TrainStack siblings + composed trainside rollout."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        diffusion_cfg: DictConfig,
        ar_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = False,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        # Offload both tracks' FSDP train state to CPU during generate so the
        # awake sglang engines have room; onload before the train backward.
        # Never runs for trainside (it samples the live FSDP modules) — see train_step.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._rollout_is_trainside = False

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        # ComposedSamplingParams(ar=N, diffusion=M) — drives PEPipeline's fan-out.
        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)

        # Per-track weight-sync bridges; None trainside (shares the modules).
        self.diffusion_sync = None
        self.ar_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.diffusion = self._wire_side(diffusion_cfg)
            self.ar = self._wire_side(ar_cfg)

            # Pass the (composed) pipeline only to engines whose role_cls
            # declares it (trainside). For a separate-process engine
            # (``composed_pe``: sglang_llm + sglang) there is no shared
            # pipeline — trained weights reach the engine via the sync bridges.
            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            takes_pipeline = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
            # Trainside samples the live FSDP modules → must not FSDP-offload.
            self._rollout_is_trainside = bool(takes_pipeline)
            if takes_pipeline:
                # Trainside: the composed PE pipeline shares both trained child
                # pipelines in-process, so the rollout samples the live FSDP
                # modules — no weight sync. ``stage_attrs: [diffusion, ar]``
                # eval-scopes both trained models.
                self.pe_pipeline = remote(
                    PEPipeline,
                    diffusion_pipeline=self.diffusion.pipeline,
                    llm_pipeline=self.ar.pipeline,
                )
                self.rollout = remote(**rollout_parsed, pipeline=self.pe_pipeline)
            else:
                self.pe_pipeline = None
                self.rollout = remote(**rollout_parsed)

            self.reward = remote_hydra(reward_cfg)

            # Non-trainside: one bridge per track, each routed to its child of
            # the composed engine by ``track_prefix`` (set in the sync block).
            if sync_cfg is not None:
                self.diffusion_sync = remote_hydra(
                    sync_cfg.diffusion, backend=self.diffusion.backend, rollout=self.rollout
                )
                self.ar_sync = remote_hydra(sync_cfg.ar, backend=self.ar.backend, rollout=self.rollout)

    def _wire_side(self, cfg: DictConfig) -> _Side:
        """Build one track's bundle → pipeline → backend → algorithm → stack.

        Identical to ``DiffusionTrainer``'s single-side chain; called twice
        (diffusion + ar) inside the shared placement block.
        """
        bundle = remote_hydra(cfg.bundle)
        pipeline = remote_hydra(cfg.pipeline, bundle=bundle)
        backend = remote_hydra(cfg.backend, bundle=bundle)
        algorithm = remote_hydra(cfg.algorithm, pipeline=pipeline)
        stack = remote_hydra(cfg.stack, fsdp_backend=backend, algorithm=algorithm)
        return _Side(bundle=bundle, pipeline=pipeline, backend=backend, algorithm=algorithm, stack=stack)

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data-source batch of ``P`` prompts into a typed ``RolloutReq``.

        No pre-expansion: ``PEPipeline`` fans out ``P → P*N → P*N*M`` internally
        from ``ComposedSamplingParams`` (``ar.samples_per_prompt`` rewrites,
        ``diffusion.samples_per_prompt`` images each). The single-track trainer
        pre-expands here; PE must not, or it would double-count.

        ``rollout_id`` keys the diffusion SDE-step schedule: the indices are
        resolved off the diffusion sub-block (``resolve_sde_indices``), stamped
        onto a per-request copy, and the ``scheduler`` is nulled so only the
        concrete ``sde_indices`` ride to the engine (mirrors
        :meth:`DiffusionTrainer._build_req` / :meth:`UnifiedModelTrainer._build_req`).
        The AR sub-block has no SDE machinery and is left untouched.
        """
        diff_params = get_diffusion_params(self.sampling_params)
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)
        sampling_params = dataclasses.replace(self.sampling_params, diffusion=diffusion)
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            sampling_params=sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
        )

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)``. ``mean_reward`` is the
        mean unnormalized image reward (for the log line).

        ``sync_weights`` pushes each track's freshly-trained adapter into the
        engine between ``wake_up`` and ``generate`` — no-op trainside (the
        rollout shares the live FSDP modules, so the bridges are ``None``).
        ``rollout_id`` only keys the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.diffusion_sync is not None:
            self.diffusion_sync.sync()
            self.ar_sync.sync()
        # Free both tracks' train state during the separate-engine generate.
        # Sync above reads the FSDP weights, so offload only after it.
        do_fsdp_offload = self._enable_fsdp_offload and not self._rollout_is_trainside
        if do_fsdp_offload:
            self.diffusion.backend.offload()
            self.ar.backend.offload()
        resp = self.rollout.generate(req)
        self.rollout.sleep()
        if do_fsdp_offload:
            self.diffusion.backend.onload()
            self.ar.backend.onload()

        # 1. Score the IMAGE track only — the AR track's TextSegment is not
        #    directly scorable; its reward is credit-assigned below.
        #    ``score_and_attach`` is DP_SCATTER: it shards the diffusion track
        #    (P*N*M) across workers, but the P-prompt ``req`` would broadcast
        #    whole, so each worker would see a (track-shard) vs P size
        #    mismatch. Expand the req prompt-major to the track size first
        #    (mirrors DiffusionTrainer's pre-expanded single-track req) so the
        #    req and track shard identically across DP workers.
        diff_track = resp.tracks["diffusion"]
        n_track, p = len(diff_track.sample_ids), max(1, req.batch_size)
        reward_req = req.repeat_interleave(n_track // p) if n_track > p and n_track % p == 0 else req
        scored = self.reward.score_and_attach(req=reward_req, track=diff_track)
        # propagate_rewards reshapes child.rewards directly (no hydration), so
        # turn the worker-returned TensorRef into a real tensor first.
        if scored.rewards is not None:
            scored.rewards = hydrate(scored.rewards)
        resp.tracks["diffusion"] = scored

        # 2. Credit-assign image reward up the lineage → fills the "ar" track
        #    (mean over the M images of each rewrite).
        resp = resp.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = resp.tracks["diffusion"].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        # 4. Per-track GRPO advantages — "ar" groups by prompt (N rewrites),
        #    "diffusion" groups by rewrite (M images).
        for name in TRACK_NAMES:
            resp.tracks[name] = resp.tracks[name].compute_advantages(normalize=True)

        # ``reward_req`` text is repeat_interleaved to the diffusion track size
        # (one prompt per sample), so it captions the image previews correctly —
        # unlike ``req`` (one prompt per group).
        self._drop_decoded(
            req,
            resp,
            rollout_id=rollout_id,
            media_prompts={"diffusion": list(reward_req.primitives["text"].texts)},
        )
        # 5. Route each track to its own stack (each DP_SCATTER-sharded on dispatch).
        results: Dict[str, TrainStepResult] = {
            name: getattr(self, name).stack.train_track(resp.tracks[name], training_progress=float(training_progress))
            for name in TRACK_NAMES
        }
        self.wandb_logger.log_rollout_step(rollout_id, results, resp, step_time_s=time.perf_counter() - t0)
        return results, mean_reward

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``.

        ``weight_sync_interval``: push each track's adapter into the engine
        every N rollouts (fused into ``train_step``'s generate; no-op trainside).
        """
        interval = max(1, weight_sync_interval)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            for rollout_id in range(num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet).
                sync_weights = rollout_id > 0 and rollout_id % interval == 0
                results, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
        finally:
            self._finish_wandb()
