"""UniRL v2 PE (Prompt Enhancement) joint trainer."""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.models.pe.pipeline import PEPipeline
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.eval_suites import build_eval_suites
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams, BaseSamplingParams, DiffusionSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

TRACK_NAMES: Tuple[str, ...] = ("ar", "diffusion")


@dataclass
class _Side:
    """The sibling Remotes that make up one track."""

    bundle: Any
    pipeline: Any
    backend: Any = None
    algorithm: Any = None
    stack: Any = None


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
        pe_cfg: Optional[DictConfig] = None,
        freeze_llm: bool = False,
        diffusion_group_scope: str = "rewrite",
        eval_interval: int = 0,
        eval_num_prompts: int = 8,
        eval_cfg_text_scale: float = 4.0,
        eval_eta: float = 0.0,
        eval_rewards_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._rollout_is_trainside = False
        self._freeze_llm = bool(freeze_llm)
        self._train_tracks: Tuple[str, ...] = ("diffusion",) if self._freeze_llm else TRACK_NAMES
        self._diffusion_group_scope = str(diffusion_group_scope)
        if self._diffusion_group_scope not in ("rewrite", "prompt"):
            raise ValueError(
                f"PETrainer.diffusion_group_scope must be 'rewrite' or 'prompt'; got {diffusion_group_scope!r}."
            )

        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.eval_eta = float(eval_eta)

        pe = pe_cfg if pe_cfg is not None else {}
        self._pe_instruction = pe.get("pe_instruction", None)
        self._pe_marker = pe.get("pe_marker", None)
        self._pe_max_chars = pe.get("pe_max_chars", None)

        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        self.diffusion_sync = None
        self.ar_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.diffusion = self._wire_side(diffusion_cfg)
            self.ar = self._wire_rollout_only_side(ar_cfg) if self._freeze_llm else self._wire_side(ar_cfg)

            rollout_parsed = parse_hydra_cfg(rollout_cfg)
            takes_pipeline = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
            # Trainside samples the live FSDP modules → must not FSDP-offload.
            self._rollout_is_trainside = bool(takes_pipeline)
            if takes_pipeline:
                self.pe_pipeline = remote(
                    PEPipeline,
                    diffusion_pipeline=self.diffusion.pipeline,
                    llm_pipeline=self.ar.pipeline,
                    pe_instruction=self._pe_instruction,
                    pe_marker=self._pe_marker,
                    pe_max_chars=self._pe_max_chars,
                )
                self.rollout = remote(**rollout_parsed, pipeline=self.pe_pipeline)
            else:
                self.pe_pipeline = None
                self.rollout = remote(**rollout_parsed)

            self.reward = remote_hydra(reward_cfg)
            self._eval_suites = build_eval_suites(
                eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
            )

            if sync_cfg is not None:
                self.diffusion_sync = remote_hydra(
                    sync_cfg.diffusion, backend=self.diffusion.backend, rollout=self.rollout
                )
                if not self._freeze_llm:
                    self.ar_sync = remote_hydra(sync_cfg.ar, backend=self.ar.backend, rollout=self.rollout)

    def _wire_side(self, cfg: DictConfig) -> _Side:
        """Build one track's bundle → pipeline → backend → algorithm → stack."""
        bundle = remote_hydra(cfg.bundle)
        pipeline = remote_hydra(cfg.pipeline, bundle=bundle)
        backend = remote_hydra(cfg.backend, bundle=bundle)
        algorithm = remote_hydra(cfg.algorithm, pipeline=pipeline)
        stack = remote_hydra(cfg.stack, fsdp_backend=backend, algorithm=algorithm)
        return _Side(bundle=bundle, pipeline=pipeline, backend=backend, algorithm=algorithm, stack=stack)

    def _wire_rollout_only_side(self, cfg: DictConfig) -> _Side:
        """Build a frozen, rollout-only side: bundle + pipeline, NO training trio."""
        bundle = remote_hydra(cfg.bundle)
        pipeline = remote_hydra(cfg.pipeline, bundle=bundle)
        return _Side(bundle=bundle, pipeline=pipeline)

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """Turn a data-source batch of ``P`` prompts into the composed request ``Sample``."""
        base = sampling if sampling is not None else self.sampling_params
        diff_params = base.get("diffusion")
        ar_params = base.get("ar")
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text"},
            caller="PETrainer._build_request_sample",
            root_control={"ar": {}, "chat": {}},
            require_single_input_part=True,
        )
        return request.fork(ar_params.samples_per_prompt, sampling_params=ar_params).fork(
            diffusion.samples_per_prompt, sampling_params=diffusion
        )

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass."""
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.diffusion_sync is not None:
            self.diffusion_sync.sync()
            if self.ar_sync is not None:
                self.ar_sync.sync()
        do_fsdp_offload = self._enable_fsdp_offload and not self._rollout_is_trainside
        if do_fsdp_offload:
            self.diffusion.backend.offload()
            if self.ar.backend is not None:
                self.ar.backend.offload()
        sample = self.rollout.generate(sample)
        self.rollout.sleep()
        if do_fsdp_offload:
            self.diffusion.backend.onload()
            if self.ar.backend is not None:
                self.ar.backend.onload()

        ar_idx = sample.gen_part_index(ARSamplingParams)
        diff_idx = sample.gen_part_index(DiffusionSamplingParams)
        parts_by_name = {"ar": ar_idx, "diffusion": diff_idx}

        sample = self.reward.score_and_attach(sample)
        diff_part = sample.parts[diff_idx]
        if diff_part.rewards is not None:
            diff_part.rewards = hydrate(diff_part.rewards)
        if isinstance(diff_part.component_rewards, dict):
            diff_part.component_rewards = {name: hydrate(value) for name, value in diff_part.component_rewards.items()}

        sample = sample.propagate_rewards(op="mean")

        mean_reward = 0.0
        di_rewards = sample.parts[diff_idx].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        new_parts = list(sample.parts)
        for name in self._train_tracks:
            idx = parts_by_name[name]
            layer = 0 if (name == "diffusion" and self._diffusion_group_scope == "prompt") else None
            new_parts[idx] = new_parts[idx].compute_advantages(normalize=True, group_layer=layer)
        sample = sample.with_parts(new_parts)

        self._drop_decoded(sample, rollout_id=rollout_id)
        results: Dict[str, TrainStepResult] = {
            name: getattr(self, name).stack.train_track(
                sample.parts[parts_by_name[name]], training_progress=float(training_progress)
            )
            for name in self._train_tracks
        }
        self.wandb_logger.log_rollout_step(rollout_id, results, sample, step_time_s=time.perf_counter() - t0)
        return results, mean_reward

    def evaluate(self, step: int) -> float:
        """Periodic eval on the eval set (no training); returns the mean image reward."""
        base_diffusion = self.sampling_params.get("diffusion")
        replace_kwargs = dict(eta=self.eval_eta)
        if "cfg_text_scale" in {f.name for f in dataclasses.fields(base_diffusion)}:
            replace_kwargs["cfg_text_scale"] = self.eval_cfg_text_scale
        else:
            replace_kwargs["guidance_scale"] = self.eval_cfg_text_scale
        eval_diffusion = dataclasses.replace(base_diffusion, **replace_kwargs)
        eval_sp = {**self.sampling_params, "diffusion": eval_diffusion}
        self.rollout.wake_up()
        try:
            if self.diffusion_sync is not None:
                self.diffusion_sync.sync()
                if self.ar_sync is not None:
                    self.ar_sync.sync()
            scorers = [("reward", self.reward)] + [
                (s.name, s.reward) for s in self._eval_suites if s.data_source is None
            ]
            metrics = self._eval_pass(self.data_source, self.eval_num_prompts, scorers, eval_sp, step)
            for suite in self._eval_suites:
                if suite.data_source is not None:
                    n = suite.num_prompts or self.eval_num_prompts
                    metrics.update(self._eval_pass(suite.data_source, n, [(suite.name, suite.reward)], eval_sp, step))
        finally:
            self.rollout.sleep()
        logger.info(
            "EVAL step %d  (cfg=%.1f eta=%.1f)  %s",
            step,
            self.eval_cfg_text_scale,
            self.eval_eta,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )
        self.wandb_logger.log_eval(step, metrics)
        return metrics["reward"]

    def _eval_pass(
        self,
        data_source: Any,
        num_prompts: int,
        scorers: List[Tuple[str, Any]],
        eval_sp: Dict[str, BaseSamplingParams],
        step: int,
    ) -> Dict[str, float]:
        """One generate→score sweep over one eval set; returns each scorer's mean."""
        all_inputs = data_source.get_eval_samples(num_prompts)
        n_prompts = all_inputs.batch_size
        chunk = max(1, self.batch_size)
        usable = n_prompts - n_prompts % chunk or n_prompts
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        for start in range(0, usable, chunk):
            sub = all_inputs.slice(start, min(start + chunk, n_prompts))
            request = self._build_request_sample(sub, step, sampling=eval_sp)
            generated = self.rollout.generate(request)
            for name, reward in scorers:
                scored = reward.score_and_attach(generated)
                rewards = scored.parts[-1].rewards
                if rewards is not None:
                    r = hydrate(rewards).to(torch.float32)
                    sums[name] += float(r.sum().item())
                    counts[name] += int(r.numel())
        return {name: sums[name] / max(1, counts[name]) for name, _ in scorers}

    def _ckpt_sides(self):
        """The trained sides to checkpoint: diffusion always; ar only when it trains."""
        sides = [("diffusion", self.diffusion)]
        if not self._freeze_llm and self.ar.backend is not None:
            sides.append(("ar", self.ar))
        return sides

    def _wait_for_checkpoints(self, *, timeout: Optional[float] = None) -> None:
        """Flush both side backends before another save or worker teardown."""
        for _, side in self._ckpt_sides():
            if timeout is None:
                side.backend.wait_for_checkpoint()
            else:
                side.backend.wait_for_checkpoint(_ray_get_timeout=timeout)

    def maybe_save_checkpoint(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str = "auto",
    ) -> None:
        """Save every ``save_interval`` rollouts (and on the last one), one subdir per trained side."""
        if save_interval <= 0:
            return
        step = rollout_id + 1
        if step % save_interval != 0 and step < num_rollouts:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step}")
        logger.info("Saving checkpoint at rollout %d/%d -> %s", step, num_rollouts, path)
        for name, side in self._ckpt_sides():
            side.backend.save(os.path.join(path, name), step=step, mode=save_mode)
        trainer_state_path = os.path.join(path, "trainer_state.json")
        trainer_state_tmp = f"{trainer_state_path}.tmp"
        with open(trainer_state_tmp, "w") as f:
            json.dump({"wandb_run_id": self.wandb_logger.run_id, "optimizer_step": self.wandb_logger.optimizer_step}, f)
        os.replace(trainer_state_tmp, trainer_state_path)
        if step >= num_rollouts:
            self._wait_for_checkpoints()

    def maybe_load_checkpoint(self, load_dir: Optional[str], *, num_rollouts: Optional[int] = None) -> int:
        """Restore both trained sides from ``load_dir``; return the resume step."""
        if not load_dir:
            return 0
        load_dir = os.path.abspath(load_dir)
        logger.info("Loading checkpoint from %s", load_dir)
        start = 0
        for name, side in self._ckpt_sides():
            result = side.backend.load(os.path.join(load_dir, name))
            if isinstance(result, list):
                result = result[0]
            start = int(result or 0)
        state_path = os.path.join(load_dir, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                self._resume_state = json.load(f)
        logger.info("Checkpoint restored; resuming at rollout %d", start)
        if num_rollouts is not None and start >= num_rollouts:
            logger.warning(
                "Checkpoint step %d >= num_rollouts %d — nothing left to train (num_rollouts is the TOTAL budget).",
                start,
                num_rollouts,
            )
        return start

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
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``."""
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            if self.eval_interval > 0:
                self.evaluate(start_rollout)
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
                sync_weights = (rollout_id > 0 and rollout_id % interval == 0) or (
                    resumed and rollout_id == start_rollout
                )
                results, mean_reward = self.train_step(
                    sample,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id + 1)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
