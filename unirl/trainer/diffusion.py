import dataclasses
import inspect
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import get_class, get_object, instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.eval_suites import EvalRewardSuite, build_eval_suites
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
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
        reward_fraction: float = 0.0,
        enable_fsdp_offload: bool = False,
        adv_use_global_std: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = 64,
        eval_samples_per_prompt: int = 4,
        eval_chunk_prompts: int = 16,
        eval_cfg_text_scale: float = 4.0,
        eval_eta: float = 0.0,
        eval_rewards_cfg: Optional[Any] = None,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._layout = str(layout)
        self._train_fraction = float(train_fraction)
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._adv_use_global_std = bool(adv_use_global_std)
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_chunk_prompts = int(eval_chunk_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.eval_eta = float(eval_eta)
        self._eval_rewards_cfg = eval_rewards_cfg
        self._eval_suites: List[EvalRewardSuite] = []
        self._task_config: Dict[str, Any] = dict(task_config) if task_config else {}
        self._rollout_is_trainside = False
        self._uses_ema = False

        self.data_source = instantiate(data_source_cfg)
        self._data_source_cfg = data_source_cfg

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        self._noise_latent_shape: Optional[list] = (
            None
            if os.environ.get("DISABLE_DRIVER_XT")
            else self._resolve_noise_latent_shape(pipeline_cfg=pipeline_cfg, model_cfg=bundle_cfg)
        )

        self.weight_sync = None

        reward_fraction = float(reward_fraction)
        if not 0.0 <= reward_fraction < 1.0:
            raise ValueError(f"reward_fraction must be in [0, 1), got {reward_fraction}")
        if self._layout == "separate" and train_fraction + reward_fraction >= 1.0:
            raise ValueError(
                f"layout='separate' leaves no rollout GPUs: train_fraction ({train_fraction}) "
                f"+ reward_fraction ({reward_fraction}) must be < 1.0"
            )
        reward_separate = reward_fraction > 0.0

        train_cfgs = dict(
            bundle_cfg=bundle_cfg,
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            reward_cfg=(None if reward_separate else reward_cfg),
            algorithm_cfg=algorithm_cfg,
            stack_cfg=stack_cfg,
        )
        if self._layout == "separate":
            with placement(self.pool, fraction=train_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                if sync_cfg is not None:
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
            with placement(self.pool, fraction=1.0 - train_fraction - reward_fraction, shared_workers=True):
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=False)
            if self.weight_sync is not None:
                self._connect_separate(sync_cfg)
        else:
            with placement(self.pool, fraction=1.0 - reward_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=True)
                if sync_cfg is not None:
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

        if reward_separate:
            with placement(self.pool, fraction=reward_fraction, shared_workers=True):
                self.reward = remote_hydra(reward_cfg)
                self._wire_eval_suites()

        # Require rollout counts to divide both policy and reward DP sizes.
        n_samples = batch_size * total_samples_per_prompt(self.sampling_params)
        if n_samples % self.rollout.dp_size or n_samples % self.reward.dp_size:
            raise ValueError(
                f"batch_size({batch_size}) * samples_per_prompt = {n_samples} samples/rollout must be "
                f"divisible by BOTH rollout dp_size={self.rollout.dp_size} and reward dp_size="
                f"{self.reward.dp_size}. reward_fraction={reward_fraction} placed reward on its own slab, "
                f"leaving the policy/rollout on {self.rollout.dp_size} GPU(s) — pick batch_size * "
                f"samples_per_prompt divisible by both."
            )

    def _wire_eval_suites(self) -> None:
        """Build the ``eval_rewards`` suites in the CALLER's placement scope.

        Called exactly where the training reward was just created (train-side
        sibling in ``_build_train_side``, or the separate ``reward_fraction``
        slab in ``__init__``), so every suite reward shares the training
        reward's placement. See :mod:`unirl.trainer.eval_suites`.
        """
        self._eval_suites = build_eval_suites(
            self._eval_rewards_cfg, data_source_cfg=self._data_source_cfg, enabled=self.eval_interval > 0
        )

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

        ``reward_cfg`` is ``None`` when reward already owns a separate slab (see
        ``reward_fraction`` in ``__init__``); reward is then built there and skipped
        here so it is not also colocated on the train slab.
        """
        self.bundle = remote_hydra(bundle_cfg)
        self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
        self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
        if reward_cfg is not None:
            self.reward = remote_hydra(reward_cfg)
            self._wire_eval_suites()
        algo_cls = get_class(str(algorithm_cfg.get("_target_", "")))
        self._uses_ema = getattr(algo_cls, "requires_ema_rollout", False)
        needs_backend = self._uses_ema or getattr(algo_cls, "requires_backend", False)
        algo_extra = {"backend": self.backend} if needs_backend else {}
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
            # Shard trainside rollout over model DP to preserve SP prompt alignment.
            return remote(**rollout_parsed, pipeline=self.pipeline, sp_size=self.backend.sp_size)
        return remote(**rollout_parsed)

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
        resolved = get_object(target)
        pipeline_cls = resolved if isinstance(resolved, type) else getattr(resolved, "__self__", None)
        latent_shape_fn = getattr(pipeline_cls, "latent_shape", None)
        if latent_shape_fn is None:
            return None
        try:
            shape = latent_shape_fn(model_config=model_cfg, sampling_spec=self.sampling_params.get("diffusion"))
        except NotImplementedError:
            return None
        return [int(x) for x in shape]

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """Turn a data source batch into a request :class:`Sample`.

        The data source's input-only Part tree is preserved while every id is
        rollout-keyed (``r{rollout_id}:…``), then ``Part.fork`` fans out the
        diffusion gen shell to the ``N``-sample GRPO group. Image/video inputs
        are already chained by the data source.

        ``rollout_id`` keys the SDE step scheduler (``resolve_sde_indices``): the
        resolved indices are stamped onto a per-request copy of the diffusion
        sampling params (which rides on the gen Part), the schedule config itself
        is nulled so only the resolved ``sde_indices`` ride to the engine, and the
        pipeline's own latent geometry (``self._noise_latent_shape``) is pinned for
        the engine-side x_T recipe.

        ``sampling`` overrides the modality-keyed sampling dict (``evaluate`` passes
        its own deterministic params); ``None`` uses ``self.sampling_params``.
        """
        sp = sampling if sampling is not None else self.sampling_params
        diffusion = sp.get("diffusion")
        sde_indices = diffusion.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(
            diffusion, sde_indices=sde_indices, scheduler=None, init_noise_latent_shape=self._noise_latent_shape
        )
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text", "image", "video"},
            caller="DiffusionTrainer._build_request_sample",
            root_control=dict(self._task_config),
        )
        samples_per_prompt = total_samples_per_prompt(sp)
        request = request.fork(samples_per_prompt, sampling_params=diffusion)

        if sampling is not None and self._noise_latent_shape is not None:
            from unirl.sde.noise import make_prompt_seed_group_id

            texts = next((value for value in request.conditioning() if isinstance(value, Texts)), None)
            if not isinstance(texts, Texts) or len(texts.texts) != len(request.parts[-1].sample_ids):
                raise ValueError(
                    "DiffusionTrainer eval cannot key x_T on prompt content: "
                    f"prompt count {len(texts.texts) if isinstance(texts, Texts) else 'None'} != "
                    f"sample count {len(request.parts[-1].sample_ids)}."
                )
            noise_group_ids = [
                make_prompt_seed_group_id(text, sample_ordinal=index % samples_per_prompt)
                for index, text in enumerate(texts.texts)
            ]
            frontier = dataclasses.replace(request.parts[-1], init_noise_group_ids=noise_group_ids)
            request = request.with_parts([*request.parts[:-1], frontier])
        return request

    def train_step(
        self,
        sample: Sample,
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
        per-sample reward of the frontier gen Part (0.0 if none), for the log line.
        """
        t0 = time.perf_counter()
        self.rollout.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.sync()
        _do_fsdp_offload = (
            self._enable_fsdp_offload
            and self._layout != "separate"
            and not self._rollout_is_trainside
            and not self._uses_ema
        )
        if _do_fsdp_offload:
            self.backend.offload()
        # Swap EMA weights only for trainside rollout; remote engines receive them through weight sync.
        _inproc_ema_swap = self._uses_ema and self._rollout_is_trainside
        if _inproc_ema_swap:
            self.backend.apply_eval_ema()
        sample = self.rollout.generate(sample)
        if _inproc_ema_swap:
            self.backend.restore_from_eval()
        self.rollout.sleep()
        if _do_fsdp_offload:
            self.backend.onload()

        sample = self.reward.score_and_attach(sample)

        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            part.rewards = hydrate(part.rewards)
            if isinstance(part.component_rewards, dict):
                part.component_rewards = {name: hydrate(value) for name, value in part.component_rewards.items()}
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
            part = part.compute_advantages(normalize=True, use_global_std=self._adv_use_global_std)
            sample = sample.with_parts([*sample.parts[:-1], part])

        self._drop_decoded(sample, rollout_id=rollout_id)
        result = self.stack.train_track(sample.parts[-1], training_progress=float(training_progress))
        self.wandb_logger.log_rollout_step(rollout_id, result, sample, step_time_s=time.perf_counter() - t0)
        return result, mean_reward

    def evaluate(
        self,
        step: int,
        *,
        sync_weights: bool = True,
        sleep_after: bool = True,
    ) -> float:
        """Periodic eval on the eval set (no training); returns the mean reward.

        Mirrors :meth:`train_step`'s rollout+reward path but skips advantage/backward.
        Generates at the deterministic best-quality setting (``cfg_text_scale=
        eval_cfg_text_scale``, ``eta=eval_eta``; ``eval_samples_per_prompt`` x_T per
        prompt) and scores. The training reward plus every shared-set
        ``eval_rewards`` suite scores the SAME generated images over the default
        eval set (``run.eval_data_path``, ``eval_num_prompts`` prompts); each
        own-set suite then gets its own generation pass over its own prompts.
        All means land in one ``eval/*`` row (``eval/reward`` + ``eval/<suite>``);
        returns ``eval/reward``.

        ``sync_weights=False`` evaluates the policy already resident in the rollout
        engine without changing its weight version, and ``sleep_after=False`` leaves
        a dedicated engine resident afterwards — what the async trainer needs so
        evaluation does not perturb its pipeline. The defaults preserve the
        synchronous trainer's existing behavior.
        """
        base_diffusion = self.sampling_params.get("diffusion")
        replace_kwargs = dict(
            samples_per_prompt=self.eval_samples_per_prompt,
            eta=self.eval_eta,
        )
        if "cfg_text_scale" in {f.name for f in dataclasses.fields(base_diffusion)}:
            replace_kwargs["cfg_text_scale"] = self.eval_cfg_text_scale
        else:
            replace_kwargs["guidance_scale"] = self.eval_cfg_text_scale
        eval_diffusion = dataclasses.replace(base_diffusion, **replace_kwargs)
        eval_sp = {**self.sampling_params, "diffusion": eval_diffusion}
        self.rollout.wake_up()
        try:
            if sync_weights and self.weight_sync is not None:
                self.weight_sync.sync()
            scorers = [("reward", self.reward)] + [
                (s.name, s.reward) for s in self._eval_suites if s.data_source is None
            ]
            metrics = self._eval_pass(self.data_source, self.eval_num_prompts, scorers, eval_sp, step)
            for suite in self._eval_suites:
                if suite.data_source is not None:
                    n = suite.num_prompts or self.eval_num_prompts
                    metrics.update(self._eval_pass(suite.data_source, n, [(suite.name, suite.reward)], eval_sp, step))
        finally:
            if sleep_after:
                self.rollout.sleep()
        logger.info(
            "EVAL step %d  (%d samples/prompt, cfg=%.1f eta=%.1f)  %s",
            step,
            self.eval_samples_per_prompt,
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
        """One generate→score sweep over one eval set; returns each scorer's mean.

        The eval prompts are CHUNKED (``eval_chunk_prompts``) so one generate
        never holds N x the KV/decoded on the driver (the it2i memory
        bottleneck). Scores the single scorable (segment-carrying) track with
        every scorer — single-track for now; revisit if multi-track lands.
        """
        all_inputs = data_source.get_eval_samples(num_prompts)
        n_prompts = all_inputs.batch_size
        chunk = max(1, self.eval_chunk_prompts)
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        for start in range(0, n_prompts, chunk):
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
                result, mean_reward = self.train_step(
                    sample,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id + 1)
                self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
