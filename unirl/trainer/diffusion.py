import dataclasses
import inspect
import logging
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from hydra.utils import get_class, get_object, instantiate
from omegaconf import DictConfig, OmegaConf

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.sde.index_schedule import AllSDEScheduler
from unirl.train.configs import FSDPConfig, resolve_fsdp_mesh_shape
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.eval_suites import EvalRewardSuite, build_eval_suites
from unirl.trainer.hydra import parse_hydra_cfg, remote_hydra
from unirl.trainer.residency import DEFAULT_RESIDENCY_POLICY, ResidencyPlanner, ResidencyPolicy, Role
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, DiffusionSamplingParams, total_samples_per_prompt
from unirl.utils.wandb_metrics import pooled_window_reward_metrics

logger = logging.getLogger(__name__)


def _run_cleanup_steps(steps: List[Tuple[str, Callable[[], None]]]) -> None:
    """Run every cleanup step, preserving an active phase failure when present."""
    preserve_active_error = sys.exc_info()[0] is not None
    first_error: Optional[Exception] = None
    for name, cleanup in steps:
        try:
            cleanup()
        except Exception as exc:
            logger.exception("Diffusion lifecycle cleanup failed during %s", name)
            if first_error is None:
                first_error = exc
    if first_error is not None and not preserve_active_error:
        raise first_error


def _validate_prompt_tree_dp_geometry(
    *,
    batch_size: int,
    rollout_dp_size: Optional[int],
    reward_dp_size: int,
    context: str,
) -> None:
    roles = [("reward", reward_dp_size)]
    if rollout_dp_size is not None:
        roles.insert(0, ("rollout", rollout_dp_size))
    for role, dp_size in roles:
        if batch_size % dp_size:
            raise ValueError(
                f"{context}: {role} dp_size={dp_size} must divide batch_size={batch_size} "
                "root prompt trees; DP_SCATTER preserves each prompt's whole subtree."
            )


def _validate_dp_geometry(
    *,
    batch_size: int,
    samples_per_prompt: int,
    num_updates_per_batch: int,
    rollout_dp_size: int,
    reward_dp_size: int,
    train_dp_size: int,
    require_rollout_dp_divisibility: bool = True,
) -> None:
    """Validate prompt-tree dispatch separately from generated-sample training."""
    values = {
        "batch_size": batch_size,
        "samples_per_prompt": samples_per_prompt,
        "num_updates_per_batch": num_updates_per_batch,
        "rollout_dp_size": rollout_dp_size,
        "reward_dp_size": reward_dp_size,
        "train_dp_size": train_dp_size,
    }
    invalid = {name: value for name, value in values.items() if value < 1}
    if invalid:
        raise ValueError(f"Diffusion DP geometry values must be positive; got {invalid}.")

    _validate_prompt_tree_dp_geometry(
        batch_size=batch_size,
        rollout_dp_size=rollout_dp_size if require_rollout_dp_divisibility else None,
        reward_dp_size=reward_dp_size,
        context="training",
    )

    total_generated = batch_size * samples_per_prompt
    if total_generated % train_dp_size:
        raise ValueError(
            f"batch_size({batch_size}) * samples_per_prompt({samples_per_prompt}) = "
            f"{total_generated} generated samples must be divisible by train dp_size={train_dp_size}."
        )
    per_train_rank = total_generated // train_dp_size
    if per_train_rank % num_updates_per_batch:
        raise ValueError(
            f"Per-train-rank generated batch {per_train_rank} must be divisible by "
            f"num_updates_per_batch={num_updates_per_batch}."
        )


def _preflight_trainside_geometry(
    *,
    num_devices: int,
    reward_fraction: float,
    batch_size: int,
    samples_per_prompt: int,
    num_updates_per_batch: int,
    prompt_local_rollout: bool,
    has_reward: bool,
    backend_cfg: DictConfig,
) -> None:
    """Reject statically-known trainside DP geometry before constructing heavy model roles."""
    shared_devices_f = (1.0 - reward_fraction) * num_devices
    shared_devices = int(round(shared_devices_f))
    if abs(shared_devices_f - shared_devices) > 1e-9:
        raise ValueError(
            f"Static trainside geometry: reward_fraction={reward_fraction} of num_devices={num_devices} "
            f"leaves {shared_devices_f} shared train/rollout devices, not an integer."
        )

    # The recipe carries a raw DictConfig, so unset keys fall back to the one
    # place the defaults live rather than to literals repeated here.
    fsdp_cfg = backend_cfg.get("fsdp_cfg", {})
    sp_size = fsdp_cfg.get("sp_size") or FSDPConfig.sp_size
    if shared_devices % sp_size:
        raise ValueError(
            f"Static trainside geometry: {shared_devices} shared devices are not divisible by sp_size={sp_size}."
        )
    # Called for its validation: raises when the shared world cannot form the
    # configured mesh, while the wrap-time call owns the real world size.
    resolve_fsdp_mesh_shape(
        fsdp_cfg.get("fsdp_mode", FSDPConfig.fsdp_mode),
        world_size=shared_devices,
        hsdp_shard_size=fsdp_cfg.get("hsdp_shard_size") or FSDPConfig.hsdp_shard_size,
    )
    shared_dp_size = shared_devices // sp_size

    # Mirror the runtime call below, which reads self.reward.dp_size. A reward
    # Handle carries no sp/tp/pp, so its dp_size is its slab width: the reward
    # slab when reward_fraction carves one, else the shared devices it colocates
    # on. Only a recipe with no reward: block at all scores with dp_size 1.
    if reward_fraction > 0.0:
        reward_devices_f = reward_fraction * num_devices
        reward_dp_size = int(round(reward_devices_f))
        if abs(reward_devices_f - reward_dp_size) > 1e-9:
            raise ValueError(
                f"Static trainside geometry: reward_fraction={reward_fraction} of num_devices={num_devices} "
                f"requests {reward_devices_f} reward devices, not an integer."
            )
    else:
        reward_dp_size = shared_devices if has_reward else 1

    _validate_dp_geometry(
        batch_size=batch_size,
        samples_per_prompt=samples_per_prompt,
        num_updates_per_batch=num_updates_per_batch,
        rollout_dp_size=shared_dp_size,
        reward_dp_size=reward_dp_size,
        train_dp_size=shared_dp_size,
        require_rollout_dp_divisibility=not prompt_local_rollout,
    )


# Per-field eval knobs the overlay replaced (or dropped), and what to write instead.
_RETIRED_EVAL_KEYS = {
    "eval_cfg_text_scale": "eval_sampling: {guidance_scale: X}",
    "eval_num_inference_steps": "eval_sampling: {num_inference_steps: X}",
    "eval_height": "eval_sampling: {height: X}",
    "eval_width": "eval_sampling: {width: X}",
    "eval_media_max_items": "logging: {log_media: true, media_max_items: X}",
    "eval_shift": "no equivalent: the per-request time-shift override was dropped (static-shift models keep their checkpoint shift)",
    "eval_mu": "not needed: dynamic-shift μ re-derives from the eval steps/resolution",
}

_RETIRED_RESIDENCY_KEYS = {
    "enable_fsdp_offload": (
        "train_resident: <inverted>   (true offloaded the trainer, so it becomes false; "
        "the one key now covers every idle phase, not just generation)"
    ),
    "offload_train_during_reward": (
        "train_resident: <inverted>   (true offloaded the trainer, so it becomes false; "
        "the same one key covers generation as well)"
    ),
    "rollout_sleep_after_generate": (
        "rollout_resident: <inverted>   (true was sleep-after-generate, so it becomes false)"
    ),
}

# Engine/driver-owned object fields the overlay cannot carry: overrides come from
# plain YAML (never hydra-instantiated), so a nested ``_target_`` would ride into
# the params as a bare dict and only blow up deep inside the first eval's request
# build (``'dict' object has no attribute 'get_sde_indices'``). Rejected by name.
_UNSUPPORTED_OVERLAY_FIELDS = frozenset(
    {"scheduler", "sde_strategy", "sigmas", "noise_group_ids", "init_noise_latent_shape"}
)


def reject_retired_residency_keys(cfg: Any) -> None:
    """Fail fast on the phase-scoped offload flags that per-role residency replaced."""
    present = sorted(key for key in _RETIRED_RESIDENCY_KEYS if cfg is not None and cfg.get(key) is not None)
    if not present:
        return
    moves = "\n".join(f"  {key}: X   ->   {_RETIRED_RESIDENCY_KEYS[key]}" for key in present)
    raise ValueError(
        "These per-phase offload flags are not supported. Residency is now one choice per role, "
        "held for as long as the role is idle, which is what lets the loop skip the offload/onload "
        f"round trip they forced between generation and scoring:\n{moves}"
    )


def reject_retired_eval_keys(cfg: Any) -> None:
    """Fail fast on the per-field ``eval_*`` knobs that ``eval_sampling:`` replaced."""
    present = sorted(key for key in _RETIRED_EVAL_KEYS if cfg is not None and cfg.get(key) is not None)
    if not present:
        return
    moves = "\n".join(f"  {key}: X   ->   {_RETIRED_EVAL_KEYS[key]}" for key in present)
    raise ValueError(
        "These per-field eval knobs are not supported — most moved into the `eval_sampling:` overlay, "
        f"which accepts ANY plain DiffusionSamplingParams field:\n{moves}"
    )


def build_eval_sampling(
    sampling_params: Dict[str, BaseSamplingParams],
    *,
    eta: float = 0.0,
    samples_per_prompt: Optional[int] = None,
    overrides: Any = None,
) -> Dict[str, BaseSamplingParams]:
    """Return ``sampling_params`` with its ``diffusion`` entry rebuilt for evaluation."""
    base = sampling_params.get("diffusion")
    if base is None:
        raise ValueError("build_eval_sampling: sampling params carry no `diffusion` entry to override.")
    field_names = {f.name for f in dataclasses.fields(base)}

    updates: Dict[str, Any] = {"eta": eta}
    if samples_per_prompt is not None:
        updates["samples_per_prompt"] = samples_per_prompt
    updates.update(_resolve_overrides(overrides, field_names))

    steps = updates.get("num_inference_steps", base.num_inference_steps)
    if updates["eta"] <= 0.0:
        updates["sde_indices"] = []
        updates["scheduler"] = None
    elif steps != base.num_inference_steps:
        raise ValueError(
            f"eval eta={updates['eta']} leaves the SDE gate on, but eval_sampling.num_inference_steps"
            f"={steps} differs from the rollout's {base.num_inference_steps}: the gated step indices "
            "are resolved against the rollout's step count and cannot address the eval schedule. "
            "Set eval_eta: 0, or drop the step override."
        )
    return {**sampling_params, "diffusion": dataclasses.replace(base, **updates)}


def _resolve_overrides(overrides: Any, field_names: Set[str]) -> Dict[str, Any]:
    """Validate a recipe ``eval_sampling:`` block into plain ``dataclasses.replace`` kwargs."""
    if overrides is None:
        return {}
    if OmegaConf.is_config(overrides):
        overrides = OmegaConf.to_container(overrides, resolve=True)
    if not isinstance(overrides, Mapping):
        raise TypeError(
            "eval_sampling must be a mapping of diffusion sampling fields, "
            f"got {type(overrides).__name__}. It overlays `sampling:`, so it takes no `_target_`."
        )
    unknown = sorted(set(overrides) - field_names)
    if unknown:
        raise ValueError(f"eval_sampling has unknown field(s) {unknown}; valid fields are {sorted(field_names)}.")
    unsupported = sorted(set(overrides) & _UNSUPPORTED_OVERLAY_FIELDS)
    if unsupported:
        raise ValueError(
            f"eval_sampling cannot override {unsupported}: engine/driver-owned object fields "
            "(plain YAML cannot carry instantiated objects), so eval keeps the rollout's own. "
            "The SDE gate is governed by eval_eta: <= 0 clears it, > 0 keeps the training gate."
        )
    return dict(overrides)


class DiffusionTrainer(BaseTrainer):
    """Reference trainer: train + rollout colocated on the whole pool."""

    _prompt_local_rollout = False

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        rollout_cfg: DictConfig,
        reward_cfg: Optional[DictConfig],
        algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        reward_stack_cfg: Optional[DictConfig] = None,
        logging_cfg: Optional[DictConfig] = None,
        layout: str = "colocate",
        train_fraction: float = 0.5,
        worker_max_concurrency: Optional[int | Sequence[int]] = None,
        reward_fraction: float = 0.0,
        train_resident: bool = DEFAULT_RESIDENCY_POLICY.train_resident,
        rollout_resident: bool = DEFAULT_RESIDENCY_POLICY.rollout_resident,
        reward_resident: bool = DEFAULT_RESIDENCY_POLICY.reward_resident,
        adv_use_global_std: bool = False,
        accumulate_rollouts: int = 1,
        eval_interval: int = 0,
        eval_num_prompts: int = 64,
        eval_samples_per_prompt: int = 4,
        eval_chunk_prompts: int = 16,
        eval_eta: float = 0.0,
        eval_sampling_cfg: Optional[Any] = None,
        eval_rewards_cfg: Optional[Any] = None,
        control: Optional[Dict[str, Any]] = None,
    ) -> None:
        reject_retired_eval_keys(cfg)
        reject_retired_residency_keys(cfg)
        if layout not in ("colocate", "separate"):
            raise ValueError(f"DiffusionTrainer: layout must be 'colocate' or 'separate', got {layout!r}.")
        rollout_parsed = parse_hydra_cfg(rollout_cfg)
        takes_pipeline = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
        if takes_pipeline:
            if layout == "separate":
                raise ValueError(
                    "layout='separate' requires a dedicated-rollout engine "
                    "(vllm/sglang); the trainside direct-sampling engine needs "
                    "the pipeline as a local sibling and cannot live on a "
                    "separate slab."
                )
            if sync_cfg is not None:
                raise ValueError(
                    "DiffusionTrainer: trainside direct-sampling engine reads the live training weights and cannot use a `sync:` block."
                )
        elif sync_cfg is None:
            raise ValueError("DiffusionTrainer: dedicated rollout engine requires a `sync:` block.")
        algo_cls = get_class(algorithm_cfg["_target_"])
        self._uses_ema = algo_cls.requires_ema_rollout
        self._flash_per_sample_sde = algo_cls.requires_per_sample_sde_index
        # requires_advantages=False algorithms keep rewards for monitoring only.
        self._algo_requires_advantages = algo_cls.requires_advantages
        self.eval_interval = eval_interval
        if reward_cfg is None:
            if self._algo_requires_advantages:
                raise ValueError(
                    "The recipe has no `reward:` block, but the algorithm requires advantages "
                    "(requires_advantages=True) — RL training cannot run without a reward model. "
                    "Only supervised/teacher-anchored algorithms may omit `reward:`."
                )
            if self.eval_interval > 0:
                raise ValueError(
                    f"eval_interval={self.eval_interval} needs a reward to score eval generations, "
                    "but the recipe has no `reward:` block. Set eval_interval: 0 or configure a "
                    "(monitoring-only) reward."
                )
        if not 0.0 <= reward_fraction < 1.0:
            raise ValueError(f"reward_fraction must be in [0, 1), got {reward_fraction}")
        if layout == "separate" and train_fraction + reward_fraction >= 1.0:
            raise ValueError(
                f"layout='separate' leaves no rollout GPUs: train_fraction ({train_fraction}) "
                f"+ reward_fraction ({reward_fraction}) must be < 1.0"
            )
        reward_separate = reward_fraction > 0.0
        if reward_separate and reward_cfg is None:
            raise ValueError(
                f"reward_fraction={reward_fraction} carves reward its own slab, but the recipe "
                "has no `reward:` block — drop reward_fraction or configure a reward."
            )
        if reward_stack_cfg is not None:
            if reward_cfg is None:
                raise ValueError("reward_stack needs a `reward:` block: it scores every micro through that sibling.")
            if layout != "colocate" or reward_fraction > 0.0:
                raise ValueError(
                    "reward_stack needs layout='colocate' with reward_fraction=0, so the engine and the reward share a "
                    f"Worker; got layout={layout!r}, reward_fraction={reward_fraction}."
                )
            if not reward_resident:
                raise ValueError(
                    "reward_stack needs reward_resident=true: it scores inside the rollout window, where a parked "
                    "reward is never woken."
                )
        super().__init__(
            cfg=cfg,
            logging_cfg=logging_cfg,
            worker_max_concurrency=worker_max_concurrency,
        )
        self.batch_size = batch_size
        self._train_fraction = train_fraction
        # Residency is per role and phase-independent: `true` keeps a role's
        # weights on the GPU while it is idle, `false` parks them on CPU. No
        # value here ever stops a role's process; only its weights move.
        residency_policy = ResidencyPolicy(
            train_resident=train_resident,
            rollout_resident=rollout_resident,
            reward_resident=reward_resident,
        )
        self._residency: Optional[ResidencyPlanner] = None
        self._adv_use_global_std = adv_use_global_std
        self.eval_num_prompts = eval_num_prompts
        self.eval_samples_per_prompt = eval_samples_per_prompt
        self.eval_chunk_prompts = eval_chunk_prompts
        self.eval_eta = eval_eta
        self._eval_suites: List[EvalRewardSuite] = []
        self._control: Dict[str, Any] = dict(control) if control else {}
        # Set from the built weight_sync's capabilities in _build_residency_planner.
        self._staged_weight_sync = False

        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        if self._flash_per_sample_sde:
            self._check_flash_rectification_pool(algorithm_cfg)
        # Frozen at init, so overlay contradictions surface at startup rather than an
        # hour in.
        self._eval_sampling_params: Dict[str, BaseSamplingParams] = build_eval_sampling(
            self.sampling_params,
            eta=self.eval_eta,
            samples_per_prompt=self.eval_samples_per_prompt,
            overrides=eval_sampling_cfg,
        )

        self._noise_latent_shape: Optional[list] = self._resolve_noise_latent_shape(
            pipeline_cfg=pipeline_cfg,
            model_cfg=bundle_cfg,
            sampling_spec=self.sampling_params["diffusion"],
        )
        # Eval may render at its own resolution, so the driver-authored x_T has to
        # match THAT geometry rather than the rollout's.
        self._eval_noise_latent_shape: Optional[list] = self._resolve_noise_latent_shape(
            pipeline_cfg=pipeline_cfg,
            model_cfg=bundle_cfg,
            sampling_spec=self._eval_sampling_params["diffusion"],
        )
        if self.eval_interval > 0 and bool((self.logging_cfg or {}).get("log_media", False)):
            eval_diffusion = self._eval_sampling_params["diffusion"]
            if self._eval_noise_latent_shape is None:
                logger.warning(
                    "logging.log_media is on, but eval has no driver-authored x_T "
                    "(disable_driver_xt, or latent_shape() opted out), so "
                    "every eval draws fresh noise and the eval panel will not be comparable "
                    "across evals."
                )
            elif eval_diffusion.eta > 0.0:
                logger.warning(
                    "logging.log_media is on with eval eta=%.2f > 0: x_T is prompt-keyed, but "
                    "per-step SDE noise is seeded from the eval step's sample ids, so eval "
                    "panels differ by noise as well as policy. Set eval_eta: 0 for a "
                    "like-for-like filmstrip.",
                    eval_diffusion.eta,
                )

        self.weight_sync = None
        # None when the recipe has no ``reward:`` block (validated above).
        self.reward = None
        self.reward_stack = None

        if takes_pipeline:
            _preflight_trainside_geometry(
                num_devices=self.num_devices,
                reward_fraction=reward_fraction,
                batch_size=batch_size,
                samples_per_prompt=total_samples_per_prompt(self.sampling_params),
                num_updates_per_batch=stack_cfg.get("num_updates_per_batch", 1),
                prompt_local_rollout=self._prompt_local_rollout,
                has_reward=reward_cfg is not None,
                backend_cfg=backend_cfg,
            )

        needs_backend = self._uses_ema or algo_cls.requires_backend
        train_cfgs = dict(
            bundle_cfg=bundle_cfg,
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            reward_cfg=(None if reward_separate else reward_cfg),
            algorithm_cfg=algorithm_cfg,
            stack_cfg=stack_cfg,
            needs_backend=needs_backend,
            eval_rewards_cfg=eval_rewards_cfg,
            data_source_cfg=data_source_cfg,
        )
        if layout == "separate":
            with placement(self.pool, fraction=train_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
            with placement(self.pool, fraction=1.0 - train_fraction - reward_fraction, shared_workers=True):
                self.rollout = remote(**rollout_parsed)
            self._connect_separate(sync_cfg)
        else:
            with placement(self.pool, fraction=1.0 - reward_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                if takes_pipeline:
                    # Shard trainside rollout over model DP to preserve SP prompt alignment.
                    self.rollout = remote(**rollout_parsed, pipeline=self.pipeline, sp_size=self.backend.sp_size)
                else:
                    self.rollout = remote(**rollout_parsed)
                if reward_stack_cfg is not None:
                    self.reward_stack = self._build_reward_stack(reward_stack_cfg)
                if sync_cfg is not None:
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

        if reward_separate:
            with placement(self.pool, fraction=reward_fraction, shared_workers=True):
                self.reward = remote_hydra(reward_cfg)
                self._eval_suites = build_eval_suites(
                    eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
                )

        # Cross-rollout gradient accumulation: one optimizer step per M rollouts
        # (the MOPD task-cycle contract).
        self.accumulate_rollouts = accumulate_rollouts
        self._validate_accumulation(stack_cfg)

        self._build_residency_planner(
            policy=residency_policy,
            layout=layout,
            reward_separate=reward_separate,
        )
        _validate_dp_geometry(
            batch_size=batch_size,
            samples_per_prompt=total_samples_per_prompt(self.sampling_params),
            num_updates_per_batch=stack_cfg.get("num_updates_per_batch", 1),
            rollout_dp_size=self.rollout.dp_size,
            reward_dp_size=self.reward.dp_size if self.reward is not None else 1,
            train_dp_size=self.stack.dp_size,
            require_rollout_dp_divisibility=not self._prompt_local_rollout,
        )
        if self._flash_per_sample_sde:
            strata = len(self._flash_candidate_pool()) * self.rollout.dp_size
            if batch_size % strata:
                raise ValueError(
                    f"FlashGRPO stratifies prompts across the SDE candidate pool, so batch_size must be a "
                    f"multiple of pool_size * rollout dp_size = {strata}; got {batch_size}."
                )

    def _validate_accumulation(self, stack_cfg: DictConfig) -> None:
        """``accumulate_rollouts > 1`` needs single-update, non-EMA, cycle-aligned cadences."""
        acc = self.accumulate_rollouts
        if acc < 1:
            raise ValueError(f"accumulate_rollouts must be >= 1, got {acc}")
        if acc == 1:
            return
        n_updates = stack_cfg.get("num_updates_per_batch", 1) or 1
        if n_updates != 1:
            raise ValueError(
                f"accumulate_rollouts={acc} requires stack.num_updates_per_batch == 1 "
                f"(got {n_updates}): extra optimizer steps inside one accumulation window "
                "would re-step on partial gradients."
            )
        if self._uses_ema:
            raise ValueError(
                f"accumulate_rollouts={acc} is not validated with requires_ema_rollout "
                "algorithms (EMA updates fire per rollout boundary, not per optimizer step)."
            )
        if self.pool.transport_kind in ("transfer_queue", "tq"):
            raise ValueError(
                f"accumulate_rollouts={acc} is not validated with the transfer_queue transport: "
                "buffers are reclaimed once per window (after train_step), so the pool would "
                "have to hold every rollout of the window."
            )
        if self.eval_interval > 0 and self.eval_interval % acc:
            raise ValueError(
                f"eval_interval={self.eval_interval} is not a multiple of accumulate_rollouts={acc}: "
                "eval runs between windows, so this cadence would never fire."
            )
        domains = getattr(self.data_source, "domains", None)
        if domains and acc % len(domains) != 0:
            raise ValueError(
                f"accumulate_rollouts={acc} must be a multiple of the data source's domain "
                f"count ({len(domains)}), or each optimizer step would cover an uneven subset "
                "of the task cycle (mirrors the official DiffusionOPD assert on "
                "num_batches_per_epoch % len(teachers))."
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
        needs_backend: bool,
        eval_rewards_cfg,
        data_source_cfg,
    ) -> None:
        """Build the train-side remotes in the *currently active* placement scope."""
        self.bundle = remote_hydra(bundle_cfg)
        self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
        self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
        if reward_cfg is not None:
            self.reward = remote_hydra(reward_cfg)
            self._eval_suites = build_eval_suites(
                eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
            )
        algo_extra = {"backend": self.backend} if needs_backend else {}
        self.algorithm = remote_hydra(algorithm_cfg, pipeline=self.pipeline, **algo_extra)
        self.stack = remote_hydra(stack_cfg, fsdp_backend=self.backend, algorithm=self.algorithm)

    def _build_residency_planner(
        self,
        *,
        policy: ResidencyPolicy,
        layout: str,
        reward_separate: bool,
    ) -> None:
        """Wire the residency planner to the roles that actually share this slab."""
        # Capability, not class name: a parked trainer needs the push split into a
        # read (weights resident) and a load (rollout awake), which only the LoRA
        # syncs expose. Probe the built object so a new implementation is picked up
        # by having the methods rather than by being named.
        staged_methods = ("extract", "push", "has_staged_adapter", "invalidate")
        self._staged_weight_sync = self.weight_sync is not None and all(
            hasattr(self.weight_sync, name) for name in staged_methods
        )
        # Train is parkable only where parking frees something the active role can
        # use: not on a separate slab, and not behind a trainside rollout, whose
        # generation reads the very weights that would be parked.
        rollout_is_trainside = self.weight_sync is None
        train_parkable = layout != "separate" and not rollout_is_trainside
        if not policy.train_resident:
            # Order matters: an unparkable trainer is the more basic reason, and
            # reporting the sync instead would send the reader after the wrong knob.
            if not train_parkable:
                reason = (
                    f"layout: {layout} puts the trainer on its own slab"
                    if layout == "separate"
                    else "a trainside rollout generates from the trainer's own weights"
                )
                raise ValueError(
                    f"train_resident=false has nothing to park here: {reason}, so parking would "
                    "move memory no other role can use. Drop the key instead of having the trainer "
                    "silently ignore the requested policy."
                )
            if not self._staged_weight_sync:
                raise ValueError(
                    "train_resident=false is not supported with this weight sync: "
                    f"{type(self.weight_sync).__name__} exposes sync() alone, which reads the "
                    "trainer's weights and loads them into the rollout in one call, so it cannot "
                    "run with the trainer parked. Use a LoRA sync, or set train_resident=true "
                    "instead of having the trainer silently ignore the requested policy."
                )
            if self._uses_ema:
                raise ValueError(
                    "train_resident=false is not supported with EMA/DiffusionNFT algorithms and an "
                    "external colocated rollout: parking the trainer has not been validated against "
                    "backend.ema state, which the contrastive branch reads. Set train_resident=true "
                    "instead of having the trainer silently ignore the requested policy."
                )
        self._residency = ResidencyPlanner(
            policy,
            train=(self.backend.onload, self.backend.offload) if train_parkable else None,
            rollout=(self.rollout.wake_up, self.rollout.sleep),
            reward=(
                (self.reward.onload, self.reward.offload) if self.reward is not None and not reward_separate else None
            ),
            run_steps=_run_cleanup_steps,
        )

    def _connect_separate(self, sync_cfg: DictConfig) -> None:
        """One-time cross-slab handshake: hand rank 0 the rollout Worker handles."""
        if sync_cfg["_target_"].endswith("NCCLWeightSync"):
            addr, port = self.weight_sync.pick_master()[0]
            self.weight_sync.set_rollout_targets(self.rollout.workers, self.rollout.role_name)
            self.weight_sync.connect(
                master_addr=addr,
                master_port=port,
                num_rollout_gpus=len(self.rollout.workers),
            )
        else:
            self.weight_sync.set_rollout_targets([(self.rollout.role_name, self.rollout.workers)])

    def _resolve_noise_latent_shape(
        self, *, pipeline_cfg: DictConfig, model_cfg: DictConfig, sampling_spec: Any
    ) -> Optional[list]:
        """Per-sample latent shape for the driver-authored x_T recipe, or ``None``."""
        if sampling_spec.disable_driver_xt:
            return None
        resolved = get_object(pipeline_cfg["_target_"])
        pipeline_cls = resolved if isinstance(resolved, type) else getattr(resolved, "__self__", None)
        latent_shape_fn = getattr(pipeline_cls, "latent_shape", None)
        if latent_shape_fn is None:
            return None
        try:
            shape = latent_shape_fn(model_config=model_cfg, sampling_spec=sampling_spec)
        except NotImplementedError:
            return None
        return list(shape)

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
        sde_index_override: Optional[int] = None,
    ) -> Sample:
        """Turn a data source batch into a request :class:`Sample`, optionally pinning one SDE step."""
        sp = sampling if sampling is not None else self.sampling_params
        noise_latent_shape = self._eval_noise_latent_shape if sampling is not None else self._noise_latent_shape
        diffusion = sp["diffusion"]
        if sde_index_override is None:
            sde_indices = diffusion.get_sde_indices(rollout_id)
        else:
            sde_indices = [sde_index_override]
        diffusion = dataclasses.replace(
            diffusion, sde_indices=sde_indices, scheduler=None, init_noise_latent_shape=noise_latent_shape
        )
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text", "image", "video"},
            caller="DiffusionTrainer._build_request_sample",
            control=self._control,
        )
        samples_per_prompt = total_samples_per_prompt(sp)
        request = request.fork(samples_per_prompt, sampling_params=diffusion)

        if sampling is not None and noise_latent_shape is not None:
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

    def _flash_candidate_pool(self) -> List[int]:
        """SDE steps FlashGRPO's per-prompt draw may pick, from the scheduler's candidate pool."""
        diffusion = self.sampling_params["diffusion"]
        scheduler = diffusion.scheduler
        if not isinstance(scheduler, AllSDEScheduler):
            raise ValueError(
                "FlashGRPO per-sample SDE-index path requires sampling_params['diffusion'] to carry "
                f"an AllSDEScheduler; got scheduler={type(scheduler).__name__ if scheduler is not None else None}."
            )
        pool = scheduler.sde_candidate_pool()
        if not pool:
            raise ValueError("FlashGRPO per-sample SDE-index path: the scheduler candidate pool is empty.")
        # The last denoising step stores only {T-1, T} latents while earlier steps
        # store three, so a group that drew it cannot be concatenated with the rest.
        num_inference_steps = diffusion.num_inference_steps
        if max(pool) >= num_inference_steps - 1:
            raise ValueError(
                f"FlashGRPO per-sample SDE candidate pool reaches step {max(pool)}, but the last denoising "
                f"step (num_inference_steps-1 = {num_inference_steps - 1}) stores a different number of "
                "trajectory latents, so mixed-shape rollout groups cannot be concatenated. Narrow the "
                "scheduler timestep_fraction so its end maps below the final step."
            )
        return pool

    def _check_flash_rectification_pool(self, algorithm_cfg: Any) -> None:
        """Fail fast when the rectification normalizer disagrees with the SDE candidate pool."""
        pool = sorted(self._flash_candidate_pool())
        if "rectification_indices" not in algorithm_cfg or algorithm_cfg["rectification_indices"] is None:
            raise ValueError(
                "FlashGRPO per-sample path requires algorithm.rectification_indices set to the SDE "
                f"candidate pool {pool}; got None."
            )
        rectification_sorted = sorted(int(i) for i in algorithm_cfg["rectification_indices"])
        if rectification_sorted != pool:
            raise ValueError(
                f"FlashGRPO rectification_indices {rectification_sorted} must equal the SDE candidate "
                f"pool {pool}: each sample draws its step from the pool but is normalized over "
                "rectification_indices, so a mismatch silently mis-scales the loss."
            )

    def _build_flash_request_samples(self, inputs: Sample, rollout_id: int) -> List[Tuple[int, Sample]]:
        """Stratify one SDE step per prompt; return ``(step, request Sample)`` per step group."""
        pool = self._flash_candidate_pool()
        # ``Sample.select`` indexes this same list, which groups by root id, so a
        # duplicate or empty root id set would collapse the batch onto one request.
        n_prompts = len(inputs.split())
        root_rows = inputs.parts[0].batch_size
        if n_prompts != root_rows:
            raise ValueError(
                "FlashGRPO per-sample SDE-index path: Sample.split() yielded "
                f"{n_prompts} root groups for {root_rows} root rows, so the per-prompt SDE "
                "draw cannot address the batch."
            )
        # One generate call per step group is DP-scattered, and the handle rejects a
        # call whose root-prompt count is not a multiple of the rollout's dp_size, so
        # each step must own a whole number of DP shards. Equal counts per step is also
        # what Flash-GRPO's stratified assignment asks for. See README.md here.
        dp_size = self.rollout.dp_size
        pool_size = len(pool)
        if n_prompts % (pool_size * dp_size):
            raise ValueError(
                f"FlashGRPO stratifies {n_prompts} prompts across the {pool_size}-step SDE candidate "
                f"pool, so batch_size must be a multiple of pool_size * rollout dp_size = "
                f"{pool_size * dp_size}; got {n_prompts}. Widen or narrow the scheduler's "
                "timestep_fraction, or resize batch_size."
            )
        rng = np.random.default_rng(rollout_id)
        # Prompt ``i`` joins stratum ``i % pool_size``, so each step owns exactly
        # n_prompts / pool_size prompts and the whole pool is covered every rollout.
        order = rng.permutation(pool_size)
        steps = [pool[int(s)] for s in order]

        positions_by_step: Dict[int, List[int]] = {}
        for position in range(n_prompts):
            positions_by_step.setdefault(steps[position % pool_size], []).append(position)

        return [
            (
                step,
                self._build_request_sample(
                    inputs.select(torch.tensor(positions_by_step[step], dtype=torch.long)),
                    rollout_id,
                    sde_index_override=step,
                ),
            )
            for step in sorted(positions_by_step)
        ]

    def _stamp_sde_index_per_sample(self, sample: Sample, sde_index: int) -> None:
        """Stamp ``segment.sde_index_per_sample`` and normalize shared step metadata for group concat."""
        frontier = sample.parts[-1]
        segment = frontier.segment
        if segment is None:
            raise ValueError("FlashGRPO: a step group's rollout returned no segment to stamp sde_index_per_sample on.")
        segment.sde_index_per_sample = torch.full((frontier.batch_size,), sde_index, dtype=torch.long)
        if segment.indices is not None and segment.latents is not None:
            expected = [sde_index, sde_index + 1, frontier.sampling_params.num_inference_steps]
            if segment.latents.shape[1] != 3 or hydrate(segment.indices).tolist() != expected:
                raise ValueError("FlashGRPO requires rollout latent slots [before, after, final].")
            segment.indices = torch.arange(3, dtype=torch.long)
        if segment.sde_indices is not None:
            segment.sde_indices = torch.zeros(1, dtype=torch.long)
        if isinstance(frontier.sampling_params, DiffusionSamplingParams):
            frontier.sampling_params = dataclasses.replace(frontier.sampling_params, sde_indices=None)

    def _prepare_for_save(self) -> None:
        """Checkpointing reads the trainer's weights, and evaluate() may have parked them."""
        self._residency.enter(Role.TRAIN)

    def _cache_adapter_for_push(self) -> bool:
        """Read the adapter while the trainer is resident; True if a later push can use it."""
        if self.weight_sync is None or not self._staged_weight_sync:
            return False
        if not self.weight_sync.has_staged_adapter()[0]:
            # enter() rather than set(): the read must not put the trainer on the
            # slab beside a reward that the rollout phase has not parked yet.
            self._residency.enter(Role.TRAIN)
            self.weight_sync.extract()
        return True

    def _push_or_sync(self, *, staged: bool) -> None:
        """Load the adapter into an awake rollout, from cache when one was taken."""
        if self.weight_sync is None:
            return
        if staged:
            self.weight_sync.push()
        else:
            self.weight_sync.sync()

    @contextmanager
    def _reward_phase(self, *, preserve_rollout: bool = False) -> Iterator[None]:
        """Give the reward the slab, leaving the trainer parked if it already is."""
        preserve = (Role.ROLLOUT,) if preserve_rollout else ()
        self._residency.enter(Role.REWARD, preserve=preserve)
        yield

    def _build_reward_stack(self, reward_stack_cfg: DictConfig) -> Any:
        """Build the stack beside the engine it wraps and reject the SP/TP layouts it inherits but cannot serve."""
        stack = remote_hydra(reward_stack_cfg, rollout=self.rollout, reward=self.reward)
        if stack.dp_size != self.rollout.dp_size:
            raise ValueError(f"reward_stack dp_size={stack.dp_size} must equal rollout dp_size={self.rollout.dp_size}.")
        if stack.sp_size != 1 or stack.tp_size != 1:
            raise ValueError(
                f"reward_stack needs sp_size=1 and tp_size=1, got sp_size={stack.sp_size}, tp_size={stack.tp_size}: "
                "every rank of an SP or TP group would score the same shard, or nothing."
            )
        return stack

    @contextmanager
    def _rollout_window(self, *, sync_weights: bool, sleep_rollout: bool) -> Iterator[None]:
        """One wake / push / EMA window spanning every generate call made inside it."""
        # No EMA term: _build_residency_planner already rejected the EMA x
        # parked-train x colocated-external combination at startup, fail-fast
        # instead of silently skipping the requested policy.
        will_park_train = not self._residency.policy.train_resident and self._residency.parkable(Role.TRAIN)
        # Swap EMA weights only for trainside rollout; remote engines receive
        # them through weight sync.
        should_swap_ema = self._uses_ema and self.weight_sync is None
        staged_sync = False
        ema_apply_attempted = False
        generation_succeeded = False
        try:
            if sync_weights and will_park_train:
                staged_sync = self._cache_adapter_for_push()
            # Parks the trainer (and a non-resident reward) before waking the
            # rollout, so the peak never holds two roles at once.
            self._residency.enter(Role.ROLLOUT)
            if sync_weights:
                self._push_or_sync(staged=staged_sync)
            if should_swap_ema:
                ema_apply_attempted = True
                self.backend.apply_eval_ema()
            yield
            generation_succeeded = True
        finally:
            cleanup_steps: List[Tuple[str, Callable[[], None]]] = []
            if ema_apply_attempted:
                cleanup_steps.append(("EMA restore", self.backend.restore_from_eval))
            _run_cleanup_steps(cleanup_steps)
            # The trainer is deliberately not reloaded here. It stays parked
            # through the reward phase and comes back once, before the optimizer
            # step, which is the only point that needs it.
            if sleep_rollout or not generation_succeeded:
                self._residency.set(Role.ROLLOUT, False)

    def _generate_with_residency(
        self,
        sample: Sample,
        *,
        sync_weights: bool,
        sleep_rollout: bool,
        score_inline: bool = False,
    ) -> Sample:
        """Generate with exception-safe EMA and residency cleanup."""
        return self._generate_many_with_residency(
            [sample],
            sync_weights=sync_weights,
            sleep_rollout=sleep_rollout,
            score_inline=score_inline,
        )[0]

    def _generate_many_with_residency(
        self,
        samples: Sequence[Sample],
        *,
        sync_weights: bool,
        sleep_rollout: bool,
        score_inline: bool,
    ) -> List[Sample]:
        """Generate every sample inside ONE wake/offload window (FlashGRPO's step groups)."""
        with self._rollout_window(sync_weights=sync_weights, sleep_rollout=sleep_rollout):
            results = [
                self.reward_stack.rollout_and_score(item) if score_inline else self.rollout.generate(item)
                for item in samples
            ]
            if score_inline:
                self.reward_stack.collect_garbage()
                self._log_reward_stack_timing()
        return results

    def _log_reward_stack_timing(self) -> None:
        """Surface the stack's per-rank generate/score split on the driver, where worker logs do not reach."""
        per_rank = self.reward_stack.timing()
        keys = ("generate_s", "score_s", "wall_s")
        peak = {k: max(float(t[k]) for t in per_rank) for k in keys}
        mean = {k: sum(float(t[k]) for t in per_rank) / len(per_rank) for k in keys}
        logger.info(
            "reward stack timing: ranks=%d rows=%d micros=%d generate_s=%.2f/%.2f score_s=%.2f/%.2f "
            "wall_s=%.2f/%.2f (max/mean)",
            len(per_rank),
            int(per_rank[0]["rows"]),
            int(per_rank[0]["micros"]),
            peak["generate_s"],
            mean["generate_s"],
            peak["score_s"],
            mean["score_s"],
            peak["wall_s"],
            mean["wall_s"],
        )
        timer = getattr(self, "_step_timer", None)
        if timer is not None:
            timer.phases["stack_generate"] = timer.phases.get("stack_generate", 0.0) + peak["generate_s"]
            timer.phases["stack_score"] = timer.phases.get("stack_score", 0.0) + peak["score_s"]

    def _rollout_and_score(
        self,
        samples: Sequence[Sample],
        *,
        sync_weights: bool = False,
        rollout_id: int = 0,
        group_sde_indices: Optional[Sequence[int]],
    ) -> Tuple[Sample, float]:
        """One ``rollout → reward → advantage`` pass over one or more request Samples."""
        generated = self._generate_many_with_residency(
            samples,
            sync_weights=sync_weights,
            sleep_rollout=not self._residency.policy.rollout_resident,
            score_inline=self.reward_stack is not None,
        )
        if group_sde_indices is not None:
            for item, step in zip(generated, group_sde_indices):
                self._stamp_sde_index_per_sample(item, step)
        sample = generated[0] if len(generated) == 1 else Sample.concat(generated)
        # With no reward configured, ``part.rewards`` stays None and the block below no-ops.
        if self.reward is not None and self.reward_stack is None:
            with self._reward_phase():
                sample = self.reward.score_and_attach(sample)

        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            part.rewards = hydrate(part.rewards)
            if isinstance(part.component_rewards, dict):
                part.component_rewards = {name: hydrate(value) for name, value in part.component_rewards.items()}
            mean_reward = float(part.rewards.to(torch.float32).mean().item())
            if self._algo_requires_advantages:
                part = part.compute_advantages(normalize=True, use_global_std=self._adv_use_global_std)
                sample = sample.with_parts([*sample.parts[:-1], part])

        # Project root-Part metadata onto the gen Part's rows (DiffusionOPD routes
        # on metadata["domain"]); only ever fills an empty field.
        gen_part = sample.parts[-1]
        if not gen_part.metadata:
            root_md = sample.root_metadata(-1)
            if any(md for md in root_md):
                gen_part.metadata = [dict(md) if md else {} for md in root_md]

        self._drop_decoded(sample, rollout_id=rollout_id)
        return sample, mean_reward

    def train_step(
        self,
        window_ids: Sequence[int],
        *,
        num_rollouts: int,
        weight_sync_interval: int = 1,
        force_sync_at: Optional[int] = None,
    ) -> Tuple[TrainStepResult, float]:
        """One accumulation window: rollouts → one optimizer step → one log point."""
        t0 = time.perf_counter()
        samples: List[Sample] = []
        window_rewards: List[float] = []
        for rollout_id in window_ids:
            inputs = self.data_source.get_samples(self.batch_size)
            # FlashGRPO fans the rollout into one generate per SDE-step group so a
            # single optimizer step averages over a spread of sigmas; every other
            # algorithm builds one request.
            if self._flash_per_sample_sde:
                groups = self._build_flash_request_samples(inputs, rollout_id)
                request_samples = [item for _, item in groups]
                group_steps: Optional[List[int]] = [step for step, _ in groups]
            else:
                request_samples = [self._build_request_sample(inputs, rollout_id)]
                group_steps = None
            sync_weights = (rollout_id > 0 and rollout_id % weight_sync_interval == 0) or (rollout_id == force_sync_at)
            sample, mean_reward = self._rollout_and_score(
                request_samples,
                sync_weights=sync_weights,
                rollout_id=rollout_id,
                group_sde_indices=group_steps,
            )
            samples.append(sample)
            window_rewards.append(mean_reward)
        final_id = window_ids[-1]
        training_progress = final_id / max(1, num_rollouts - 1)
        parts = tuple(sample.parts[-1] for sample in samples)
        # First point in the window that needs the trainer on the GPU. With
        # accumulate_rollouts > 1 it stayed parked across every rollout and
        # reward above, so this is one onload per optimizer step rather than one
        # per rollout.
        self._residency.enter(Role.TRAIN)
        # Invalidate before the step, not after: once it has begun the weights can
        # change, so a step that raises must not leave the cache looking current.
        if self._staged_weight_sync:
            self.weight_sync.invalidate()
        result = self.stack.train_track(
            parts if len(parts) > 1 else parts[0],
            training_progress=float(training_progress),
            rollout_id=final_id,
        )
        # Reward stats must cover the whole window: with per-domain scorers each
        # rollout is NaN outside its own domain, so the final sample alone would
        # leave every other domain's curve empty. The pooled keys override the
        # final sample's partial ones inside log_rollout_step.
        extra = pooled_window_reward_metrics(list(parts)) if len(parts) > 1 else None
        self.wandb_logger.log_rollout_step(
            final_id, result, samples[-1], step_time_s=time.perf_counter() - t0, extra_metrics=extra
        )
        return result, sum(window_rewards) / len(window_rewards)

    def evaluate(
        self,
        step: int,
        *,
        sync_weights: bool = True,
        sleep_after: bool = True,
    ) -> float:
        """Periodic eval on the eval set (no training); returns the mean reward."""
        if self.reward is None:
            raise RuntimeError(
                "DiffusionTrainer.evaluate: no reward configured (the recipe has no `reward:` "
                "block) — evaluation scores generations and needs one."
            )
        eval_sp = self._eval_sampling_params
        eval_diffusion = eval_sp["diffusion"]
        sync_requested = bool(sync_weights)
        sleep_requested = sleep_after and not self._residency.policy.rollout_resident
        # A no-sync evaluation must preserve the already-resident adapter.
        # Engines such as SGLang discard their LoRA pool on sleep and cannot
        # reconstruct it without a push, so keep them awake across chunks and
        # sleep once at the end. Base-model engines have no such restriction.
        sleep_each_chunk = sleep_requested and (sync_requested or self.weight_sync is None)
        sleep_at_end = sleep_requested and not sleep_each_chunk
        resync_after_sleep = sleep_requested and sync_requested
        sync_pending = sync_requested
        generated_any = False
        evaluation_succeeded = False
        try:
            scorers = [("reward", self.reward)] + [
                (s.name, s.reward) for s in self._eval_suites if s.data_source is None
            ]
            metrics, sync_pending, pass_generated = self._eval_pass(
                self.data_source,
                self.eval_num_prompts,
                scorers,
                eval_sp,
                step,
                sync_weights=sync_pending,
                resync_after_sleep=resync_after_sleep,
                sleep_rollout=sleep_each_chunk,
                media_prefix="eval",
            )
            generated_any = generated_any or pass_generated
            for suite in self._eval_suites:
                if suite.data_source is not None:
                    n = suite.num_prompts or self.eval_num_prompts
                    suite_metrics, sync_pending, pass_generated = self._eval_pass(
                        suite.data_source,
                        n,
                        [(suite.name, suite.reward)],
                        eval_sp,
                        step,
                        sync_weights=sync_pending,
                        resync_after_sleep=resync_after_sleep,
                        sleep_rollout=sleep_each_chunk,
                        media_prefix=f"eval/{suite.name}",
                    )
                    generated_any = generated_any or pass_generated
                    metrics.update(suite_metrics)
            if not generated_any:
                self._prepare_empty_evaluation(sync_weights=sync_pending, sleep_rollout=sleep_requested)
            elif sleep_at_end:
                self._residency.set(Role.ROLLOUT, False)
            evaluation_succeeded = True
        finally:
            if not evaluation_succeeded:
                _run_cleanup_steps([("evaluation rollout sleep", lambda: self._residency.set(Role.ROLLOUT, False))])
        logger.info(
            "EVAL step %d  (%d samples/prompt, %d steps, %dx%d, cfg=%.1f eta=%.1f)  %s",
            step,
            eval_diffusion.samples_per_prompt,
            eval_diffusion.num_inference_steps,
            eval_diffusion.height,
            eval_diffusion.width,
            eval_diffusion.guidance_scale,
            eval_diffusion.eta,
            "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()),
        )
        self.wandb_logger.log_eval(step, metrics)
        return metrics["reward"]

    def _prepare_empty_evaluation(self, *, sync_weights: bool, sleep_rollout: bool) -> None:
        """Preserve evaluation wake/sync/sleep semantics when every set is empty."""
        prepare_succeeded = False
        try:
            staged = self._cache_adapter_for_push() if sync_weights else False
            self._residency.enter(Role.ROLLOUT)
            if sync_weights:
                self._push_or_sync(staged=staged)
            prepare_succeeded = True
        finally:
            if sleep_rollout or not prepare_succeeded:
                _run_cleanup_steps(
                    [("empty evaluation rollout sleep", lambda: self._residency.set(Role.ROLLOUT, False))]
                )

    def _eval_pass(
        self,
        data_source: Any,
        num_prompts: int,
        scorers: List[Tuple[str, Any]],
        eval_sp: Dict[str, BaseSamplingParams],
        step: int,
        *,
        sync_weights: bool,
        resync_after_sleep: bool,
        sleep_rollout: bool,
        media_prefix: Optional[str] = None,
    ) -> Tuple[Dict[str, float], bool, bool]:
        """One generate→score sweep, pending-sync state, and whether it generated."""
        all_inputs = data_source.get_eval_samples(num_prompts)
        n_prompts = all_inputs.batch_size
        chunk = max(1, self.eval_chunk_prompts)
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        sync_pending = sync_weights
        for start in range(0, n_prompts, chunk):
            sub = all_inputs.slice(start, min(start + chunk, n_prompts))
            _validate_prompt_tree_dp_geometry(
                batch_size=sub.batch_size,
                rollout_dp_size=self.rollout.dp_size,
                reward_dp_size=self.reward.dp_size,
                context=f"evaluation chunk [{start}:{start + sub.batch_size}]",
            )
            request = self._build_request_sample(sub, step, sampling=eval_sp)
            generated = self._generate_with_residency(
                request,
                # Engines whose sleep releases the adapter (e.g. SGLang) need
                # the requested policy re-pushed after every eval-chunk wake.
                sync_weights=sync_pending or resync_after_sleep,
                sleep_rollout=sleep_rollout,
            )
            sync_pending = False
            first_scored: Optional[Sample] = None
            with self._reward_phase(preserve_rollout=not sleep_rollout):
                for name, reward in scorers:
                    scored = reward.score_and_attach(generated)
                    if first_scored is None:
                        first_scored = scored
                    part = scored.parts[-1]
                    rewards = part.rewards
                    if rewards is not None:
                        r = hydrate(rewards).to(torch.float32)
                        if scored is first_scored:
                            # Captions read part.rewards, which remote scoring returns dehydrated.
                            part.rewards = r
                        sums[name] += float(r.sum().item())
                        counts[name] += int(r.numel())
                    components = part.component_rewards
                    if isinstance(components, dict):
                        for component_name, component_values in components.items():
                            component = hydrate(component_values).to(torch.float32)
                            metric_name = f"{name}_{str(component_name).replace('/', '_')}"
                            sums.setdefault(metric_name, 0.0)
                            counts.setdefault(metric_name, 0)
                            sums[metric_name] += float(component.sum().item())
                            counts[metric_name] += int(component.numel())
            # Outside _reward_phase: the driver-side media upload must not hold
            # the train-offload window open.
            if media_prefix and start == 0 and first_scored is not None:
                self._log_eval_media(first_scored, step, prefix=media_prefix)
        metrics = {name: total / max(1, counts[name]) for name, total in sums.items()}
        return metrics, sync_pending, n_prompts > 0

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
        """Minimal training loop: ``num_rollouts`` rollouts in windows of"""
        # ${oc.env:...} interpolations arrive as strings.
        num_rollouts = int(num_rollouts)
        save_interval = int(save_interval)
        interval = max(1, int(weight_sync_interval))
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        # A window is ONE train_track call, so checkpoints can never split it;
        # cadences are checked at window ends and must land on them to fire.
        acc = self.accumulate_rollouts
        if acc > 1:
            if num_rollouts % acc:
                raise ValueError(
                    f"num_rollouts={num_rollouts} is not a multiple of accumulate_rollouts={acc}: "
                    "the MOPD protocol steps once per FULL task cycle."
                )
            if save_interval > 0 and save_interval % acc:
                raise ValueError(
                    f"save_interval={save_interval} is not a multiple of accumulate_rollouts={acc}: "
                    "checkpoints are written between windows, so this cadence would never fire."
                )
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            if self.eval_interval > 0:
                self.evaluate(start_rollout)
            for window_start in range(start_rollout, num_rollouts, acc):
                window_ids = range(window_start, min(window_start + acc, num_rollouts))
                result, mean_reward = self.train_step(
                    window_ids,
                    num_rollouts=num_rollouts,
                    weight_sync_interval=interval,
                    force_sync_at=start_rollout if resumed else None,
                )
                final_id = window_ids[-1]
                self.wandb_logger.log_progress(final_id, num_rollouts, result, mean_reward, logger=logger)
                if self.eval_interval > 0 and (final_id + 1) % self.eval_interval == 0:
                    self.evaluate(final_id + 1)
                self.maybe_save_checkpoint(
                    final_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
        finally:
            self._finish_wandb()
