import dataclasses
import inspect
import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch
from hydra.utils import get_class, get_object, instantiate
from omegaconf import DictConfig, OmegaConf

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import hydrate
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.eval_suites import EvalRewardSuite, build_eval_suites
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import BaseSamplingParams, total_samples_per_prompt
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra
from unirl.utils.wandb_metrics import pooled_window_reward_metrics

logger = logging.getLogger(__name__)


# Per-field eval knobs the overlay replaced (or dropped), and what to write instead.
_RETIRED_EVAL_KEYS = {
    "eval_cfg_text_scale": "eval_sampling: {guidance_scale: X}   (BAGEL family: cfg_text_scale)",
    "eval_num_inference_steps": "eval_sampling: {num_inference_steps: X}",
    "eval_height": "eval_sampling: {height: X}",
    "eval_width": "eval_sampling: {width: X}",
    "eval_media_max_items": "logging: {log_media: true, media_max_items: X}",
    "eval_shift": "no equivalent: the per-request time-shift override was dropped (static-shift models keep their checkpoint shift)",
    "eval_mu": "not needed: dynamic-shift μ re-derives from the eval steps/resolution",
}

# Engine/driver-owned object fields the overlay cannot carry: overrides come from
# plain YAML (never hydra-instantiated), so a nested ``_target_`` would ride into
# the params as a bare dict and only blow up deep inside the first eval's request
# build (``'dict' object has no attribute 'get_sde_indices'``). Rejected by name.
_UNSUPPORTED_OVERLAY_FIELDS = frozenset(
    {"scheduler", "sde_strategy", "sigmas", "noise_group_ids", "init_noise_latent_shape"}
)


def cfg_scale_of(params: Any) -> float:
    """The CFG scale a diffusion params object will actually be sampled with.

    BAGEL-family params carry ``cfg_text_scale``; every other family carries
    ``guidance_scale``. Log lines read the scale through here so they report the
    field the pipeline actually consumes.
    """
    scale = getattr(params, "cfg_text_scale", None)
    return float(params.guidance_scale if scale is None else scale)


def reject_retired_eval_keys(cfg: Any) -> None:
    """Fail fast on the per-field ``eval_*`` knobs that ``eval_sampling:`` replaced.

    Ignoring them would silently evaluate at the rollout's own setting — the exact
    train/eval mismatch the overlay exists to make explicit.
    """
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
    """Return ``sampling_params`` with its ``diffusion`` entry rebuilt for evaluation.

    Eval INHERITS the training ``sampling:`` block and overlays only what the
    recipe asks for, later winning over earlier:

    1. ``eta`` — recipe ``eval_eta`` (default ``0.0``: deterministic ODE eval).
    2. ``samples_per_prompt`` when given — recipe ``eval_samples_per_prompt``.
    3. ``overrides`` — the recipe's ``eval_sampling:`` block: any plain
       :class:`~unirl.types.sampling.DiffusionSamplingParams` field
       (``guidance_scale``, ``num_inference_steps``, ``height`` / ``width``,
       ``seed``, ...). Unknown keys raise rather than being silently dropped,
       and so do the engine/driver-owned object fields
       (``_UNSUPPORTED_OVERLAY_FIELDS``) that plain YAML cannot express.

    CFG needs no knob of its own: an unmentioned ``guidance_scale`` inherits the
    training guidance, so a CFG-off run cannot silently evaluate with CFG on, and
    naming it decouples the two. It is the field the pipeline consumes, so a
    family that reads ``cfg_text_scale`` must be given THAT one — the inert
    sibling raises instead of being accepted and ignored.

    A resolved ``eta <= 0`` then clears the SDE gate (``sde_indices=[]``,
    ``scheduler=None``): eta=0 with gated steps is a contradictory request — the
    central kernel degrades such steps to ODE, and worker-resident schedulers
    (BAGEL) refuse the pair outright. A resolved ``eta > 0`` keeps the training
    gate, whose indices are resolved against the ROLLOUT's step count, so a step
    override is rejected here rather than addressing a schedule it cannot reach.

    The rollout's params are never mutated, so eval settings cannot leak into the
    trajectories the policy is trained on.
    """
    base = sampling_params.get("diffusion")
    if base is None:
        raise ValueError("build_eval_sampling: sampling params carry no `diffusion` entry to override.")
    field_names = {f.name for f in dataclasses.fields(base)}

    updates: Dict[str, Any] = {"eta": float(eta)}
    if samples_per_prompt is not None:
        updates["samples_per_prompt"] = int(samples_per_prompt)
    updates.update(_resolve_overrides(overrides, field_names))

    # Only the cfg_text_scale families declare both; elsewhere the sibling is not a
    # field at all and _resolve_overrides already rejected it.
    if "cfg_text_scale" in field_names and "guidance_scale" in updates:
        raise ValueError(
            f"eval_sampling sets `guidance_scale`, which {type(base).__name__} declares but its "
            "pipeline discards — the eval would silently run at the training CFG. "
            "Set `cfg_text_scale` instead."
        )

    steps = int(updates.get("num_inference_steps", base.num_inference_steps))
    if float(updates["eta"]) <= 0.0:
        updates["sde_indices"] = []
        updates["scheduler"] = None
    elif steps != int(base.num_inference_steps):
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
        reward_cfg: Optional[DictConfig],
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
        accumulate_rollouts: int = 1,
        eval_interval: int = 0,
        eval_num_prompts: int = 64,
        eval_samples_per_prompt: int = 4,
        eval_chunk_prompts: int = 16,
        eval_eta: float = 0.0,
        eval_sampling_cfg: Optional[Any] = None,
        eval_rewards_cfg: Optional[Any] = None,
        task_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        reject_retired_eval_keys(cfg)
        self.batch_size = batch_size
        self._layout = str(layout)
        self._train_fraction = float(train_fraction)
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._adv_use_global_std = bool(adv_use_global_std)
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_chunk_prompts = int(eval_chunk_prompts)
        self.eval_eta = float(eval_eta)
        self._eval_rewards_cfg = eval_rewards_cfg
        self._eval_suites: List[EvalRewardSuite] = []
        self._task_config: Dict[str, Any] = dict(task_config) if task_config else {}
        self._rollout_is_trainside = False
        self._uses_ema = False

        self.data_source = instantiate(data_source_cfg)
        self._data_source_cfg = data_source_cfg

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        # Frozen at init, so overlay contradictions surface at startup rather than an
        # hour in.
        self._eval_sampling_params: Dict[str, BaseSamplingParams] = build_eval_sampling(
            self.sampling_params,
            eta=self.eval_eta,
            samples_per_prompt=self.eval_samples_per_prompt,
            overrides=eval_sampling_cfg,
        )

        self._noise_latent_shape: Optional[list] = (
            None
            if os.environ.get("DISABLE_DRIVER_XT")
            else self._resolve_noise_latent_shape(
                pipeline_cfg=pipeline_cfg,
                model_cfg=bundle_cfg,
                sampling_spec=self.sampling_params.get("diffusion"),
            )
        )
        # Eval may render at its own resolution, so the driver-authored x_T has to
        # match THAT geometry rather than the rollout's.
        self._eval_noise_latent_shape: Optional[list] = (
            None
            if self._noise_latent_shape is None
            else self._resolve_noise_latent_shape(
                pipeline_cfg=pipeline_cfg,
                model_cfg=bundle_cfg,
                sampling_spec=self._eval_sampling_params.get("diffusion"),
            )
        )
        if self.eval_interval > 0 and bool((self.logging_cfg or {}).get("log_media", False)):
            eval_diffusion = self._eval_sampling_params.get("diffusion")
            if self._eval_noise_latent_shape is None or bool(eval_diffusion.disable_driver_xt):
                logger.warning(
                    "logging.log_media is on, but eval has no driver-authored x_T "
                    "(DISABLE_DRIVER_XT / disable_driver_xt, or latent_shape() opted out), so "
                    "every eval draws fresh noise and the eval panel will not be comparable "
                    "across evals."
                )
            elif float(eval_diffusion.eta) > 0.0:
                logger.warning(
                    "logging.log_media is on with eval eta=%.2f > 0: x_T is prompt-keyed, but "
                    "per-step SDE noise is seeded from the eval step's sample ids, so eval "
                    "panels differ by noise as well as policy. Set eval_eta: 0 for a "
                    "like-for-like filmstrip.",
                    float(eval_diffusion.eta),
                )

        self.weight_sync = None
        # None when the recipe has no ``reward:`` block (validated below).
        self.reward = None

        reward_fraction = float(reward_fraction)
        if not 0.0 <= reward_fraction < 1.0:
            raise ValueError(f"reward_fraction must be in [0, 1), got {reward_fraction}")
        if self._layout == "separate" and train_fraction + reward_fraction >= 1.0:
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

        self._validate_reward_config()
        # Cross-rollout gradient accumulation: one optimizer step per M rollouts
        # (the MOPD task-cycle contract).
        self.accumulate_rollouts = int(accumulate_rollouts)
        self._validate_accumulation(stack_cfg)

        # Require rollout counts to divide both policy and reward DP sizes.
        n_samples = batch_size * total_samples_per_prompt(self.sampling_params)
        reward_dp = self.reward.dp_size if self.reward is not None else 1
        if n_samples % self.rollout.dp_size or n_samples % reward_dp:
            raise ValueError(
                f"batch_size({batch_size}) * samples_per_prompt = {n_samples} samples/rollout must be "
                f"divisible by BOTH rollout dp_size={self.rollout.dp_size} and reward dp_size="
                f"{reward_dp}. reward_fraction={reward_fraction} placed reward on its own slab, "
                f"leaving the policy/rollout on {self.rollout.dp_size} GPU(s) — pick batch_size * "
                f"samples_per_prompt divisible by both."
            )

    def _validate_reward_config(self) -> None:
        """A missing ``reward:`` block is legal only for requires_advantages=False algorithms."""
        if self.reward is not None:
            return
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

    def _validate_accumulation(self, stack_cfg: DictConfig) -> None:
        """``accumulate_rollouts > 1`` needs single-update, non-EMA, cycle-aligned cadences."""
        acc = self.accumulate_rollouts
        if acc < 1:
            raise ValueError(f"accumulate_rollouts must be >= 1, got {acc}")
        if acc == 1:
            return
        n_updates = int(stack_cfg.get("num_updates_per_batch", 1) or 1)
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
        # requires_advantages=False algorithms keep rewards for monitoring only.
        self._algo_requires_advantages = getattr(algo_cls, "requires_advantages", True)
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

    def _resolve_noise_latent_shape(
        self, *, pipeline_cfg: DictConfig, model_cfg: DictConfig, sampling_spec: Any
    ) -> Optional[list]:
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

        ``sampling_spec`` selects whose geometry to resolve — the rollout's or
        eval's, which may render at a different resolution.
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
            shape = latent_shape_fn(model_config=model_cfg, sampling_spec=sampling_spec)
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
        noise_latent_shape = self._eval_noise_latent_shape if sampling is not None else self._noise_latent_shape
        diffusion = sp.get("diffusion")
        sde_indices = diffusion.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(
            diffusion, sde_indices=sde_indices, scheduler=None, init_noise_latent_shape=noise_latent_shape
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

    def _rollout_and_score(
        self,
        sample: Sample,
        *,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Sample, float]:
        """One ``rollout → reward → advantage`` pass; training happens per window.

        ``sync_weights`` pushes the latest LoRA into the engine between
        ``wake_up`` and ``generate`` — one wake/sleep instead of two, with this
        ``generate`` already using the fresh adapter.

        Returns ``(sample, mean_reward)`` — the scored Sample (advantages and
        row metadata attached) and the mean unnormalized per-sample reward of
        the frontier gen Part (0.0 if none), for the log line.
        """
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

        # With no reward configured, ``part.rewards`` stays None and the block below no-ops.
        if self.reward is not None:
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
        """One accumulation window: rollouts → one optimizer step → one log point.

        The NAME is a framework seam — :class:`BaseTrainer` wraps ``train_step``
        for transfer-queue buffer reclamation and ``install_phase_timing`` for
        ``perf/*`` attribution. The reclaim fires when this returns: after the
        window has trained, the only point with no live ``TensorRef`` into the
        queue. Returns ``(train_result, window-mean reward)``.
        """
        t0 = time.perf_counter()
        samples: List[Sample] = []
        window_rewards: List[float] = []
        for rollout_id in window_ids:
            inputs = self.data_source.get_samples(self.batch_size)
            sample = self._build_request_sample(inputs, rollout_id)
            sync_weights = (rollout_id > 0 and rollout_id % weight_sync_interval == 0) or (rollout_id == force_sync_at)
            sample, mean_reward = self._rollout_and_score(sample, sync_weights=sync_weights, rollout_id=rollout_id)
            samples.append(sample)
            window_rewards.append(mean_reward)
        final_id = window_ids[-1]
        training_progress = final_id / max(1, num_rollouts - 1)
        parts = tuple(sample.parts[-1] for sample in samples)
        result = self.stack.train_track(
            parts if len(parts) > 1 else parts[0], training_progress=float(training_progress)
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
        """Periodic eval on the eval set (no training); returns the mean reward.

        Mirrors :meth:`_rollout_and_score`'s rollout+reward path but skips advantage/backward.
        Eval sampling INHERITS the training ``sampling:`` block and overlays the
        ``eval_*`` knobs plus the recipe's ``eval_sampling:`` block on top — see
        :func:`build_eval_sampling` for the precedence.
        The training reward plus every shared-set ``eval_rewards`` suite scores the
        SAME generated images over the default
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
        if self.reward is None:
            raise RuntimeError(
                "DiffusionTrainer.evaluate: no reward configured (the recipe has no `reward:` "
                "block) — evaluation scores generations and needs one."
            )
        eval_sp = self._eval_sampling_params
        eval_diffusion = eval_sp.get("diffusion")
        self.rollout.wake_up()
        try:
            if sync_weights and self.weight_sync is not None:
                self.weight_sync.sync()
            scorers = [("reward", self.reward)] + [
                (s.name, s.reward) for s in self._eval_suites if s.data_source is None
            ]
            metrics = self._eval_pass(
                self.data_source, self.eval_num_prompts, scorers, eval_sp, step, media_prefix="eval"
            )
            for suite in self._eval_suites:
                if suite.data_source is not None:
                    n = suite.num_prompts or self.eval_num_prompts
                    metrics.update(
                        self._eval_pass(
                            suite.data_source,
                            n,
                            [(suite.name, suite.reward)],
                            eval_sp,
                            step,
                            media_prefix=f"eval/{suite.name}",
                        )
                    )
        finally:
            if sleep_after:
                self.rollout.sleep()
        logger.info(
            "EVAL step %d  (%d samples/prompt, %d steps, %dx%d, cfg=%.1f eta=%.1f)  %s",
            step,
            int(eval_diffusion.samples_per_prompt),
            int(eval_diffusion.num_inference_steps),
            int(eval_diffusion.height),
            int(eval_diffusion.width),
            cfg_scale_of(eval_diffusion),
            float(eval_diffusion.eta),
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
        *,
        media_prefix: Optional[str] = None,
    ) -> Dict[str, float]:
        """One generate→score sweep over one eval set; returns each scorer's mean.

        The eval prompts are CHUNKED (``eval_chunk_prompts``) so one generate
        never holds N x the KV/decoded on the driver (the it2i memory
        bottleneck). Scores the single scorable (segment-carrying) track with
        every scorer — single-track for now; revisit if multi-track lands.

        ``media_prefix`` names the wandb key family for the preview grid drawn
        from the FIRST chunk (see :meth:`BaseTrainer._log_eval_media`); the first
        scorer's Sample is used, so captions carry its reward.
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
            first_scored: Optional[Sample] = None
            for name, reward in scorers:
                scored = reward.score_and_attach(generated)
                if first_scored is None:
                    first_scored = scored
                rewards = scored.parts[-1].rewards
                if rewards is not None:
                    r = hydrate(rewards).to(torch.float32)
                    if scored is first_scored:
                        # Captions read part.rewards, which remote scoring returns dehydrated.
                        scored.parts[-1].rewards = r
                    sums[name] += float(r.sum().item())
                    counts[name] += int(r.numel())
            if media_prefix and start == 0 and first_scored is not None:
                self._log_eval_media(first_scored, step, prefix=media_prefix)
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
        """Minimal training loop: ``num_rollouts`` rollouts in windows of
        ``accumulate_rollouts``, one ``train_track`` call (= one optimizer
        step) per window; logging/eval/save run between windows.

        ``weight_sync_interval``: sync the adapter into the engine every N
        rollouts (fused into the rollout's generate; no-op trainside).

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` is the output folder (defaults to
        ``./checkpoints``); ``save_mode="auto"`` writes LoRA-only checkpoints
        when LoRA is active and full checkpoints otherwise.
        ``load_dir``: restore from a checkpoint directory and RESUME from its
        saved step — ``num_rollouts`` is the TOTAL budget, so resuming
        checkpoint-500 with ``num_rollouts=600`` runs rollouts 500..599.
        """
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
