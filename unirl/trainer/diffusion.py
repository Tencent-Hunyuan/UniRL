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
        # Colocate memory dance: offload the FSDP train state (params + grads +
        # optimizer) to CPU during the rollout's generate so a colocate
        # vLLM/SGLang engine fits, onload before the train backward. Off by
        # default; only safe (and only set true) for layout=="colocate" with a
        # SEPARATE engine rollout under GRPO — gated again in train_step.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        # FlowDPPO advantage parity: when True, Part.compute_advantages
        # keeps the per-group mean but divides by ONE batch-wide std (the v1
        # ``use_global_std=True`` scale) instead of each prompt's own std. Off by
        # default → unchanged per-group GRPO normalization for every other recipe.
        self._adv_use_global_std = bool(adv_use_global_std)
        # Periodic eval on the eval set (run.eval_data_path), logged under eval/*.
        # eval_interval=0 disables it (zero-impact for runs that don't set it).
        # Diffusion eval generates at the deterministic best-quality setting
        # (cfg_text=eval_cfg_text_scale, eta=eval_eta) and CHUNKS the eval prompts
        # (eval_chunk_prompts): one generate over the whole eval set would hold N x
        # the KV/decoded on the driver (the it2i memory bottleneck).
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_samples_per_prompt = int(eval_samples_per_prompt)
        self.eval_chunk_prompts = int(eval_chunk_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.eval_eta = float(eval_eta)
        # eval_rewards: extra eval-only reward suites (multi-reward eval for
        # checkpoint selection), built next to the training reward in
        # _wire_eval_suites so they share its placement — see unirl.trainer.eval_suites.
        self._eval_rewards_cfg = eval_rewards_cfg
        self._eval_suites: List[EvalRewardSuite] = []
        # Per-request routing metadata pinned by the recipe (e.g. {"task": "it2i"}),
        # forwarded onto every request Part's ``control``. Pinning the task makes a
        # dataset that is MISSING source images fail loudly in the pipeline (it2i
        # requires an input image) instead of silently degrading to t2i. Empty ⇒ the
        # pipeline infers the task as before (unchanged for every other recipe).
        self._task_config: Dict[str, Any] = dict(task_config) if task_config else {}
        # Set in _build_rollout: True when the rollout is the trainside
        # direct-sampling engine (it reuses the train model → must NOT offload).
        self._rollout_is_trainside = False
        # Set in _build_train_side: True only for the DiffusionNFT algorithm, which
        # needs the EMA dual-adapter swap around rollout. Stays False for GRPO
        # so its hot path is untouched.
        self._uses_ema = False

        # Driver-side data iterator (not a Remote). The raw cfg is kept so
        # eval_rewards suites can clone it with their own prompt paths.
        self.data_source = instantiate(data_source_cfg)
        self._data_source_cfg = data_source_cfg

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)

        # Per-sample latent shape for the driver-authored x_T recipe (see
        # _build_request_sample), resolved ONCE here via the pipeline's framework-level
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

        # Reward placement, orthogonal to the rollout ``layout`` below.
        # ``reward_fraction > 0`` carves reward its OWN disjoint slab of that
        # fraction of the pool, opened LAST so train/rollout keep their cards —
        # mirroring how ``layout="separate"`` gives the secondary role (rollout)
        # the tail of the pool. The reward model then lives on dedicated GPUs and
        # never shares — nor offload-thrashes — the policy's cards ("reward doesn't
        # steal cards"). ``reward_fraction == 0`` (default) leaves reward as a
        # train-side sibling = the unchanged behavior; ``train_fraction`` keeps its
        # whole-pool meaning either way. Cross-slab ``reward.score_and_attach``
        # needs no change: it is DP_SCATTER-dispatched over the reward role's own
        # workgroup and its tensor args cross slabs via the standard TensorRef NCCL
        # path, exactly as the existing ``layout="separate"`` rollout slab does.
        reward_fraction = float(reward_fraction)
        if not 0.0 <= reward_fraction < 1.0:
            raise ValueError(f"reward_fraction must be in [0, 1), got {reward_fraction}")
        if self._layout == "separate" and train_fraction + reward_fraction >= 1.0:
            raise ValueError(
                f"layout='separate' leaves no rollout GPUs: train_fraction ({train_fraction}) "
                f"+ reward_fraction ({reward_fraction}) must be < 1.0"
            )
        reward_separate = reward_fraction > 0.0

        # Construction (_build_train_side / _build_rollout) is shared; only the
        # placement topology and the train→rollout sync wiring differ per layout.
        # ``reward_cfg=None`` when reward owns its own slab (built last, below) so
        # ``_build_train_side`` skips it; otherwise reward is a train-side sibling.
        train_cfgs = dict(
            bundle_cfg=bundle_cfg,
            pipeline_cfg=pipeline_cfg,
            backend_cfg=backend_cfg,
            reward_cfg=(None if reward_separate else reward_cfg),
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
            # Rollout slab = the pool minus train minus the (optional) reward slab.
            # Top-level ``fraction`` is relative to the WHOLE pool (placement.py).
            with placement(self.pool, fraction=1.0 - train_fraction - reward_fraction, shared_workers=True):
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=False)
            if self.weight_sync is not None:
                self._connect_separate(sync_cfg)
        else:
            # Single slab: train + rollout are siblings on one Worker; reward (if
            # separate) takes the tail below, so this slab is the pool minus reward.
            with placement(self.pool, fraction=1.0 - reward_fraction, shared_workers=True):
                self._build_train_side(**train_cfgs)
                self.rollout = self._build_rollout(rollout_cfg, allow_pipeline=True)
                if sync_cfg is not None:
                    # Colocated handlers (tensor/ipc) take the engine as a local sibling.
                    self.weight_sync = remote_hydra(sync_cfg, backend=self.backend, rollout=self.rollout)

        # Reward's own disjoint slab = the tail of the pool, opened LAST so
        # train/rollout keep their cards. Skipped when colocated (built above as a
        # train-side sibling); cross-slab scoring uses the same dispatch/NCCL path.
        if reward_separate:
            with placement(self.pool, fraction=reward_fraction, shared_workers=True):
                self.reward = remote_hydra(reward_cfg)
                self._wire_eval_suites()

        # Pre-flight for the reward_fraction footgun: when reward takes its own
        # slab the policy/rollout DP is the REDUCED card count, and the per-rollout
        # sample count must divide BOTH the rollout DP and the reward DP — else the
        # DP_SCATTER fails deep in generate()/score_and_attach with an opaque "not
        # divisible by dp_size" error. Fail early here, naming the knob.
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
        # DiffusionNFT resolves its frozen reference adapter off ``backend.ema`` (the
        # FSDPBackend owns the dual-adapter EMA), and FlowGRPO / FlowDPPO reach the
        # trainable model directly (LoRA disabled = the ``beta`` KL reference policy),
        # so both need the backend sibling injected alongside ``pipeline``. GRPO takes
        # neither and would reject the extra kwarg, so gate on the algorithms' declared
        # flags (``requires_ema_rollout`` / ``requires_backend``).
        # ``requires_ema_rollout`` additionally drives the eval-EMA swap around
        # ``generate`` in ``train_step``: on-policy algorithms MUST sample with the
        # trainable weights so the first-step importance ratio is 1.
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
        # ``_target_`` may point at the pipeline class (e.g. SD3Pipeline) OR at a
        # factory classmethod (e.g. LTX2Pipeline.from_bundle, whose __init__ needs
        # pre-built stages). ``get_class`` rejects the latter ("non-class of type
        # 'method'"), so resolve the dotpath generically with ``get_object`` and
        # recover the owning class from a bound (class)method via ``__self__``.
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
        # Driver-authoritative x_T, shipped as a deterministic RECIPE. The driver is
        # the single source of initial noise: the engine derives the per-rollout
        # x_T noise key from the gen Part's ``sample_ids``, so the rollout salt MUST
        # live in those ids — hence ``r{rollout_id}:{sample_id}`` roots, fanned out
        # by ``fork`` so every gen sample inherits the rollout-keyed lineage.
        # base_seed rides on ``sampling_params.seed`` and the latent shape is the
        # pipeline's own geometry (``self._noise_latent_shape``, resolved once in
        # __init__, pinned above). Each engine regenerates the BYTE-IDENTICAL x_T
        # from this recipe (CPU-fp32 randn, then moved to the engine device — CPU
        # randn is bit-stable across machines for a fixed torch version, which is
        # what makes trainside / vllm / sglang agree to the byte; verified across
        # nodes+clusters on torch 2.11.0).
        # So x_T is:
        #   - per-rollout-VARYING (rollout_id in the key) → genuine exploration,
        #   - per-sample-UNIQUE   (stable sample id in the key) → diverse GRPO groups,
        #   - IDENTICAL across engines for a given (seed, rollout) → curves align,
        #   - reproducible under resume / re-shard / re-batch (ids are STABLE, not a
        #     positional batch index, so a sample keeps its x_T wherever it lands).
        # ``init_same_noise=True`` (read off the params engine-side) keys by prompt
        # group instead (siblings share). Root cause this fixes: each engine used to
        # draw its OWN x_T from independent RNG → divergent reward curves; a single
        # driver-authored x_T removes that. Opt out with DISABLE_DRIVER_XT=1
        # (resolved in __init__ → shape None here).
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text", "image", "video"},
            caller="DiffusionTrainer._build_request_sample",
            root_control=dict(self._task_config),
        )
        samples_per_prompt = total_samples_per_prompt(sp)
        request = request.fork(samples_per_prompt, sampling_params=diffusion)

        # Eval x_T is keyed on prompt CONTENT plus sibling ordinal instead of the
        # rollout id. This keeps the same prompt/sample slot byte-identical across
        # steps/checkpoints while K>1 siblings remain distinct. The explicit keys
        # live on the gen Part as a CONCAT field so DP splitting preserves alignment.
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
        sample = self.rollout.generate(sample)
        if _inproc_ema_swap:
            self.backend.restore_from_eval()
        self.rollout.sleep()
        if _do_fsdp_offload:
            self.backend.onload()

        # Score the frontier gen Part (Sample -> Sample; the reward service is
        # migrated alongside on its own branch — see the LIN-480 plan).
        sample = self.reward.score_and_attach(sample)

        part = sample.parts[-1]
        mean_reward = 0.0
        if part.rewards is not None:
            # Hydrate in place so the wandb reward/advantage stats reuse this
            # fetch instead of re-pulling the TensorRef from the worker.
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
        # Override only the "diffusion" entry of the modality-keyed sampling dict
        # (mirrors the AR trainer's evaluate()). ``cfg_text_scale`` only exists
        # on Bagel's sampling params; for the standard DiffusionSamplingParams
        # (Qwen-Image, SD3, ...) the CFG strength lives in ``guidance_scale``,
        # so fall back to that when the field is absent.
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
            # Default pass: training reward + shared-set suites score the SAME images.
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
                # Every scorer receives the same unscored Sample. Feeding one
                # scorer's returned Sample into the next would be rejected as a
                # pre-scored frontier and would couple otherwise independent suites.
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
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(num_rollouts=num_rollouts)
        try:
            if self.eval_interval > 0:
                self.evaluate(start_rollout)  # baseline eval before any training
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate.
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
