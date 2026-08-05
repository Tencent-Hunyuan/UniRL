"""UniRL v2 HunyuanImage3 unified-backbone trainer.

One shared HunyuanImage3 backbone (a single MoE transformer that operates in
``mode="gen_text"`` for AR and ``mode="gen_image"`` for DiT) trained jointly by
two algorithms — ``GRPO`` over the AR ``TextSegment`` and ``FlowGRPO``
over the DiT ``LatentSegment`` — both backward-accumulating into ONE LoRA
adapter with a single optimizer step (see :class:`UnifiedModelTrainStack`).

Two-engine design (mirrors :class:`~unirl.models.pe.pipeline.PEPipeline`'s
two-level fan-out but with the backbone shared). PE composes two in-process
child pipelines (SD3 + Qwen3, two LoRAs); HI3 instead drives TWO standalone
vLLM-Omni engine Remotes that share ONE backbone / ONE LoRA:

- ``ar_rollout`` (modality ``hi3_ar_recaption``, GPUs 0-3): original prompt → ``N``
  think/recaption texts (group-by-prompt → AR GRPO).
- ``dit_rollout`` (modality ``hi3_dit_recaption``, GPUs 4-7): each recaption → ``M``
  images of distinct noise (group-by-recaption → FlowGRPO).

The trainer assembles the lineage itself (pre-forks ``[input, ar_shell,
image_shell]`` then re-roots a flat 1:1 sub-request per engine and fills the
shells, exactly like ``ComposedRolloutEngine.generate``) because the two engines
are independent Remotes, not a composed pipeline. Reward routing then matches
:class:`~unirl.trainer.pe.PETrainer`: score the image Part, credit-assign
the mean image reward up to the AR Part, per-Part GRPO advantages, then ONE
:class:`UnifiedModelTrainStack` step (ar.loss + image.loss → one optimizer step on the
single shared LoRA).

GPU partition: each engine is ONE multi-GPU actor anchored on a distinct worker
via ``pool.create_remote(device_ids=[0])`` / ``[4]`` (NOT plain ``remote()``,
which would bind it to the whole fraction=1.0 scope and collide both engines'
device-env in one process). Each engine clears ``CUDA_VISIBLE_DEVICES`` for its
multi-GPU HI3 modality (see ``engine._HI3_MULTI_GPU_MODALITIES``) and its stage
YAML's ``runtime.devices`` pins AR→0-3 / DiT→4-7 — disjoint physical cards. The
boot-smoke anchor was unsafe only because nothing time-shared the cards; here
the colocate dance (base offloaded during rollout, engines asleep during train)
makes anchoring correct — see ``train_step`` and ``_wire_engine``.

One ``train_step``::

    wake ar+dit; [sync → both]; sample = run_rollout(sample)  # → [input, ar, image]
    sleep ar+dit
    reward.score_and_attach(sample)              # only the frontier image Part is scorable
    sample.propagate_rewards("mean")             # image reward → ar Part
    part.compute_advantages() per Part           # ar groups by prompt, image by recaption
    unified_model_stack.train_track(sample)      # tree-shard lineage → 2 backward → 1 step

Pairs with ``examples/unified_model/hi3_vllmomni.yaml`` and ``unirl/train_unified_model.py``.
Deferred (same as the reference trainers): multi-epoch replay, checkpoint /
eval cadence, structured logging.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import TensorRef, hydrate
from unirl.distributed.tensor.batch import Batch
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict, prepare_input_sample
from unirl.trainer.eval_suites import build_eval_suites
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, BaseSamplingParams, DiffusionSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)


def deep_hydrate(obj: Any) -> Any:
    """Materialize every ``TensorRef`` leaf in ``obj`` to a real tensor, in place.

    The anchored single-actor engines return each track as ONE transport handle
    (a single ref spanning all samples), but the train side is num_devices-way DP and
    slices each track into per-rank shards — a single ref can't be intra-handle
    sliced. Hydrating on the driver fixes the mismatch (the DP dispatch then
    re-shards real tensors), but the driver has no ``TensorTransportRuntime``
    installed, so the runtime-backed ``TensorTransport.hydrate`` is
    unavailable here. ``hydrate`` instead pulls each leaf through
    its ref's ``.materialize(backend=None)`` (a plain ``ray.get`` from the owning worker's store),
    which works from the driver — we walk the nested Batch/dict/list/TUPLE
    structure and apply it to every ``TensorRef``.

    NB: this walks tuples too (rebuilding them), unlike ``_collect_leaves``.
    HI3's trainside rope is now a stacked ``[B, 2, L, D]`` CONCAT tensor;
    the tuple case remains supported for other nested transport payloads.
    """
    if isinstance(obj, TensorRef):
        return hydrate(obj)
    if isinstance(obj, Batch):
        for f in dataclasses.fields(obj):
            v = getattr(obj, f.name)
            if v is not None:
                new = deep_hydrate(v)
                if new is not v:
                    setattr(obj, f.name, new)
        return obj
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            obj[k] = deep_hydrate(obj[k])
        return obj
    if isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = deep_hydrate(obj[i])
        return obj
    if isinstance(obj, tuple):
        return tuple(deep_hydrate(x) for x in obj)
    return obj


class UnifiedModelTrainer(BaseTrainer):
    """HunyuanImage3 unified-backbone joint trainer (AR + DiT, one LoRA)."""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        batch_size: int,
        bundle_cfg: DictConfig,
        pipeline_cfg: DictConfig,
        backend_cfg: DictConfig,
        reward_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        task_config: Optional[Dict[str, Any]] = None,
        ar_rollout_cfg: Optional[DictConfig] = None,
        dit_rollout_cfg: Optional[DictConfig] = None,
        rollout_cfg: Optional[DictConfig] = None,
        sync_cfg: Optional[DictConfig] = None,
        dump_dir: Optional[str] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = True,
        eval_interval: int = 0,
        eval_num_prompts: int = 32,
        eval_cfg_text_scale: float = 4.0,
        eval_eta: float = 0.0,
        eval_rewards_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._enable_fsdp_offload = bool(enable_fsdp_offload)

        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.eval_eta = float(eval_eta)

        self.dump_dir = str(dump_dir) if dump_dir else None
        self._dump_rollout_id = 0
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        self._task_config: Dict[str, Any] = dict(task_config) if task_config else {}

        self.weight_sync = None

        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)
            self._eval_suites = build_eval_suites(
                eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
            )

            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)

            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )

            self._single_engine = rollout_cfg is not None
            diffusion_params = self.sampling_params.get("diffusion")
            self._shared_advantage = int(diffusion_params.samples_per_prompt) == 1
            self._rollout_is_trainside = False
            if self._single_engine:
                self.dp = 1
                self.ar_rollouts = []
                self.dit_rollouts = []
                self.ar_rollout = None
                self.dit_rollout = None
                rollout_parsed = parse_hydra_cfg(rollout_cfg)
                self._rollout_is_trainside = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
                if self._rollout_is_trainside:
                    self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)
                    self._enable_fsdp_offload = False
                else:
                    self.rollout = remote(**rollout_parsed)
                return

            if ar_rollout_cfg is None or dit_rollout_cfg is None:
                raise ValueError(
                    "UnifiedModelTrainer: two-engine mode needs ar_rollout_cfg + dit_rollout_cfg; "
                    "pass a single rollout_cfg for single-engine (M=1 / UniGRPO) mode."
                )

            # Offload the frozen base before engine boot to avoid colocated OOMs.
            if self._enable_fsdp_offload:
                self.backend.offload()

            # Anchor AR and DiT engines on distinct workers to prevent GPU overlap.
            per_node = self.pool.devices_per_node
            # Require at least eight GPUs per node so engine pairs do not split across nodes.
            if per_node < 8:
                raise ValueError(
                    "UnifiedModelTrainer: HI3 needs >= 8 devices/node for one "
                    "(AR 0-3, DiT 4-7) engine pair per node; got "
                    f"devices_per_node={per_node}."
                )
            self.dp = max(1, self.pool.num_devices // per_node)
            self.ar_rollouts = []
            self.dit_rollouts = []
            for r in range(self.dp):
                base = r * per_node
                # Boot engines serially and off rank 0 to avoid warmup and LoRA-sync deadlocks.
                ar = self._wire_engine(ar_rollout_cfg, anchor_device=base + 1)
                ar.sleep()
                self.ar_rollouts.append(ar)
                dit = self._wire_engine(dit_rollout_cfg, anchor_device=base + 4)
                dit.sleep()
                self.dit_rollouts.append(dit)
            self.ar_rollout = self.ar_rollouts[0]
            self.dit_rollout = self.dit_rollouts[0]

            if sync_cfg is not None:
                self.weight_sync = remote_hydra(sync_cfg, backend=self.backend)
                self.weight_sync.set_rollout_targets(
                    [(eng.role_name, eng.workers) for eng in self.ar_rollouts + self.dit_rollouts]
                )

    def _wire_engine(self, cfg: DictConfig, *, anchor_device: int) -> Any:
        """Build ONE multi-GPU vLLM-Omni engine actor anchored on one worker.

        ``device_ids=[anchor_device]`` pins the actor to a SINGLE worker (one
        process), not the whole placement scope — the engine is one TP-parallel
        Omni server, not a per-device DP replica. Inside the Omni subprocess the
        engine clears ``CUDA_VISIBLE_DEVICES`` and its stage YAML's
        ``runtime.devices`` spreads the TP group across its physical cards; using
        a distinct anchor per engine keeps the two engines' device-env setup in
        separate processes so they pin to disjoint cards (see the call site).
        The standalone HI3 engines take no ``pipeline`` (they boot their own
        Omni), so nothing sibling-handle-resolved is forwarded.
        """
        parsed = parse_hydra_cfg(cfg)
        role_cls = parsed.pop("role_cls")
        return self.pool.create_remote(role_cls, device_ids=[anchor_device], init_kwargs=parsed)

    def _build_request_sample(
        self,
        inputs: Sample,
        rollout_id: int,
        *,
        sampling: Optional[Dict[str, BaseSamplingParams]] = None,
    ) -> Sample:
        """Turn a data-source batch of ``P`` prompts into the unified request ``Sample``.

        Namespaces the data source's single text input Part, then pre-forks
        the unified lineage shells ``[input, ar_shell(P*N), image_shell(P*N*M)]``
        (located by sampling-params type); ``run_rollout`` drives the two engines
        and fills these shells. ``rollout_id`` keys the
        diffusion SDE-step schedule (``scheduler`` nulled so only the concrete
        ``sde_indices`` ride) and salts the root ids. The AR sub-block has no SDE
        machinery and is left untouched. ``sampling`` optionally supplies the
        evaluation sampling parameters.
        """
        base = sampling if sampling is not None else self.sampling_params
        diff_params = base.get("diffusion")
        ar_params = base.get("ar")
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        disable_xt = bool(os.environ.get("DISABLE_DRIVER_XT")) or bool(getattr(diff_params, "disable_driver_xt", False))
        diffusion = dataclasses.replace(
            diff_params, sde_indices=sde_indices, scheduler=None, disable_driver_xt=disable_xt
        )
        request = prepare_input_sample(
            inputs,
            rollout_id,
            allowed_primitives={"text"},
            caller="UnifiedModelTrainer._build_request_sample",
            root_control=dict(self._task_config),
            require_single_input_part=True,
        )
        return request.fork(ar_params.samples_per_prompt, sampling_params=ar_params).fork(
            diffusion.samples_per_prompt, sampling_params=diffusion
        )

    def run_rollout(self, sample: Sample) -> Sample:
        """DP rollout: scatter the ``P`` prompt-trees of the request ``Sample``
        across the ``dp`` engine replicas (one (AR, DiT) pair per node), run each
        on its replica, then ``Sample.concat`` the per-replica filled Samples.
        ``dp<=1`` or ``P<=1`` falls back to the single-replica path.

        v1 runs the replicas SEQUENTIALLY — this validates placement + the
        scatter/concat correctness; issuing the per-replica ``generate()`` as Ray
        futures for true concurrent throughput is the follow-up (handoff §8).

        HI3 trainside carries rope_cache as a stacked per-sample CONCAT tensor,
        so DP concat preserves row alignment. The vLLM-Omni response deliberately
        omits its engine-layout rope and replay rebuilds an HF-native rope instead.
        """
        valid_layout = (
            len(sample.parts) == 3
            and sample.parts[0].is_root
            and isinstance(sample.parts[1].sampling_params, ARSamplingParams)
            and isinstance(sample.parts[2].sampling_params, DiffusionSamplingParams)
        )
        if not valid_layout:
            raise ValueError(
                "UnifiedModelTrainer.run_rollout requires exactly "
                f"[input, ar_shell, image_shell]; got {len(sample.parts)} Parts with sampling types "
                f"{[type(part.sampling_params).__name__ for part in sample.parts]}."
            )
        if self._single_engine:
            return self.rollout.generate(sample)

        n = sample.parts[0].batch_size
        if self.dp <= 1 or n <= 1:
            return self._run_rollout_one(self.ar_rollouts[0], self.dit_rollouts[0], sample)

        groups = sample.split()
        bounds = [(n * r) // self.dp for r in range(self.dp + 1)]
        shards: list[Sample] = []
        for r in range(self.dp):
            lo, hi = bounds[r], bounds[r + 1]
            if lo >= hi:
                continue
            sub = Sample.concat(groups[lo:hi])
            shards.append(self._run_rollout_one(self.ar_rollouts[r], self.dit_rollouts[r], sub))
        return Sample.concat(shards)

    def _run_rollout_one(self, ar_engine: Any, dit_engine: Any, sample: Sample) -> Sample:
        """One (AR, DiT) engine pair: fill the unified ``[input, ar, image]`` lineage.

        Drives the given ``ar_engine`` / ``dit_engine`` pair for this replica's
        prompt-trees::

            P prompts ──AR engine──▶ P*N recaptions  (root "ar", groups by prompt)
                      ──DiT engine─▶ P*N*M images     ("image", groups by recaption)

        Each engine runs FLAT (re-rooted, 1:1) — the vLLM-Omni adapters require the
        input primitive 1:1 with the gen samples — so the AR engine sees ``P*N``
        pre-expanded prompts and the DiT engine sees ``P*N*M`` (the original prompt
        plus the recaption chained as a ``cot_text`` input Part via
        :meth:`Part.input_child`). Their per-sample outputs are mapped back, by row
        order, onto the unified lineage shells (:meth:`Part.fill`) — both sides are
        group-by-parent in the same order, so the rows line up. Each image's
        ``r{rollout_id}:d{k}`` root makes its x_T per-rollout-VARYING (the engine
        derives the noise key from the gen Part ids).
        """
        input_part = sample.parts[0]
        ar_shell = sample.gen_part(ARSamplingParams)
        image_shell = sample.gen_part(DiffusionSamplingParams)
        prompts = input_part.primitives.get("text")
        if not isinstance(prompts, Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: input Part must contain a 'text' Texts primitive.")
        n_rec = int(ar_shell.sampling_params.samples_per_prompt)
        n_img = int(image_shell.sampling_params.samples_per_prompt)
        rid = int(self._dump_rollout_id)

        ar_texts = Texts(texts=[t for t in prompts.texts for _ in range(n_rec)])
        n_ar = len(ar_texts.texts)
        ar_input = Part.input(
            [f"r{rid}:a{k}" for k in range(n_ar)],
            primitives={"text": ar_texts},
            control=dict(input_part.control),
        )
        ar_request = (
            Sample.request(ar_input)
            .fork(1, sampling_params=image_shell.sampling_params)
            .fork(1, sampling_params=ar_shell.sampling_params)
        )
        ar_out = ar_engine.generate(ar_request)
        ar_gen = ar_out.parts[-1]
        recaptions = ar_gen.primitives.get("text")
        if not isinstance(recaptions, Texts) or len(recaptions.texts) != n_ar:
            got = len(recaptions.texts) if isinstance(recaptions, Texts) else type(recaptions).__name__
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: AR engine must return {n_ar} decoded Texts (= P*N); got {got}."
            )
        ar_part = ar_shell.fill(
            segment=ar_gen.segment,
            primitives={"text": recaptions},
            conditions=dict(ar_gen.conditions),
            weight_version=ar_gen.weight_version,
        )

        dit_prompts = Texts(texts=[prompts.texts[i // n_rec] for i in range(n_ar) for _ in range(n_img)])
        dit_cot = Texts(texts=[recaptions.texts[i] for i in range(n_ar) for _ in range(n_img)])
        dit_input = Part.input(
            [sid.replace("/", "_") for sid in image_shell.sample_ids],
            primitives={"text": dit_prompts},
            control=dict(input_part.control),
        )
        cot_input = dit_input.input_child(primitives={"text": dit_cot})
        dit_out = dit_engine.generate(
            Sample.request(dit_input, cot_input).fork(1, sampling_params=image_shell.sampling_params)
        )
        img_gen = dit_out.parts[-1]
        if len(img_gen.sample_ids) != len(image_shell.sample_ids):
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: DiT engine returned {len(img_gen.sample_ids)} image(s) "
                f"but the image shell expects {len(image_shell.sample_ids)} (= P*N*M). The DiT engine must be 1:1."
            )
        image_part = image_shell.fill(
            segment=img_gen.segment,
            primitives=dict(img_gen.primitives),
            primitive_metadata=dict(img_gen.primitive_metadata),
            conditions=dict(img_gen.conditions),
            media_preview=img_gen.media_preview,
            weight_version=img_gen.weight_version,
        )

        # Materialize engine outputs before DP reshards a single transport handle.
        deep_hydrate(ar_part)
        deep_hydrate(image_part)
        return Sample(parts=[input_part, ar_part, image_part])

    def train_step(
        self,
        sample: Sample,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)`` — ``mean_reward`` is the
        mean unnormalized image reward (for the log line). ``rollout_id`` keys
        the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        if self._single_engine:
            sample = self.run_rollout(sample)
        else:
            if sync_weights and self.weight_sync is not None:
                if self._enable_fsdp_offload:
                    self.backend.onload()
                self.weight_sync.extract()
                if self._enable_fsdp_offload:
                    self.backend.offload()
            for _eng in self.ar_rollouts + self.dit_rollouts:
                _eng.wake_up()
            try:
                if sync_weights and self.weight_sync is not None:
                    self.weight_sync.push()
                sample = self.run_rollout(sample)
            finally:
                for _eng in self.ar_rollouts + self.dit_rollouts:
                    _eng.sleep()
                if self._enable_fsdp_offload:
                    self.backend.onload()

        ar_idx = sample.gen_part_index(ARSamplingParams)
        img_idx = sample.gen_part_index(DiffusionSamplingParams)

        sample = self.reward.score_and_attach(sample)
        img_part = sample.parts[img_idx]
        if img_part.rewards is not None:
            img_part.rewards = hydrate(img_part.rewards)
        if isinstance(img_part.component_rewards, dict):
            img_part.component_rewards = {name: hydrate(value) for name, value in img_part.component_rewards.items()}

        sample = sample.propagate_rewards(op="mean")

        mean_reward = 0.0
        di_rewards = sample.parts[img_idx].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        if self.dump_dir:
            self._dump_rollout(self._dump_rollout_id, sample)

        new_parts = list(sample.parts)
        new_parts[ar_idx] = new_parts[ar_idx].compute_advantages(normalize=True)
        if self._shared_advantage:
            if new_parts[img_idx].batch_size != new_parts[ar_idx].batch_size:
                raise ValueError(
                    "UnifiedModelTrainer: shared M=1 advantage requires equal AR/image batch sizes; "
                    f"got {new_parts[ar_idx].batch_size} and {new_parts[img_idx].batch_size}."
                )
            new_parts[img_idx] = dataclasses.replace(new_parts[img_idx], advantages=new_parts[ar_idx].advantages)
        else:
            new_parts[img_idx] = new_parts[img_idx].compute_advantages(normalize=True)
        sample = sample.with_parts(new_parts)

        self._drop_decoded(sample, rollout_id=rollout_id)
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            sample,
            training_progress=float(training_progress),
        )
        self.wandb_logger.log_rollout_step(
            rollout_id,
            results,
            sample,
            step_time_s=time.perf_counter() - t0,
            extra_metrics={"sync_weights": float(bool(sync_weights))},
        )

        if self._enable_fsdp_offload and not self._single_engine:
            self.backend.offload()
        return results, mean_reward

    def _dump_rollout(self, rollout_id: int, sample: Any) -> None:
        """Best-effort intrusive dump of one rollout to ``self.dump_dir``.

        Writes ``rollout_<id>/`` with:

        - ``samples.jsonl`` — one line per sample: original prompt, AR output
          text (the ``<think>``/``<recaption>`` that conditions DiT in
          think_recaption mode), image reward, sample/parent ids.
        - ``img_<k>.png`` — the decoded DiT image for sample ``k``.

        Wrapped so a dump failure never aborts training — observation only.
        """
        try:
            out_dir = os.path.join(self.dump_dir, f"rollout_{rollout_id}")
            os.makedirs(out_dir, exist_ok=True)

            prompts_obj = sample.parts[0].primitives.get("text")
            prompts = list(prompts_obj.texts) if isinstance(prompts_obj, Texts) else []

            ar_part = next((p for p in sample.parts[1:] if isinstance(p.sampling_params, ARSamplingParams)), None)
            ar_decoded = ar_part.primitives.get("text") if ar_part is not None else None
            ar_texts = list(ar_decoded.texts) if isinstance(ar_decoded, Texts) else []

            image_part = next(
                (p for p in sample.parts[1:] if isinstance(p.sampling_params, DiffusionSamplingParams)), None
            )
            img_decoded = image_part.primitives.get("image") if image_part is not None else None
            sample_ids = list(image_part.sample_ids) if image_part is not None else []
            parent_ids = list(image_part.group_ids) if image_part is not None else []

            rewards = None
            if image_part is not None and image_part.rewards is not None:
                rewards = hydrate(image_part.rewards).to(torch.float32).tolist()

            n_imgs = 0
            if img_decoded is not None and getattr(img_decoded, "pixels", None) is not None:
                from torchvision.utils import save_image

                pixels = hydrate(img_decoded.pixels).detach().to(torch.float32).clamp(0, 1).cpu()
                n_imgs = int(pixels.shape[0])
                for k in range(n_imgs):
                    save_image(pixels[k], os.path.join(out_dir, f"img_{k}.png"))

            ar_params = self.sampling_params.get("ar")
            diff_params = self.sampling_params.get("diffusion")
            n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            n_img = max(1, int(diff_params.samples_per_prompt))
            n = max(len(sample_ids), n_imgs)
            with open(os.path.join(out_dir, "samples.jsonl"), "w") as f:
                for k in range(n):
                    p_idx = k // (n_rec * n_img)
                    a_idx = k // n_img
                    f.write(
                        json.dumps(
                            {
                                "sample_id": sample_ids[k] if k < len(sample_ids) else None,
                                "parent_id": parent_ids[k] if k < len(parent_ids) else None,
                                "prompt": prompts[p_idx] if p_idx < len(prompts) else None,
                                "ar_text_fed_to_dit": ar_texts[a_idx] if a_idx < len(ar_texts) else None,
                                "image_reward": rewards[k] if (rewards is not None and k < len(rewards)) else None,
                                "image_file": f"img_{k}.png" if k < n_imgs else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info("[HI3-DUMP] rollout %d → %s (%d samples, %d images)", rollout_id, out_dir, n, n_imgs)
        except Exception as exc:  # noqa: BLE001 — dump must never break training
            logger.warning("[HI3-DUMP] rollout %d dump failed (non-fatal): %s", rollout_id, exc)

    def evaluate(self, step: int) -> float:
        """Periodic eval on the eval set (no training); returns the mean image reward.

        Mirrors :meth:`train_step`'s rollout+reward path but skips
        credit-assign/advantage/backward: run the ``P→P*N→P*N*M`` fan-out through
        :meth:`run_rollout` (works on both the single-engine trainside and the
        two-engine HI3 path) at the deterministic best-quality setting (CFG at
        ``eval_cfg_text_scale``, ``eta=eval_eta``) and score ONLY the image
        track — the training reward plus the ``eval_rewards`` suites (see
        :mod:`unirl.trainer.eval_suites`). Logs one ``eval/*`` row; returns
        ``eval/reward``.

        The two-engine path syncs the live adapter into the engines once per
        eval (EXTRACT with the base onloaded → wake → PUSH → sleep, mirroring
        :meth:`train_step`'s ordering) — train_step syncs BEFORE its generate,
        so without this the engines would eval one update stale, and a
        restored-checkpoint baseline eval would see fresh engine weights.
        Pushed weights persist across sleep/wake cycles (as train_step relies
        on), so the passes below just wake/sleep around each chunk's rollout.
        Unlike train_step, eval never onloads the base after the extract: there
        is no backward, so the FSDP state stays offloaded (the steady state)
        throughout. The single-engine trainside path needs none of it (the
        rollout shares the live FSDP modules; ``_enable_fsdp_offload`` is
        forced False).
        """
        base_diffusion = self.sampling_params.get("diffusion")
        replace_kwargs = dict(eta=self.eval_eta)
        if "cfg_text_scale" in {f.name for f in dataclasses.fields(base_diffusion)}:
            replace_kwargs["cfg_text_scale"] = self.eval_cfg_text_scale
        else:
            replace_kwargs["guidance_scale"] = self.eval_cfg_text_scale
        eval_diffusion = dataclasses.replace(base_diffusion, **replace_kwargs)
        eval_sp = {**self.sampling_params, "diffusion": eval_diffusion}
        if not self._single_engine and self.weight_sync is not None:
            if self._enable_fsdp_offload:
                self.backend.onload()
            self.weight_sync.extract()
            if self._enable_fsdp_offload:
                self.backend.offload()
            for eng in self.ar_rollouts + self.dit_rollouts:
                eng.wake_up()
            try:
                self.weight_sync.push()
            finally:
                for eng in self.ar_rollouts + self.dit_rollouts:
                    eng.sleep()
        scorers = [("reward", self.reward)] + [(s.name, s.reward) for s in self._eval_suites if s.data_source is None]
        metrics = self._eval_pass(self.data_source, self.eval_num_prompts, scorers, eval_sp, step)
        for suite in self._eval_suites:
            if suite.data_source is not None:
                n = suite.num_prompts or self.eval_num_prompts
                metrics.update(self._eval_pass(suite.data_source, n, [(suite.name, suite.reward)], eval_sp, step))
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
        """One generate→score sweep over one eval set; returns each scorer's mean.

        Chunked by ``self.batch_size`` (the un-expanded P-prompt req DP-splits,
        so the chunk must be dp-divisible; ``batch_size`` is what training
        runs). A ragged tail (``num_prompts`` not a multiple of ``batch_size``)
        is floored off.
        """
        all_inputs = data_source.get_eval_samples(num_prompts)
        n_prompts = all_inputs.batch_size
        chunk = max(1, self.batch_size)
        usable = n_prompts - n_prompts % chunk or n_prompts
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        for start in range(0, usable, chunk):
            sub = all_inputs.slice(start, min(start + chunk, n_prompts))
            request = self._build_request_sample(sub, step, sampling=eval_sp)
            if self._single_engine:
                generated = self.run_rollout(request)
            else:
                for eng in self.ar_rollouts + self.dit_rollouts:
                    eng.wake_up()
                try:
                    generated = self.run_rollout(request)
                finally:
                    for eng in self.ar_rollouts + self.dit_rollouts:
                        eng.sleep()
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

        ``save_interval``: write a checkpoint every N rollouts (and on the last
        one); ``0`` disables it. ``save_dir`` defaults to ``./checkpoints``;
        ``save_mode="auto"`` writes LoRA-only checkpoints when LoRA is active
        and full checkpoints otherwise. ``load_dir``: restore from a checkpoint
        directory and RESUME from its saved step — ``num_rollouts`` is the TOTAL
        budget.
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
                self._dump_rollout_id = rollout_id
                inputs = self.data_source.get_samples(self.batch_size)
                sample = self._build_request_sample(inputs, rollout_id)
                force_sync = (resumed and rollout_id == start_rollout) or (
                    rollout_id == 0 and bool(os.environ.get("HI3_SYNC_FIRST"))
                )
                sync_weights = force_sync or (rollout_id > 0 and rollout_id % interval == 0)
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
        except Exception:
            logger.exception("Training loop aborted at rollout %s", locals().get("rollout_id", "?"))
            raise
        finally:
            self._finish_wandb()


__all__ = ["UnifiedModelTrainer"]
