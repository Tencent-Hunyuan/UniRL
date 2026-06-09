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

- ``ar_rollout`` (modality ``ar_recaption``, GPUs 0-3): original prompt → ``N``
  think/recaption texts (group-by-prompt → AR GRPO).
- ``dit_rollout`` (modality ``dit_recaption``, GPUs 4-7): each recaption → ``M``
  images of distinct noise (group-by-recaption → FlowGRPO).

The trainer assembles the lineage itself (``make_root_track(N)`` /
``fork_track(M)``, exactly like ``PEPipeline.generate``) because the two engines
are independent Remotes, not a composed pipeline. Reward routing then matches
:class:`~unirl.trainer.pe.PETrainer`: score the image track, credit-assign
the mean image reward up to the AR track, per-track GRPO advantages, then ONE
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

    wake ar+dit; [sync → both]; ar_resp = ar_rollout.generate(ar_req)
    img_shell = ar_track.fork_track(M); dit_resp = dit_rollout.generate(dit_req)
    sleep ar+dit
    reward.score_and_attach(image track)         # only the image track is scorable
    resp.propagate_rewards("mean")               # image reward → ar track
    track.compute_advantages() per track         # ar groups by prompt, image by ar-sample
    unified_model_stack.train_track(ar_track, image_track) # 2 backward → 1 optimizer step

Pairs with ``examples/unified_model/hi3_vllmomni.yaml`` and ``unirl/train_unified_model.py``.
Deferred (same as the reference trainers): multi-epoch replay, checkpoint /
eval cadence, structured logging.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement
from unirl.distributed.tensor.batch import Batch
from unirl.distributed.tensor.transport import TensorMeta
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack, _hydrate_tensor_meta, _track_with_field
from unirl.types.sampling import BaseSamplingParams, get_ar_params, get_diffusion_params
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names produced by the vLLM-Omni HI3 rollout (see
# ``rollout/engine/vllm_omni/response.py``): "ar" is the root (TextSegment,
# groups by prompt), "image" is its 1:1 child (LatentSegment).
AR_TRACK = "ar"
IMAGE_TRACK = "image"


def deep_hydrate(obj: Any) -> Any:
    """Materialize every ``TensorMeta`` leaf in ``obj`` to a real tensor, in place.

    The anchored single-actor engines return each track as ONE transport handle
    (a single ref spanning all samples), but the train side is num_devices-way DP and
    slices each track into per-rank shards — a single ref can't be intra-handle
    sliced. Hydrating on the driver fixes the mismatch (the DP dispatch then
    re-shards real tensors), but the driver has no ``TensorTransportRuntime``
    installed, so ``TensorTransport.hydrate`` / ``TensorMeta.local`` are
    unavailable here. ``_hydrate_tensor_meta`` instead pulls each leaf through
    its ref's ``.local()`` (a plain ``ray.get`` from the owning worker's store),
    which works from the driver — we walk the nested Batch/dict/list/TUPLE
    structure and apply it to every ``TensorMeta``.

    NB: this walks TUPLES too (rebuilding them), unlike ``_collect_leaves``
    which skips them. HunyuanImage3's fused condition stores ``rope_cache`` as a
    ``tuple`` of two TensorMeta; the DP scatter's driver-side
    ``RolloutTrack.concat`` pads that rope (``conditions.concat`` → ``_pad_seq``
    → ``t.ndim``), so the rope MUST be real tensors here. (dp=1 never concats on
    the driver, so it never tripped on this.)
    """
    if isinstance(obj, TensorMeta):
        return _hydrate_tensor_meta(obj)
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
        ar_rollout_cfg: DictConfig,
        dit_rollout_cfg: DictConfig,
        reward_cfg: DictConfig,
        ar_algorithm_cfg: DictConfig,
        image_algorithm_cfg: DictConfig,
        stack_cfg: DictConfig,
        data_source_cfg: DictConfig,
        sampling_cfg: DictConfig,
        sync_cfg: Optional[DictConfig] = None,
        dump_dir: Optional[str] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = True,
    ) -> None:
        super().__init__(cfg=cfg)
        self.batch_size = batch_size
        # Colocate memory dance: offload the FSDP train state (base + grads +
        # optimizer) to CPU during rollout so the awake engines fit, onload
        # before the train backward. HI3's ~150GB base needs this → default True.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)

        # W&B reporting config (the top-level ``logging`` block). Logger is
        # lazily created in ``train`` on the driver (rank 0); None / disabled
        # block → no-op. Secrets (key, entity) come from env, never the conf.
        self.logging_cfg = logging_cfg
        self.wandb_logger = None
        # Advances once per optimizer step (one per rollout today; per PPO
        # inner update if num_updates_per_batch grows) — the train/ panel axis.
        self._global_optimizer_step = 0

        # Intrusive debug dump: per rollout, write original prompt + AR output
        # text (= the think/recaption that conditions DiT) + decoded images +
        # rewards under ``dump_dir/rollout_<id>/``. None disables. Best-effort —
        # never breaks training (see :meth:`_dump_rollout`).
        self.dump_dir = str(dump_dir) if dump_dir else None
        self._dump_rollout_id = 0
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        # Driver-side data iterator (not a Remote).
        self.data_source = instantiate(data_source_cfg)

        self.sampling_params: BaseSamplingParams = instantiate(sampling_cfg)

        # Set below from the `sync` block; None means no sync (e.g. trainside).
        self.weight_sync = None

        # Single shared slab: train backbone + both algorithms + rollout +
        # reward are siblings on one Worker (colocate; mirrors DiffusionTrainer's
        # non-separate branch).
        with placement(self.pool, fraction=1.0, shared_workers=True):
            self.bundle = remote_hydra(bundle_cfg)
            self.pipeline = remote_hydra(pipeline_cfg, bundle=self.bundle)
            self.backend = remote_hydra(backend_cfg, bundle=self.bundle)
            self.reward = remote_hydra(reward_cfg)

            # Two algorithms over the SAME shared pipeline (each resolves its
            # own stage via ``stage_attr``: ar→pipeline.ar, image→pipeline.diffusion).
            self.ar_algorithm = remote_hydra(ar_algorithm_cfg, pipeline=self.pipeline)
            self.image_algorithm = remote_hydra(image_algorithm_cfg, pipeline=self.pipeline)

            # One stack owns the single backend + both algorithms → one step.
            self.stack = remote_hydra(
                stack_cfg,
                fsdp_backend=self.backend,
                ar_algorithm=self.ar_algorithm,
                image_algorithm=self.image_algorithm,
            )

            # COLOCATE MEMORY: offload the ~150GB frozen base to CPU BEFORE
            # booting the engines. Each engine grabs ~70GB (AR) / ~45GB (DiT) on
            # its 4 cards at boot; with the FSDP base still resident (~19GB/card)
            # that overlaps to >78GB and OOMs. With the base on CPU the engines
            # boot on their disjoint cards (AR 0-3, DiT 4-7) with room to spare.
            if self._enable_fsdp_offload:
                self.backend.offload()

            # Two standalone vLLM-Omni engines, each ONE multi-GPU actor anchored
            # on a DISTINCT worker (AR→device 0, DiT→device 4). The anchor is
            # load-bearing: plain remote() binds the engine to the whole
            # fraction=1.0 scope (all 8 devices, shared base worker), so BOTH
            # engines land in the same worker process and their device-env setup
            # collides — vllm-omni's set_stage_devices then remaps DiT's yaml
            # "4,5,6,7" back onto physical 0-3, overlapping AR → OOM. Anchoring on
            # separate workers keeps them in separate processes: each pops
            # CUDA_VISIBLE_DEVICES and its stage YAML's runtime.devices pins the
            # TP group to disjoint physical cards (AR 0-3, DiT 4-7) — the layout
            # boot smoke gotcha C verified. Colocate-safe because the train base is
            # offloaded during rollout and the engines sleep during train (the
            # memory dance in train_step time-shares the cards — so this is NOT
            # the boot-smoke landmine of engine+FSDP residing simultaneously).
            # DP over engine REPLICAS, one (AR, DiT) pair per node. dp = nodes
            # (16 devices / 8 per node → dp=2; single node → dp=1, fully
            # backward-compatible: range(1), anchors 0/4 = the original path).
            # Replica r is anchored on node r (DevicePool is node-aware,
            # node = device_id // devices_per_node): AR host-worker on device
            # r*8+1, DiT on r*8+4; each engine still spans cards r*8..r*8+3 /
            # r*8+4..r*8+7 via its stage YAML. AR is +1 (not r*8) to keep its host
            # worker off the train rank-0 worker (device 0) — see the push
            # self-deadlock note at the _wire_engine call below.
            per_node = self.pool.devices_per_node
            # Each replica pins ONE (AR 0-3, DiT 4-7) engine pair to a single
            # node, anchored at base+1 / base+4 with base = r*per_node. That
            # layout needs >= 8 cards on the node; with fewer, base+4 spills onto
            # the next node and silently splits the pair cross-node. Fail loud.
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
                # SERIALIZE engine boot: build one engine, then immediately
                # .sleep() it before building the next. Every @distributed Handle
                # call is synchronous (ray.get) and the heavy boot is Omni(...) in
                # the engine's __init__, so .sleep() blocks until THIS engine has
                # finished booting. Booting all dp*2 engines concurrently deadlocks
                # in the DiT warmup's kv_transfer_manager handshake (the 4-way-boot
                # blocker), so the per-engine quiesce is load-bearing — and it also
                # leaves every engine asleep, the steady state train_step expects.
                # AR anchor is base+1, NOT base: weight_sync rank 0 lives on the
                # train DP rank-0 worker = global device 0. If the AR engine were
                # anchored there too (base==0 for replica 0), it shares that one
                # worker PROCESS, and RemoteLoraWeightSync.push() — which runs on
                # rank 0 and does ray.get([... set_lora on the AR engine ...]) —
                # would block-call its own actor (the set_lora task queues behind
                # the in-flight push) → self-deadlock (push never returns, AR
                # set_lora never runs; DiT on device 4 is a separate process so it
                # loads fine). base+1 keeps the AR host worker off device 0 while
                # the engine still uses cards 0-3 via its stage YAML's runtime.devices.
                ar = self._wire_engine(ar_rollout_cfg, anchor_device=base + 1)
                ar.sleep()
                self.ar_rollouts.append(ar)
                dit = self._wire_engine(dit_rollout_cfg, anchor_device=base + 4)
                dit.sleep()
                self.dit_rollouts.append(dit)
            # Back-compat aliases for replica 0 (single-node code paths, dump,
            # debug, and any single-engine references still use these).
            self.ar_rollout = self.ar_rollouts[0]
            self.dit_rollout = self.dit_rollouts[0]

            if sync_cfg is not None:
                # LoRA sync gets ONLY the backend (a same-worker sibling); the
                # engines are cross-slab. RemoteLoraWeightSync.sync() extracts on
                # the train workers and pushes from rank 0 to EACH engine via a
                # plain Ray RPC, so hand it every replica's (role, workers) here.
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

    def _build_req(self, inputs: RolloutInputs, rollout_id: int) -> RolloutReq:
        """Turn a data-source batch of ``P`` prompts into a typed ``RolloutReq``.

        Like :meth:`PETrainer._build_req`, NO pre-expansion: ``train_step`` fans
        out ``P → P*N → P*N*M`` itself (make_root_track / fork_track), and the
        reward expands ``req.primitives`` by ``N*M`` to align prompts to images.
        Pre-expanding here would double-count. The composed sampling params are
        kept whole (the reward reads ``ar.samples_per_prompt * diffusion.``
        ``samples_per_prompt`` to validate the expansion factor); the SDE step
        schedule is resolved off the diffusion sub-block per rollout and stamped
        back onto a per-request copy.
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

    def run_rollout(self, req: RolloutReq) -> RolloutResp:
        """DP rollout: scatter the P prompts across the ``dp`` engine replicas
        (one (AR, DiT) pair per node), run each sub-batch on its replica, then
        ``RolloutTrack.concat`` the per-replica tracks. ``dp<=1`` or ``P<=1``
        falls back to the single-replica path (the original single-node rollout),
        so this is a transparent wrapper when not multinode.

        v1 runs the replicas SEQUENTIALLY — this validates placement + the
        scatter/concat correctness; issuing the per-replica ``generate()`` as Ray
        futures for true concurrent throughput is the follow-up (handoff §8).
        """
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: req.primitives['text'] must be a Texts primitive.")
        prompts = list(texts.texts)
        n = len(prompts)
        if self.dp <= 1 or n <= 1:
            return self._run_rollout_one(self.ar_rollouts[0], self.dit_rollouts[0], req)

        # Contiguous near-equal prompt bounds across the dp replicas.
        bounds = [(n * r) // self.dp for r in range(self.dp + 1)]
        shards: list[RolloutResp] = []
        for r in range(self.dp):
            lo, hi = bounds[r], bounds[r + 1]
            if lo >= hi:
                continue
            # Only "text" is consumed downstream by _run_rollout_one; slice it
            # to this replica's prompt range and rebuild a standalone sub-req.
            sub_req = RolloutReq(
                sample_ids=list(req.sample_ids[lo:hi]),
                group_ids=list(req.group_ids[lo:hi]),
                primitives={"text": Texts(texts=prompts[lo:hi])},
                request_conditions=dict(req.request_conditions),
                sampling_params=req.sampling_params,
                metadata=list(req.metadata[lo:hi]) if req.metadata else [],
            )
            shards.append(self._run_rollout_one(self.ar_rollouts[r], self.dit_rollouts[r], sub_req))
        # Merge per-replica tracks via the default Batch.concat per-field
        # merge; segment rows are 1:1 with track samples, so the AR/image
        # segments stay globally consistent across replicas.
        #
        # CAVEAT — the fused condition's rope_cache is a ``shared_field``
        # (FusedMultimodalCondition), so this concat keeps replica-0's tensor
        # verbatim: the merged condition carries a rope_cache whose batch dim is
        # replica-0's sample count, NOT the global P*N*M. Harmless TODAY because
        # HI3 replay rebuilds rope from gen_image_mask + the real latent shape
        # (diffusion.py ``predict_noise`` [ROPE-FIX]; ar.py likewise) and never
        # reads the track's rope_cache — it only rides along in the KV-propagation
        # kwargs. If a future change makes replay consume ``fused.rope_cache``,
        # dp>1 would SILENTLY feed replica-0 rope to every sample (wrong gradient,
        # no crash, reward unaffected); make rope_cache a tuple-aware CONCAT field
        # before relying on it.
        return RolloutResp(
            tracks={name: RolloutTrack.concat([s.tracks[name] for s in shards]) for name in (AR_TRACK, IMAGE_TRACK)}
        )

    def _run_rollout_one(self, ar_engine: Any, dit_engine: Any, req: RolloutReq) -> RolloutResp:
        """One (AR, DiT) engine pair: PE-style fan-out → 2-track ``RolloutResp`` {"ar","image"}.

        Drives the given ``ar_engine`` / ``dit_engine`` pair (one replica). The
        DP wrapper :meth:`run_rollout` calls this once per replica with that
        node's engines; ``dp=1`` calls it once with replica 0.

        ::

            P prompts ─make_root_track(N)─▶ P*N recaptions  (AR engine, root "ar")
                      ─fork_track(M)──────▶ P*N*M images     (DiT engine, "image")

        Mirrors :meth:`PEPipeline.generate`, but the two 1:1 child generators are
        independent vLLM-Omni engine Remotes sharing one backbone/LoRA — so this
        assembles the lineage explicitly and grafts each engine's
        segment/decoded/conditions onto the lineage shell. The DiT engine reads
        the ORIGINAL prompt (``primitives['text']``) plus the recaption
        (``primitives['cot_text']``); each image's unique ``sample_id`` drives
        ``engine.seed_from_sample_id`` so the M images of a recaption differ.
        """
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError("UnifiedModelTrainer.run_rollout: req.primitives['text'] must be a Texts primitive.")
        prompts = list(texts.texts)

        ar_params = get_ar_params(req.sampling_params)
        diff_params = get_diffusion_params(req.sampling_params)
        n_recaptions = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        n_images = int(diff_params.samples_per_prompt)

        # ── Level 1: P → P*N recaptions. Root "ar" track groups by prompt.
        ar_shell = req.make_root_track(track_name=AR_TRACK, branch=n_recaptions)
        ar_texts = Texts(texts=[t for t in prompts for _ in range(n_recaptions)])
        ar_req = RolloutReq(
            sample_ids=list(ar_shell.sample_ids),
            group_ids=list(ar_shell.parent_ids),
            primitives={"text": ar_texts},
            request_conditions={},
            sampling_params=ar_params,
        )
        ar_resp = ar_engine.generate(ar_req)
        ar_inner = ar_resp.tracks.get(AR_TRACK)
        recaptions = ar_inner.decoded if ar_inner is not None else None
        if not isinstance(recaptions, Texts):
            raise RuntimeError("UnifiedModelTrainer.run_rollout: AR engine returned no decoded Texts on tracks['ar'].")
        if len(recaptions.texts) != len(ar_shell.sample_ids):
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: AR engine returned {len(recaptions.texts)} recaption(s) "
                f"but the AR track expects {len(ar_shell.sample_ids)} (= P*N). The AR engine must be 1:1."
            )
        ar_track = _track_with_field(ar_shell, "segment", ar_inner.segment)
        ar_track = _track_with_field(ar_track, "decoded", recaptions)
        ar_track = _track_with_field(ar_track, "conditions", dict(ar_inner.conditions))

        # ── Level 2: P*N → P*N*M images. Fork "image" from "ar". For AR sample i
        # (0..P*N-1) the original prompt is prompts[i // N] and the recaption is
        # recaptions[i]; replicate each M× for the 1:1 DiT engine.
        img_shell = ar_track.fork_track(parent_name=AR_TRACK, child_name=IMAGE_TRACK, branch=n_images)
        n_ar = len(ar_shell.sample_ids)
        dit_prompts = Texts(texts=[prompts[i // n_recaptions] for i in range(n_ar) for _ in range(n_images)])
        dit_cot = Texts(texts=[recaptions.texts[i] for i in range(n_ar) for _ in range(n_images)])
        # Driver-authoritative x_T RECIPE (per-IMAGE, ROLLOUT-keyed gids). HI3's
        # DiT latent shape is AR-dynamic, so we ship only the recipe (no shape);
        # the worker's prepare_latents hook fills the shape post-AR and regenerates
        # the byte-identical x_T (NoiseRecipe). Keying on (rollout_id, image
        # sample_id) makes x_T per-rollout-VARYING — overriding the engine's
        # seed_from_sample_id, which is keyed on the rollout-STABLE sample_id alone
        # and so reused the SAME x_T every rollout (frozen-noise overfit, the bug
        # this fixes). ``_dump_rollout_id`` is set to the current rollout_id by the
        # train loop just before train_step. Opt out via DISABLE_DRIVER_XT.
        dit_noise_gids = (
            []
            if os.environ.get("DISABLE_DRIVER_XT")
            else [f"r{int(self._dump_rollout_id)}:{sid}" for sid in img_shell.sample_ids]
        )
        dit_req = RolloutReq(
            sample_ids=list(img_shell.sample_ids),
            group_ids=list(img_shell.parent_ids),
            primitives={"text": dit_prompts, "cot_text": dit_cot},
            request_conditions={},
            sampling_params=diff_params,
            init_noise_group_ids=dit_noise_gids,
        )
        dit_resp = dit_engine.generate(dit_req)
        img_inner = dit_resp.tracks.get(IMAGE_TRACK)
        if img_inner is None:
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: DiT engine returned no 'image' track (got {sorted(dit_resp.tracks.keys())})."
            )
        if len(img_inner.sample_ids) != len(img_shell.sample_ids):
            raise RuntimeError(
                f"UnifiedModelTrainer.run_rollout: DiT engine returned {len(img_inner.sample_ids)} image(s) "
                f"but the image track expects {len(img_shell.sample_ids)} (= P*N*M). The DiT engine must be 1:1."
            )
        img_track = _track_with_field(img_shell, "segment", img_inner.segment)
        img_track = _track_with_field(img_track, "decoded", img_inner.decoded)
        img_track = _track_with_field(img_track, "conditions", dict(img_inner.conditions))
        img_track = _track_with_field(img_track, "media_preview", img_inner.media_preview)

        # Each anchored engine returns its track as ONE transport handle (a single
        # ref spanning all P*N / P*N*M samples). The train side is num_devices-way DP and
        # slices each track into per-rank shards — but a single ref can't be
        # intra-handle-sliced ("does not align to ref boundaries"). Materialize
        # the tracks to real tensors on the driver here; the reward / advantage /
        # train DP dispatch then re-shards real tensors. (DiffusionTrainer dodges
        # this because its per-worker DP engine already emits one ref per rank,
        # aligned to the train DP boundaries — our single-actor TP engines don't.)
        deep_hydrate(ar_track)
        deep_hydrate(img_track)

        return RolloutResp(tracks={AR_TRACK: ar_track, IMAGE_TRACK: img_track})

    def train_step(
        self,
        req: RolloutReq,
        *,
        training_progress: float = 0.0,
        sync_weights: bool = False,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)`` — ``mean_reward`` is the
        mean unnormalized image reward (for the log line).
        """
        # Colocate memory dance (150GB base can't coexist with an awake engine on
        # the same card). Steady state on entry: base offloaded, engines asleep.
        #   1. EXTRACT while engines ASLEEP + base ONLOADED — extract() runs a
        #      train-mesh collective whose state_dict() gathers the full FSDP model
        #      to GPU; an awake engine (~85GB) alongside the onloaded base
        #      (~19GB/card) would OOM. extract() caches the adapter on rank 0
        #      (nothing returned); offload the base again right after.
        if sync_weights and self.weight_sync is not None:
            if self._enable_fsdp_offload:
                self.backend.onload()
            self.weight_sync.extract()
            if self._enable_fsdp_offload:
                self.backend.offload()
        #   2. Wake both engines (base on CPU → room; AR 0-3, DiT 4-7 disjoint),
        #      then PUSH the cached adapter from rank 0 into each engine's
        #      set_lora_from_tensors_copy (cross-process; engines are not siblings).
        for _eng in self.ar_rollouts + self.dit_rollouts:
            _eng.wake_up()
        if sync_weights and self.weight_sync is not None:
            self.weight_sync.push()
        #   3. Rollout (base offloaded), then sleep engines and onload the base
        #      for the train backward.
        resp = self.run_rollout(req)
        for _eng in self.ar_rollouts + self.dit_rollouts:
            _eng.sleep()
        if self._enable_fsdp_offload:
            self.backend.onload()

        # 1. Score the IMAGE track only — the AR track's TextSegment is not
        #    directly scorable; its reward is credit-assigned below.
        #    Build a reward req aligned 1:1 with the image track (each image's
        #    ORIGINAL prompt). score_and_attach is DP_SCATTER: it splits the track
        #    across ranks but broadcasts the req, so a P-prompt req leaves each
        #    rank with req(P) > track(P*N*M/dp) → "not an integer multiple". A
        #    1:1 req shards together with the track (req==track per rank).
        img_track = resp.tracks[IMAGE_TRACK]
        ar_params = get_ar_params(req.sampling_params)
        diff_params = get_diffusion_params(req.sampling_params)
        n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        n_img = int(diff_params.samples_per_prompt)
        orig_texts = req.primitives.get("text")
        reward_texts = Texts(texts=[orig_texts.texts[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))])
        reward_req = RolloutReq(
            sample_ids=list(img_track.sample_ids),
            group_ids=list(img_track.parent_ids) if img_track.parent_ids else list(img_track.sample_ids),
            primitives={"text": reward_texts},
            request_conditions={},
            sampling_params=req.sampling_params,
            metadata=[],
        )
        scored = self.reward.score_and_attach(req=reward_req, track=img_track)
        if scored.rewards is not None:
            scored.rewards = _hydrate_tensor_meta(scored.rewards)
        resp.tracks[IMAGE_TRACK] = scored

        # 2. Credit-assign image reward up the lineage → fills the "ar" track.
        resp = resp.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = resp.tracks[IMAGE_TRACK].rewards
        if di_rewards is not None:
            mean_reward = float(_hydrate_tensor_meta(di_rewards).to(torch.float32).mean().item())

        # 3b. Intrusive debug dump (best-effort) — observe what AR generated and
        #     what DiT rendered before advantages/training mutate the tracks.
        if self.dump_dir:
            self._dump_rollout(self._dump_rollout_id, req, resp)

        # 4. Per-track GRPO advantages.
        for name in (AR_TRACK, IMAGE_TRACK):
            resp.tracks[name] = resp.tracks[name].compute_advantages(normalize=True)

        # after the debug dump (which reads decoded), before training
        self._drop_decoded(resp)
        # 5. Two backward (shared backbone) → one optimizer step.
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            resp.tracks[AR_TRACK],
            resp.tracks[IMAGE_TRACK],
            training_progress=float(training_progress),
        )

        # 6. Back to steady state (base on CPU) so the next rollout's engines
        #    have room to wake.
        if self._enable_fsdp_offload:
            self.backend.offload()
        return results, mean_reward

    def _dump_rollout(self, rollout_id: int, req: RolloutReq, resp: Any) -> None:
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

            prompts_obj = req.primitives.get("text")
            prompts = list(prompts_obj.texts) if prompts_obj is not None else []

            ar_track = resp.tracks.get(AR_TRACK)
            ar_decoded = getattr(ar_track, "decoded", None) if ar_track is not None else None
            ar_texts = list(ar_decoded.texts) if ar_decoded is not None else []

            image_track = resp.tracks.get(IMAGE_TRACK)
            img_decoded = getattr(image_track, "decoded", None) if image_track is not None else None
            sample_ids = list(image_track.sample_ids) if image_track is not None else []
            parent_ids = list(image_track.parent_ids) if (image_track is not None and image_track.parent_ids) else []

            rewards = None
            if image_track is not None and image_track.rewards is not None:
                rewards = _hydrate_tensor_meta(image_track.rewards).to(torch.float32).tolist()

            # Save images (best-effort): hydrate pixels and write per-sample PNGs.
            n_imgs = 0
            if img_decoded is not None and getattr(img_decoded, "pixels", None) is not None:
                from torchvision.utils import save_image

                pixels = _hydrate_tensor_meta(img_decoded.pixels).detach().to(torch.float32).clamp(0, 1).cpu()
                n_imgs = int(pixels.shape[0])
                for k in range(n_imgs):
                    save_image(pixels[k], os.path.join(out_dir, f"img_{k}.png"))

            # Two-level lineage: image sample k (0..P*N*M-1) descends from AR
            # sample k // M and original prompt k // (N*M). Index the smaller
            # prompt / recaption lists through those factors.
            ar_params = get_ar_params(self.sampling_params)
            diff_params = get_diffusion_params(self.sampling_params)
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
                                # In think_recaption the AR output IS the text fed
                                # into DiT (the recaption conditions the DiT stage).
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

    def train(self, *, num_rollouts: int, weight_sync_interval: int = 1) -> None:
        """Minimal training loop: ``num_rollouts`` iterations of ``train_step``."""
        interval = max(1, weight_sync_interval)
        self._init_wandb()
        for rollout_id in range(num_rollouts):
            training_progress = rollout_id / max(1, num_rollouts - 1)
            self._dump_rollout_id = rollout_id  # picked up by train_step's dump
            inputs = self.data_source.get_samples(self.batch_size)
            req = self._build_req(inputs, rollout_id)
            # Sync before generate; skip step 0 (nothing trained yet). The
            # HI3_SYNC_FIRST env forces a sync on rollout 0 too — a debug knob to
            # exercise the LoRA-sync path early (cheaply) without a full extra
            # rollout; the rollout-0 adapter is ~0 but that's fine for testing
            # the register→activate mechanism.
            sync_weights = (rollout_id > 0 or os.environ.get("HI3_SYNC_FIRST")) and rollout_id % interval == 0
            results, mean_reward = self.train_step(
                req,
                training_progress=training_progress,
                sync_weights=sync_weights,
            )
            ar, di = results[AR_TRACK], results[IMAGE_TRACK]

            # Step-0 ratio probe (π_old vs π_θ alignment). On rollout 0 the LoRA
            # is ~0 so a correct replay should give ratio≈1, std≈0; a systematic
            # offset means the logp convention (temperature / top-k-p filtering /
            # full-vs-renorm softmax) doesn't match vLLM's sampler — toggle the
            # replay temperature and re-check before trusting the gradient.
            def _m(res, key):
                metrics = getattr(res, "metrics", None) or {}
                v = metrics.get(key)
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")

            logger.info(
                "rollout %d/%d  reward=%.4f  "
                "ar[loss=%.4f ratio=%.4f±%.4f clip=%.2f]  "
                "image[loss=%.4f gn=%.4f lr=%.2e ratio=%.4f±%.4f clip=%.2f]",
                rollout_id + 1,
                num_rollouts,
                mean_reward,
                ar.loss,
                _m(ar, "ratio_mean"),
                _m(ar, "ratio_std"),
                _m(ar, "clip_fraction"),
                di.loss,
                di.grad_norm,
                di.lr,
                _m(di, "ratio_mean"),
                _m(di, "ratio_std"),
                _m(di, "clip_fraction"),
            )

            if self.wandb_logger is not None:
                # Mirror train.py (SD3.5): the per-track training scalars
                # (loss / ratio / clip / grad_norm / lr) go to BOTH panels —
                # ``rollout/*`` keyed by rollout_id (1 point per rollout) and
                # ``train/*`` keyed by global_optimizer_step. With one update
                # per rollout the two axes coincide; under PPO multi-epoch the
                # train/ axis advances per optimizer step and shows the
                # intra-rollout ratio drift. Reward stats stay rollout-only.
                training_metrics = {
                    "ar/loss": ar.loss,
                    "ar/ratio_mean": _m(ar, "ratio_mean"),
                    "ar/ratio_std": _m(ar, "ratio_std"),
                    "ar/clip_fraction": _m(ar, "clip_fraction"),
                    "image/loss": di.loss,
                    "image/grad_norm": di.grad_norm,
                    "image/lr": di.lr,
                    "image/ratio_mean": _m(di, "ratio_mean"),
                    "image/ratio_std": _m(di, "ratio_std"),
                    "image/clip_fraction": _m(di, "clip_fraction"),
                }
                self.wandb_logger.log_rollout(
                    rollout_id,
                    {
                        **training_metrics,
                        "reward_mean": mean_reward,
                        "sync_weights": float(bool(sync_weights)),
                    },
                )
                self._global_optimizer_step += 1
                self.wandb_logger.log_step(self._global_optimizer_step, training_metrics)

        if self.wandb_logger is not None:
            self.wandb_logger.finish()

    def _init_wandb(self) -> None:
        """Create the driver-side W&B logger from the ``logging`` block.

        No-op when the block is absent or ``report_to_wandb`` is off. Mirrors
        ``train.py``: secrets (API key, entity) come from the environment
        (``WANDB_API_KEY`` / ``WANDB_ENTITY``), never the committed conf.
        """
        cfg = self.logging_cfg
        if cfg is None or not cfg.get("report_to_wandb") or not cfg.get("project_name"):
            return
        from unirl.utils.wandb_logger import init_logger

        raw_tags = cfg.get("tags")
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif raw_tags:
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags = None
        self.wandb_logger = init_logger(
            project=str(cfg.get("project_name")),
            run_name=cfg.get("run_name"),
            config=None,
            log_dir=cfg.get("logging_dir"),
            rank=0,
            tags=tags,
            entity=cfg.get("entity") or os.environ.get("WANDB_ENTITY") or None,
            require_success=True,
        )
        if self.wandb_logger.initialized:
            logger.info(
                "WandB initialized: project=%s run=%s",
                cfg.get("project_name"),
                cfg.get("run_name"),
            )


__all__ = ["UnifiedModelTrainer"]
