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
import inspect
import json
import logging
import os
import sys
import time
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from hydra.utils import get_object, instantiate
from omegaconf import DictConfig

from unirl.distributed.group.placement import placement, remote
from unirl.distributed.tensor import TensorRef, hydrate
from unirl.distributed.tensor.batch import Batch
from unirl.models.bagel.conditions import BagelT2TIDiffusionConditions
from unirl.train.stack import TrainStepResult
from unirl.trainer.base import BaseTrainer, build_sampling_dict
from unirl.trainer.eval_suites import build_eval_suites
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack, _track_with_field
from unirl.types.sampling import BaseSamplingParams
from unirl.utils.hydra import parse_hydra_cfg, remote_hydra

logger = logging.getLogger(__name__)

# Track names produced by the vLLM-Omni HI3 rollout (see
# ``rollout/engine/vllm_omni/response.py``): "ar" is the root (TextSegment,
# groups by prompt), "image" is its 1:1 child (LatentSegment).
AR_TRACK = "ar"
IMAGE_TRACK = "image"


def _bagel_t2ti_replay_permutation(
    track: RolloutTrack,
    *,
    num_shards: int,
    num_updates: int,
) -> Optional[torch.Tensor]:
    """Plan equal-size DP shards with aligned exact-replay depths.

    Each optimizer update keeps exactly the samples it owned before the
    permutation. Within that membership, adjacent replay depths are distributed
    one per rank so the per-position MAX used by collective padding stays close
    to real work on every rank.
    """
    num_shards = int(num_shards)
    num_updates = int(num_updates)
    total = int(track.batch_size)
    if num_shards <= 1 or num_updates <= 0 or total == 0 or total % num_shards != 0:
        return None
    shard_size = total // num_shards
    if shard_size % num_updates != 0:
        return None

    try:
        conditions = BagelT2TIDiffusionConditions.from_dict(track.conditions)
    except ValueError:
        return None
    if conditions.batch_size != total:
        return None

    update_size = shard_size // num_updates
    replay_depths = [len(spec.chunk_offsets) - 1 for spec in conditions.replay_specs]
    planned: List[List[List[int]]] = [[[] for _ in range(num_updates)] for _ in range(num_shards)]

    for update in range(num_updates):
        members = [
            rank * shard_size + update * update_size + offset
            for rank in range(num_shards)
            for offset in range(update_size)
        ]
        members.sort(key=lambda index: (-replay_depths[index], index))
        loads = [0 for _ in range(num_shards)]
        for start in range(0, len(members), num_shards):
            bucket = members[start : start + num_shards]
            ranks = sorted(range(num_shards), key=lambda rank: (loads[rank], rank))
            for rank, index in zip(ranks, bucket):
                planned[rank][update].append(index)
                loads[rank] += replay_depths[index]

    permutation = [
        index for rank in range(num_shards) for update in range(num_updates) for index in planned[rank][update]
    ]
    if len(permutation) != total or len(set(permutation)) != total:
        raise RuntimeError("BAGEL T2TI replay planner did not produce a complete permutation.")
    return torch.tensor(permutation, dtype=torch.long)


def _bagel_t2ti_replay_depth_metrics(depths: List[int]) -> Dict[str, float]:
    """Summarize native Stage-0 scheduler depth without retaining a histogram."""
    if not depths:
        return {}
    ordered = sorted(int(depth) for depth in depths)

    def percentile(fraction: float) -> float:
        index = round((len(ordered) - 1) * float(fraction))
        return float(ordered[index])

    return {
        "bagel_t2ti_replay_depth_min": float(ordered[0]),
        "bagel_t2ti_replay_depth_mean": float(sum(ordered) / len(ordered)),
        "bagel_t2ti_replay_depth_p50": percentile(0.50),
        "bagel_t2ti_replay_depth_p90": percentile(0.90),
        "bagel_t2ti_replay_depth_p99": percentile(0.99),
        "bagel_t2ti_replay_depth_max": float(ordered[-1]),
    }


def _reduce_rollout_boundary_metrics(reports: Any) -> Dict[str, float]:
    """Reduce per-DP-worker optimizer-boundary telemetry on the driver.

    ``Dispatch.BROADCAST`` returns one mapping per worker. Byte counters are
    physical per-rank allocations and therefore sum across the DP slab; host
    times use the slowest worker because handle dispatch is a barrier.
    """
    worker_reports: List[Mapping[str, Any]] = []

    def collect(value: Any) -> None:
        if isinstance(value, Mapping):
            worker_reports.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    collect(reports)
    keys = {str(key) for report in worker_reports for key in report}
    reduced: Dict[str, float] = {}
    for key in keys:
        values = [float(report[key]) for report in worker_reports if report.get(key) is not None]
        if not values:
            continue
        reduced[key] = max(values) if key.endswith("_host_time_s") else sum(values)
    return reduced


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

    NB: this walks TUPLES too (rebuilding them), unlike ``_collect_leaves``
    which skips them. HunyuanImage3's fused condition stores ``rope_cache`` as a
    ``tuple`` of two TensorRef; the DP scatter's driver-side
    ``RolloutTrack.concat`` pads that rope (``conditions.concat`` → ``_pad_seq``
    → ``t.ndim``), so the rope MUST be real tensors here. (dp=1 never concats on
    the driver, so it never tripped on this.)
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
        ar_rollout_cfg: Optional[DictConfig] = None,
        dit_rollout_cfg: Optional[DictConfig] = None,
        rollout_cfg: Optional[DictConfig] = None,
        sync_cfg: Optional[DictConfig] = None,
        dump_dir: Optional[str] = None,
        logging_cfg: Optional[DictConfig] = None,
        enable_fsdp_offload: bool = True,
        park_optimizer_state_during_rollout: bool = False,
        eval_interval: int = 0,
        eval_num_prompts: int = 32,
        eval_cfg_text_scale: float = 4.0,
        eval_eta: float = 0.0,
        eval_rewards_cfg: Optional[Any] = None,
    ) -> None:
        super().__init__(cfg=cfg, logging_cfg=logging_cfg)
        self.batch_size = batch_size
        self._train_dp_size = int(self.pool.num_devices)
        self._num_updates_per_batch = int(stack_cfg.get("num_updates_per_batch", 1))
        # Colocate memory dance: offload the FSDP train state (base + grads +
        # optimizer) to CPU during rollout so the awake engines fit, onload
        # before the train backward. HI3's ~150GB base needs this → default True.
        self._enable_fsdp_offload = bool(enable_fsdp_offload)
        self._park_optimizer_state_during_rollout = bool(park_optimizer_state_during_rollout)
        fsdp_cfg = backend_cfg.get("fsdp_cfg")
        self._backend_persistent_cpu_offload = bool(fsdp_cfg is not None and fsdp_cfg.get("cpu_offload", False))

        # Periodic eval on the eval set (run.eval_data_path), logged under eval/*;
        # eval_interval=0 disables it. Scores only the image track, generated at
        # the deterministic best-quality setting (CFG=eval_cfg_text_scale,
        # eta=eval_eta) — same knobs/semantics as DiffusionTrainer; extra
        # eval-only rewards: unirl.trainer.eval_suites. No eval_samples_per_prompt
        # knob: the 2-track fan-out makes a single count ambiguous (bagel is M=1).
        self.eval_interval = int(eval_interval)
        self.eval_num_prompts = int(eval_num_prompts)
        self.eval_cfg_text_scale = float(eval_cfg_text_scale)
        self.eval_eta = float(eval_eta)

        # W&B logging (logging_cfg, wandb_logger, optimizer-step counter) is owned
        # by BaseTrainer + UniRLWandBLogger now — see super().__init__ above.

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

        self.sampling_params: Dict[str, BaseSamplingParams] = build_sampling_dict(sampling_cfg)
        rollout_engine_cfg = rollout_cfg.get("config") if rollout_cfg is not None else None
        self._strict_bagel_t2ti = bool(
            rollout_engine_cfg is not None and rollout_engine_cfg.get("modality") == "bagel_t2ti"
        )
        if self._strict_bagel_t2ti:
            self._validate_bagel_t2ti_contract(self.sampling_params, sync_cfg)
        # Driver-authored x_T recipe for native BAGEL T2TI. Keep one base key
        # per source prompt; the adapter either shares it across N thoughts or
        # expands it to /aN/i0 according to init_same_noise.
        self._noise_latent_shape: Optional[List[int]] = (
            None
            if os.environ.get("DISABLE_DRIVER_XT")
            else self._resolve_noise_latent_shape(
                pipeline_cfg=pipeline_cfg,
                model_cfg=bundle_cfg,
            )
        )

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
            # Extra eval-only rewards (eval_rewards) — siblings of the training
            # reward on this slab; see unirl.trainer.eval_suites.
            self._eval_suites = build_eval_suites(
                eval_rewards_cfg, data_source_cfg=data_source_cfg, enabled=self.eval_interval > 0
            )

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

            # Rollout wiring. Single-engine (M=1 / UniGRPO — a trainside or single
            # engine on the SHARED pipeline) short-circuits the two-engine HI3 path
            # below: no GPU partition, no weight sync, and (trainside) no base
            # offload since it samples the live FSDP modules. ``_shared_advantage``
            # makes train_step copy the AR's prompt-level advantage onto the 1:1
            # image track (M=1) instead of the degenerate per-rewrite grouping.
            self._single_engine = rollout_cfg is not None
            self._shared_advantage = self._single_engine
            self._rollout_is_trainside = False
            self._single_engine_staged_sync = False
            if self._single_engine:
                self.dp = 1
                self.ar_rollouts = []
                self.dit_rollouts = []
                self.ar_rollout = None
                self.dit_rollout = None
                rollout_parsed = parse_hydra_cfg(rollout_cfg)
                self._rollout_is_trainside = "pipeline" in inspect.signature(rollout_parsed["role_cls"]).parameters
                self._validate_optimizer_state_parking_contract(
                    enabled=self._park_optimizer_state_during_rollout,
                    single_engine=True,
                    rollout_is_trainside=self._rollout_is_trainside,
                    enable_fsdp_offload=self._enable_fsdp_offload,
                    backend_persistent_cpu_offload=self._backend_persistent_cpu_offload,
                )
                if self._rollout_is_trainside:
                    self.rollout = remote(**rollout_parsed, pipeline=self.pipeline)
                    self._enable_fsdp_offload = False  # shares live FSDP modules
                else:
                    sync_role_cls = None
                    if sync_cfg is not None:
                        sync_role_cls = parse_hydra_cfg(sync_cfg)["role_cls"]
                        self._single_engine_staged_sync = all(
                            callable(getattr(sync_role_cls, method, None)) for method in ("extract", "push", "discard")
                        )
                        if self._enable_fsdp_offload and not self._single_engine_staged_sync:
                            raise ValueError(
                                "UnifiedModelTrainer: an offloaded external single engine "
                                "requires a staged weight sync exposing extract(), push(), "
                                "and discard(); use CPUStagedFullWeightSync. A one-shot "
                                "sync would require the trainer and awake engine to coexist."
                            )
                        needs_cpu_policy = bool(
                            getattr(
                                sync_role_cls,
                                "requires_persistent_cpu_offload_on_single_device",
                                False,
                            )
                        )
                        if needs_cpu_policy and self.pool.num_devices == 1 and not self._backend_persistent_cpu_offload:
                            raise ValueError(
                                "UnifiedModelTrainer: single-device staged full fine-tuning "
                                "requires backend.fsdp_cfg.cpu_offload=true. Coarse engine/FSDP "
                                "time-sharing alone is insufficient because fp32 masters and "
                                "Adam state exceed one 80GB GPU."
                            )
                    # External single-engine mode time-shares one GPU with the
                    # trainer. Boot only after FSDP has left the device, then
                    # immediately sleep the engine to establish the steady state
                    # expected by train_step/evaluate.
                    if self._enable_fsdp_offload:
                        self.backend.prepare_for_rollout()
                    self.rollout = remote(**rollout_parsed)
                    try:
                        self.rollout.sleep()
                        if sync_cfg is not None:
                            self.weight_sync = remote_hydra(
                                sync_cfg,
                                backend=self.backend,
                                rollout=self.rollout,
                            )
                    except BaseException:
                        # Construction has no enclosing rollout session yet. Do
                        # an explicit best-effort release so a failed initial
                        # sleep or sync constructor cannot orphan Omni pools.
                        try:
                            self.rollout.sleep()
                        except BaseException as exc:
                            logger.exception("Initial Omni cleanup sleep failed: %s", exc)
                        try:
                            self.rollout.shutdown()
                        except BaseException as exc:
                            logger.exception("Initial Omni shutdown failed: %s", exc)
                        raise
                return

            self._validate_optimizer_state_parking_contract(
                enabled=self._park_optimizer_state_during_rollout,
                single_engine=False,
                rollout_is_trainside=False,
                enable_fsdp_offload=self._enable_fsdp_offload,
                backend_persistent_cpu_offload=self._backend_persistent_cpu_offload,
            )
            if ar_rollout_cfg is None or dit_rollout_cfg is None:
                raise ValueError(
                    "UnifiedModelTrainer: two-engine mode needs ar_rollout_cfg + dit_rollout_cfg; "
                    "pass a single rollout_cfg for single-engine (M=1 / UniGRPO) mode."
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

    @staticmethod
    def _validate_bagel_t2ti_contract(
        sampling_params: Dict[str, BaseSamplingParams],
        sync_cfg: Optional[DictConfig],
    ) -> None:
        """Reject BAGEL settings that cannot represent on-policy P*N*1 training."""
        ar_params = sampling_params.get("ar")
        diff_params = sampling_params.get("diffusion")
        n_thoughts = int(getattr(ar_params, "samples_per_prompt", 0))
        n_images = int(getattr(diff_params, "samples_per_prompt", 0))
        if n_thoughts < 2:
            raise ValueError(
                "UnifiedModelTrainer: bagel_t2ti requires ar.samples_per_prompt >= 2; "
                "N=1 produces zero prompt-group advantages."
            )
        if n_images != 1:
            raise ValueError(
                f"UnifiedModelTrainer: bagel_t2ti requires diffusion.samples_per_prompt == 1; got {n_images}."
            )
        if sync_cfg is None:
            raise ValueError(
                "UnifiedModelTrainer: external bagel_t2ti requires strict full-weight sync to both Omni stages."
            )
        sync_target = str(sync_cfg.get("_target_", ""))
        if not sync_target.endswith("CPUStagedFullWeightSync"):
            raise ValueError(
                "UnifiedModelTrainer: bagel_t2ti requires CPUStagedFullWeightSync; "
                f"got {sync_target or '<missing target>'}."
            )
        if sync_cfg.get("load_plan") != "bagel_vllm_omni_0_20":
            raise ValueError("UnifiedModelTrainer: bagel_t2ti requires sync.load_plan='bagel_vllm_omni_0_20'.")
        stage_ids = {int(stage_id) for stage_id in sync_cfg.get("stage_ids", ())}
        if stage_ids != {0, 1}:
            raise ValueError(
                "UnifiedModelTrainer: bagel_t2ti requires sync.stage_ids to contain exactly {0, 1}; "
                f"got {sorted(stage_ids)}."
            )

    @staticmethod
    def _validate_optimizer_state_parking_contract(
        *,
        enabled: bool,
        single_engine: bool,
        rollout_is_trainside: bool,
        enable_fsdp_offload: bool,
        backend_persistent_cpu_offload: bool,
    ) -> None:
        """Keep optimizer-only parking distinct from either FSDP offload mode."""
        if not enabled:
            return
        if not single_engine or rollout_is_trainside:
            raise ValueError(
                "UnifiedModelTrainer: park_optimizer_state_during_rollout is only valid "
                "for an external single-engine rollout."
            )
        if enable_fsdp_offload:
            raise ValueError(
                "UnifiedModelTrainer: optimizer-state parking requires enable_fsdp_offload=false; "
                "the feature keeps FSDP parameters and shards on GPU."
            )
        if backend_persistent_cpu_offload:
            raise ValueError(
                "UnifiedModelTrainer: optimizer-state parking requires backend.fsdp_cfg.cpu_offload=false."
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

    def _resolve_noise_latent_shape(
        self,
        *,
        pipeline_cfg: DictConfig,
        model_cfg: DictConfig,
    ) -> Optional[List[int]]:
        """Resolve the pipeline-owned packed x_T geometry without instantiation."""
        target = getattr(pipeline_cfg, "_target_", None)
        if not isinstance(target, str):
            return None
        resolved = get_object(target)
        pipeline_cls = resolved if isinstance(resolved, type) else getattr(resolved, "__self__", None)
        latent_shape = getattr(pipeline_cls, "latent_shape", None)
        if latent_shape is None:
            return None
        try:
            shape = latent_shape(
                model_config=model_cfg,
                sampling_spec=self.sampling_params.get("diffusion"),
            )
        except NotImplementedError:
            return None
        return [int(value) for value in shape]

    @contextmanager
    def _external_single_engine_session(
        self,
        *,
        sync_weights: bool,
        onload_trainer_after: bool,
    ):
        """Wake one external engine under an exception-safe memory lifecycle.

        Entry and exit steady state is engine-asleep. For the legacy full-state
        lifecycle, training exit also onloads FSDP for replay/backward while
        eval leaves it offloaded. The optimizer-only lifecycle instead leaves
        FSDP parameters/shards resident, parks completed grads + Adam tensors
        before wake, and restores Adam only after every stage sleeps.
        """
        if not self._single_engine or self._rollout_is_trainside:
            raise RuntimeError("_external_single_engine_session is only valid for an external single engine.")

        do_sync = bool(sync_weights and self.weight_sync is not None)
        staged = bool(do_sync and self._single_engine_staged_sync)
        park_optimizer = bool(getattr(self, "_park_optimizer_state_during_rollout", False))
        optimizer_park_attempted = False
        boundary_metrics: Dict[str, float] = {}
        cleanup_errors: List[Tuple[str, BaseException]] = []
        try:
            if staged:
                if self._enable_fsdp_offload:
                    self.backend.prepare_for_compute()
                self.weight_sync.extract()
                if self._enable_fsdp_offload:
                    self.backend.prepare_for_rollout()

            if park_optimizer:
                # Extract first while the engine sleeps, then release only the
                # post-step gradient + optimizer footprint. FSDP parameters and
                # shards remain on the compute device for the whole session.
                optimizer_park_attempted = True
                boundary_metrics.update(
                    _reduce_rollout_boundary_metrics(self.backend.park_optimizer_state_for_rollout())
                )

            # Keep wake, push, and every generate call inside this try. Even a
            # partially failed wake is followed by sleep in the finally block.
            self.rollout.wake_up()
            if do_sync:
                if staged:
                    self.weight_sync.push()
                else:
                    self.weight_sync.sync()
            yield boundary_metrics
        finally:
            active_error = sys.exc_info()[0] is not None
            rollout_asleep = False
            try:
                self.rollout.sleep()
                rollout_asleep = True
            except BaseException as first_exc:
                # Backend stage tracking retains only failed/possibly-awake
                # stages, so one bounded retry completes transient ACK failures
                # without re-sleeping stages that already succeeded.
                try:
                    self.rollout.sleep()
                    rollout_asleep = True
                    logger.warning("Recovered from initial rollout.sleep failure: %s", first_exc)
                except BaseException as retry_exc:  # preserve the primary rollout failure
                    cleanup_errors.append(("rollout.sleep", first_exc))
                    cleanup_errors.append(("rollout.sleep retry", retry_exc))
                    try:
                        self.rollout.shutdown()
                    except BaseException as shutdown_exc:
                        cleanup_errors.append(("rollout.shutdown", shutdown_exc))

            if staged:
                try:
                    # Idempotent after a successful push; essential if extract,
                    # wake, or push failed with a CPU snapshot still pending.
                    self.weight_sync.discard()
                except BaseException as exc:
                    cleanup_errors.append(("weight_sync.discard", exc))

            # Restoring Adam while any Omni stage may still be awake recreates
            # the exact overlap this lifecycle prevents. A failed park is still
            # restored here: the backend operation is intentionally idempotent
            # and repairs a partially moved optimizer state after safe sleep.
            if optimizer_park_attempted and rollout_asleep:
                try:
                    boundary_metrics.update(
                        _reduce_rollout_boundary_metrics(self.backend.restore_optimizer_state_after_rollout())
                    )
                except BaseException as exc:
                    cleanup_errors.append(("optimizer state restore", exc))

            # Never move trainer state back onto the GPU unless every Omni stage
            # acknowledged sleep. A partial sleep plus FSDP onload can OOM before
            # the original lifecycle error reaches the caller.
            if self._enable_fsdp_offload and rollout_asleep:
                try:
                    if onload_trainer_after:
                        self.backend.prepare_for_compute()
                    else:
                        self.backend.prepare_for_rollout()
                except BaseException as exc:
                    cleanup_errors.append(("backend lifecycle restore", exc))

            if cleanup_errors:
                if active_error:
                    for action, exc in cleanup_errors:
                        logger.exception(
                            "External single-engine cleanup failed in %s: %s",
                            action,
                            exc,
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
                else:
                    action, exc = cleanup_errors[0]
                    raise RuntimeError(f"External single-engine cleanup failed in {action}.") from exc

            if boundary_metrics:
                logger.info(
                    "External rollout optimizer boundary: cleared=%.3f GiB parked=%.3f GiB "
                    "restored=%.3f GiB park=%.3fs restore=%.3fs",
                    boundary_metrics.get("grad_bytes_cleared", 0.0) / 2**30,
                    boundary_metrics.get("optimizer_state_bytes_parked", 0.0) / 2**30,
                    boundary_metrics.get("optimizer_state_bytes_restored", 0.0) / 2**30,
                    boundary_metrics.get("optimizer_park_host_time_s", 0.0),
                    boundary_metrics.get("optimizer_restore_host_time_s", 0.0),
                )

    def _build_req(
        self, inputs: RolloutInputs, rollout_id: int, *, base_sampling: Optional[Dict[str, BaseSamplingParams]] = None
    ) -> RolloutReq:
        """Turn a data-source batch of ``P`` prompts into a typed ``RolloutReq``.

        Like :meth:`PETrainer._build_req`, NO pre-expansion: ``train_step`` fans
        out ``P → P*N → P*N*M`` itself (make_root_track / fork_track), and the
        reward expands ``req.primitives`` by ``N*M`` to align prompts to images.
        Pre-expanding here would double-count. The composed sampling params are
        kept whole (the reward reads ``ar.samples_per_prompt * diffusion.``
        ``samples_per_prompt`` to validate the expansion factor); the SDE step
        schedule is resolved off the diffusion sub-block per rollout and stamped
        back onto a per-request copy.

        ``base_sampling`` overrides the modality-keyed sampling dict (``evaluate``
        passes its own deterministic params); ``None`` uses ``self.sampling_params``.
        """
        base = base_sampling if base_sampling is not None else self.sampling_params
        diff_params = base.get("diffusion")
        sde_indices = diff_params.resolve_sde_indices(rollout_id)
        diffusion = dataclasses.replace(diff_params, sde_indices=sde_indices, scheduler=None)
        sampling_params = {**base, "diffusion": diffusion}
        init_noise_group_ids: List[str] = []
        if self._noise_latent_shape is not None:
            source_ids = inputs.group_ids if bool(getattr(diff_params, "init_same_noise", False)) else inputs.sample_ids
            init_noise_group_ids = [f"r{rollout_id}:{source_id}" for source_id in source_ids]
        return RolloutReq(
            sample_ids=list(inputs.sample_ids),
            group_ids=list(inputs.group_ids),
            primitives=dict(inputs.primitives),
            request_conditions={},
            stage_config={"rollout_id": int(rollout_id)},
            sampling_params=sampling_params,
            metadata=list(inputs.metadata) if inputs.metadata else [],
            init_noise_group_ids=init_noise_group_ids,
            init_noise_latent_shape=self._noise_latent_shape,
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
        # Single-engine (M=1 / UniGRPO): the shared pipeline returns the 2-track
        # {"ar","image"} resp directly (DP_SCATTER-sharded like the train stack —
        # no anchored-engine single-handle hydration needed).
        if self._single_engine:
            return self.rollout.generate(req)
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

        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        n_recaptions = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        n_images = int(diff_params.samples_per_prompt)

        # ── Level 1: P → P*N recaptions. Root "ar" track groups by prompt.
        ar_shell = req.make_root_track(track_name=AR_TRACK, branch=n_recaptions)
        ar_texts = Texts(texts=[t for t in prompts for _ in range(n_recaptions)])
        # Ship the WHOLE composed params: the hi3_ar_recaption adapter reads its AR
        # slice for sampling AND the diffusion slice's height/width for the
        # recaption prompt (the engine keeps no sampling defaults).
        ar_req = RolloutReq(
            sample_ids=list(ar_shell.sample_ids),
            group_ids=list(ar_shell.parent_ids),
            primitives={"text": ar_texts},
            request_conditions={},
            sampling_params=req.sampling_params,
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
            sampling_params={"diffusion": diff_params},
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
        rollout_id: int = 0,
    ) -> Tuple[Dict[str, TrainStepResult], float]:
        """One ``rollout → reward → credit-assign → advantage → step`` pass.

        Returns ``(per_track_results, mean_reward)`` — ``mean_reward`` is the
        mean unnormalized image reward (for the log line). ``rollout_id`` keys
        the wandb panels (see :meth:`UniRLWandBLogger.log_rollout_step`).
        """
        t0 = time.perf_counter()
        rollout_boundary_metrics: Dict[str, float] = {}
        if self._single_engine and self._rollout_is_trainside:
            # Trainside (M=1): rollout shares the live FSDP modules, so there is
            # no engine lifecycle or weight transfer.
            resp = self.run_rollout(req)
        elif self._single_engine:
            # External single-engine (BAGEL T2TI): entry is FSDP-offloaded and
            # engine-asleep. A staged sync snapshots while asleep, then the
            # session wakes/pushes/generates and ALWAYS sleeps/cleans up before
            # onloading FSDP for replay and backward.
            with self._external_single_engine_session(
                sync_weights=sync_weights,
                onload_trainer_after=True,
            ) as rollout_boundary_metrics:
                resp = self.run_rollout(req)
        else:
            # Colocate memory dance (150GB base can't coexist with an awake engine
            # on the same card). Steady state on entry: base offloaded, engines
            # asleep. EXTRACT (base onloaded) -> wake engines -> PUSH adapter ->
            # rollout (base offloaded) -> sleep engines -> onload base for backward.
            if sync_weights and self.weight_sync is not None:
                if self._enable_fsdp_offload:
                    self.backend.onload()
                self.weight_sync.extract()
                if self._enable_fsdp_offload:
                    self.backend.offload()
            for _eng in self.ar_rollouts + self.dit_rollouts:
                _eng.wake_up()
            if sync_weights and self.weight_sync is not None:
                self.weight_sync.push()
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
        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        n_img = int(diff_params.samples_per_prompt)
        orig_texts = req.primitives.get("text")
        reward_texts = Texts(texts=[orig_texts.texts[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))])
        # Per-sample metadata, expanded 1:1 with the image track by the SAME
        # prompt-index map as reward_texts. GenEval-style rewards read each
        # image's compositional spec (tag/include/exclude) from metadata[i];
        # without this expansion the geneval scorer raises (no per-item spec).
        # Empty when the data source carries no metadata (e.g. pickscore prompts).
        reward_metadata = (
            [req.metadata[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))] if req.metadata else []
        )
        reward_req = RolloutReq(
            sample_ids=list(img_track.sample_ids),
            group_ids=list(img_track.parent_ids) if img_track.parent_ids else list(img_track.sample_ids),
            primitives={"text": reward_texts},
            request_conditions={},
            sampling_params=req.sampling_params,
            metadata=reward_metadata,
        )
        scored = self.reward.score_and_attach(req=reward_req, track=img_track)
        if scored.rewards is not None:
            scored.rewards = hydrate(scored.rewards)
        resp.tracks[IMAGE_TRACK] = scored

        # 2. Credit-assign image reward up the lineage → fills the "ar" track.
        resp = resp.propagate_rewards(op="mean")

        # 3. Mean image reward for the log line.
        mean_reward = 0.0
        di_rewards = resp.tracks[IMAGE_TRACK].rewards
        if di_rewards is not None:
            mean_reward = float(hydrate(di_rewards).to(torch.float32).mean().item())

        # 3b. Intrusive debug dump (best-effort) — observe what AR generated and
        #     what DiT rendered before advantages/training mutate the tracks.
        if self.dump_dir:
            self._dump_rollout(self._dump_rollout_id, req, resp)

        # 4. GRPO advantages. AR always groups by prompt. In single-engine
        #    (M=1 / UniGRPO) mode the image is 1:1 with its AR chain, so it SHARES
        #    the AR's prompt-level advantage (Â_i) — the framework's per-rewrite
        #    image grouping would be a degenerate size-1 group (advantage 0) at M=1.
        resp.tracks[AR_TRACK] = resp.tracks[AR_TRACK].compute_advantages(normalize=True)
        if self._shared_advantage:
            resp.tracks[IMAGE_TRACK] = _track_with_field(
                resp.tracks[IMAGE_TRACK], "advantages", resp.tracks[AR_TRACK].advantages
            )
        else:
            resp.tracks[IMAGE_TRACK] = resp.tracks[IMAGE_TRACK].compute_advantages(normalize=True)

        # after the debug dump (which reads decoded), before training.
        # ``reward_texts`` is 1:1 with the image track (built at scoring), so it
        # captions the image previews correctly.
        self._drop_decoded(
            req,
            resp,
            rollout_id=rollout_id,
            media_prompts={IMAGE_TRACK: list(reward_texts.texts)},
        )
        extra_metrics: Dict[str, float] = {
            "sync_weights": float(bool(sync_weights)),
            **rollout_boundary_metrics,
        }
        if self._strict_bagel_t2ti:
            ar_track = resp.tracks[AR_TRACK]
            image_track = resp.tracks[IMAGE_TRACK]
            if ar_track.batch_size != image_track.batch_size:
                raise RuntimeError(
                    "BAGEL T2TI replay balancing requires 1:1 AR/image tracks; "
                    f"got {ar_track.batch_size} and {image_track.batch_size}."
                )
            if image_track.parent_ids is None or list(image_track.parent_ids) != list(ar_track.sample_ids):
                raise RuntimeError(
                    "BAGEL T2TI replay balancing requires positional M=1 lineage: "
                    "image parent_ids must equal AR sample_ids."
                )
            replay_specs = BagelT2TIDiffusionConditions.from_dict(image_track.conditions).replay_specs
            depths = [len(spec.chunk_offsets) - 1 for spec in replay_specs]
            extra_metrics.update(_bagel_t2ti_replay_depth_metrics(depths))
            permutation = _bagel_t2ti_replay_permutation(
                image_track,
                num_shards=self._train_dp_size,
                num_updates=self._num_updates_per_batch,
            )
            if permutation is not None:
                shard_size = len(depths) // self._train_dp_size

                def padded_work(order: List[int]) -> int:
                    return sum(
                        max(depths[order[rank * shard_size + offset]] for rank in range(self._train_dp_size))
                        for offset in range(shard_size)
                    )

                original_work = padded_work(list(range(len(depths))))
                balanced_work = padded_work(permutation.tolist())
                extra_metrics.update(
                    {
                        "bagel_t2ti_replay_collective_work_original": float(original_work),
                        "bagel_t2ti_replay_collective_work_balanced": float(balanced_work),
                    }
                )
                logger.info(
                    "BAGEL T2TI replay balancing: collective traversal target %d -> %d (%.1f%% reduction)",
                    original_work,
                    balanced_work,
                    100.0 * (original_work - balanced_work) / max(original_work, 1),
                )
                resp.tracks[AR_TRACK] = ar_track.select(permutation)
                resp.tracks[IMAGE_TRACK] = image_track.select(permutation)

        # 5. Two backward (shared backbone) → one optimizer step.
        results: Dict[str, TrainStepResult] = self.stack.train_track(
            resp.tracks[AR_TRACK],
            resp.tracks[IMAGE_TRACK],
            training_progress=float(training_progress),
        )
        self.wandb_logger.log_rollout_step(
            rollout_id,
            results,
            resp,
            step_time_s=time.perf_counter() - t0,
            extra_metrics=extra_metrics,
        )

        # 6. Back to steady state (base on CPU) so the next rollout's engines
        #    have room to wake.
        if self._enable_fsdp_offload:
            if self._single_engine and not self._rollout_is_trainside:
                self.backend.prepare_for_rollout()
            else:
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
            replay_conditions = image_track.conditions.get("bagel_t2ti") if image_track is not None else None
            replay_specs = list(getattr(replay_conditions, "replay_specs", ()))

            rewards = None
            if image_track is not None and image_track.rewards is not None:
                rewards = hydrate(image_track.rewards).to(torch.float32).tolist()

            # Save images (best-effort): hydrate pixels and write per-sample PNGs.
            n_imgs = 0
            if img_decoded is not None and getattr(img_decoded, "pixels", None) is not None:
                from torchvision.utils import save_image

                pixels = hydrate(img_decoded.pixels).detach().to(torch.float32).clamp(0, 1).cpu()
                n_imgs = int(pixels.shape[0])
                for k in range(n_imgs):
                    save_image(pixels[k], os.path.join(out_dir, f"img_{k}.png"))

            # Two-level lineage: image sample k (0..P*N*M-1) descends from AR
            # sample k // M and original prompt k // (N*M). Index the smaller
            # prompt / recaption lists through those factors.
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
                                # In think_recaption the AR output IS the text fed
                                # into DiT (the recaption conditions the DiT stage).
                                "ar_text_fed_to_dit": ar_texts[a_idx] if a_idx < len(ar_texts) else None,
                                "image_reward": rewards[k] if (rewards is not None and k < len(rewards)) else None,
                                "image_file": f"img_{k}.png" if k < n_imgs else None,
                                # Metadata only (token IDs + scheduler boundaries),
                                # never KV tensors. This makes exact/collapsed
                                # real-trace parity reproducible from a debug dump.
                                "t2ti_replay": dataclasses.asdict(replay_specs[k]) if k < len(replay_specs) else None,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            logger.info("[HI3-DUMP] rollout %d → %s (%d samples, %d images)", rollout_id, out_dir, n, n_imgs)
        except Exception as exc:  # noqa: BLE001 — dump must never break training
            logger.warning("[HI3-DUMP] rollout %d dump failed (non-fatal): %s", rollout_id, exc)

    def _eval_sampling_params(self) -> Dict[str, BaseSamplingParams]:
        """Build eval params, making deterministic eta=0 an explicit ODE rollout."""
        base_diffusion = self.sampling_params.get("diffusion")
        replace_kwargs: Dict[str, Any] = {"eta": self.eval_eta}
        if "cfg_text_scale" in {field.name for field in dataclasses.fields(base_diffusion)}:
            replace_kwargs["cfg_text_scale"] = self.eval_cfg_text_scale
        else:
            replace_kwargs["guidance_scale"] = self.eval_cfg_text_scale
        if self.eval_eta <= 1e-7:
            replace_kwargs.update(scheduler=None, sde_indices=[])
        eval_diffusion = dataclasses.replace(base_diffusion, **replace_kwargs)
        return {**self.sampling_params, "diffusion": eval_diffusion}

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

        External engines sync current weights once per eval. The single-engine
        path uses the same staged extract → wake/push/generate → sleep lifecycle
        as training, with exception-safe snapshot cleanup; unlike training, it
        leaves FSDP offloaded because eval has no backward. The trainside path
        needs none of it because it shares the live FSDP modules.
        """
        eval_sp = self._eval_sampling_params()
        # Two-engine: sync the CURRENT adapter once for the whole eval (PUSH
        # needs awake engines, so it rides one short wake/sleep cycle here).
        if not self._single_engine and self.weight_sync is not None:
            if self._enable_fsdp_offload:
                self.backend.onload()
            self.weight_sync.extract()
            if self._enable_fsdp_offload:
                self.backend.offload()
            for eng in self.ar_rollouts + self.dit_rollouts:
                eng.wake_up()
            self.weight_sync.push()
            for eng in self.ar_rollouts + self.dit_rollouts:
                eng.sleep()
        single_session = (
            self._external_single_engine_session(
                sync_weights=True,
                onload_trainer_after=False,
            )
            if self._single_engine and not self._rollout_is_trainside
            else nullcontext()
        )
        with single_session:
            # Default pass: training reward + shared-set suites score the SAME images.
            scorers = [("reward", self.reward)] + [
                (s.name, s.reward) for s in self._eval_suites if s.data_source is None
            ]
            metrics = self._eval_pass(
                self.data_source,
                self.eval_num_prompts,
                scorers,
                eval_sp,
                step,
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
                        )
                    )
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
        n_prompts = len(all_inputs.sample_ids)
        chunk = max(1, self.batch_size)
        usable = n_prompts - n_prompts % chunk or n_prompts
        sums = {name: 0.0 for name, _ in scorers}
        counts = {name: 0 for name, _ in scorers}
        for start in range(0, usable, chunk):
            sub = all_inputs.slice(start, min(start + chunk, n_prompts))
            req = self._build_req(sub, step, base_sampling=eval_sp)
            if self._single_engine:
                resp = self.run_rollout(req)
            else:
                for eng in self.ar_rollouts + self.dit_rollouts:
                    eng.wake_up()
                resp = self.run_rollout(req)
                for eng in self.ar_rollouts + self.dit_rollouts:
                    eng.sleep()
            # Score the image track only; build a reward req aligned 1:1 with the
            # image track (each image's ORIGINAL prompt) so req and track shard
            # together across DP ranks (mirrors train_step).
            img_track = resp.tracks[IMAGE_TRACK]
            ar_params = req.sampling_params.get("ar")
            diff_params = req.sampling_params.get("diffusion")
            n_rec = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            n_img = int(diff_params.samples_per_prompt)
            orig_texts = req.primitives.get("text")
            reward_texts = Texts(
                texts=[orig_texts.texts[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))]
            )
            reward_metadata = (
                [req.metadata[i // (n_rec * n_img)] for i in range(len(img_track.sample_ids))] if req.metadata else []
            )
            reward_req = RolloutReq(
                sample_ids=list(img_track.sample_ids),
                group_ids=list(img_track.parent_ids) if img_track.parent_ids else list(img_track.sample_ids),
                primitives={"text": reward_texts},
                request_conditions={},
                sampling_params=req.sampling_params,
                metadata=reward_metadata,
            )
            for name, reward in scorers:
                scored = reward.score_and_attach(req=reward_req, track=img_track)
                if scored.rewards is not None:
                    r = hydrate(scored.rewards).to(torch.float32)
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
        if self._strict_bagel_t2ti and int(weight_sync_interval) != 1:
            raise ValueError(
                "UnifiedModelTrainer: bagel_t2ti requires weight_sync_interval=1 so every rollout uses "
                "the current trainer policy; skipped syncs invalidate the replay old-policy anchor."
            )
        interval = max(1, weight_sync_interval)
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        if load_dir and self._strict_bagel_t2ti:
            self.image_algorithm.load_reference_checkpoint(os.path.abspath(load_dir))
        resumed = bool(load_dir)
        # Fast-forward the data stream to the resume point — exact when
        # run.seed is set (deterministic shuffle); with seed=null the stream
        # is non-reproducible anyway.
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "park_optimizer_state_during_rollout": self._park_optimizer_state_during_rollout,
            },
        )
        try:
            if self.eval_interval > 0:
                self.evaluate(start_rollout)  # baseline eval before any training
            for rollout_id in range(start_rollout, num_rollouts):
                training_progress = rollout_id / max(1, num_rollouts - 1)
                self._dump_rollout_id = rollout_id  # picked up by train_step's dump
                inputs = self.data_source.get_samples(self.batch_size)
                req = self._build_req(inputs, rollout_id)
                # Sync before generate; skip step 0 (nothing trained yet). On
                # resume, force the first sync — the engine booted with fresh
                # weights and needs the restored adapter before generate. The
                # HI3_SYNC_FIRST env forces a sync on rollout 0 too — a debug knob
                # to exercise the LoRA-sync path early (cheaply) without a full
                # extra rollout; the rollout-0 adapter is ~0 but that's fine for
                # testing the register→activate mechanism.
                force_sync = (resumed and rollout_id == start_rollout) or (
                    rollout_id == 0 and bool(os.environ.get("HI3_SYNC_FIRST"))
                )
                sync_weights = force_sync or (rollout_id > 0 and rollout_id % interval == 0)
                results, mean_reward = self.train_step(
                    req,
                    training_progress=training_progress,
                    sync_weights=sync_weights,
                    rollout_id=rollout_id,
                )
                # Per-track console line (ar / image) with the step-0 ratio probe
                # (π_old vs π_θ alignment): on rollout 0 the LoRA is ~0 so a correct
                # replay should give ratio≈1, std≈0; a systematic offset means the
                # logp convention (temperature / top-k-p filtering / full-vs-renorm
                # softmax) doesn't match vLLM's sampler.
                self.wandb_logger.log_progress(rollout_id, num_rollouts, results, mean_reward, logger=logger)
                # eval(k) BEFORE save(checkpoint-k) at the same step, so a
                # resumed checkpoint re-runs the same eval (A/B consistency).
                if self.eval_interval > 0 and (rollout_id + 1) % self.eval_interval == 0:
                    self.evaluate(rollout_id + 1)
                checkpoint_path = self.maybe_save_checkpoint(
                    rollout_id, num_rollouts, save_interval=save_interval, save_dir=save_dir, save_mode=save_mode
                )
                if checkpoint_path is not None and self._strict_bagel_t2ti:
                    self.image_algorithm.save_reference_checkpoint(checkpoint_path)
        finally:
            self._finish_wandb()


__all__ = ["UnifiedModelTrainer"]
