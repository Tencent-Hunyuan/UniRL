import functools
import json
import logging
import os
import sys
from dataclasses import replace
from typing import Any, Dict, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.device_pool import DevicePool
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams, BaseSamplingParams, total_samples_per_prompt

logger = logging.getLogger(__name__)

# Upper bound (seconds) on the teardown checkpoint/weight-sync flush, applied ONLY on
# the exception path. A healthy exit (Ctrl-C, a driver-side error) drains an in-flight
# async DCP save well within this; a worker wedged in an NCCL collective (e.g. one that
# OOM'd mid-backward) cannot service the flush's ray.get, so the bound stops it hanging
# forever and masking the primary exception. Env-tunable.
_TEARDOWN_FLUSH_TIMEOUT_S = float(os.environ.get("UNIRL_TEARDOWN_FLUSH_TIMEOUT_S", "120"))


def prepare_input_sample(
    inputs: Sample,
    rollout_id: int,
    *,
    allowed_primitives: set[str],
    caller: str,
    root_control: Optional[Dict[str, Any]] = None,
    require_single_input_part: bool = False,
) -> Sample:
    """Prepare a data-source input tree for one rollout without rebuilding it.

    The data-source boundary is an input-only :class:`Sample`: metadata lives on
    its root and multimodal inputs are already chained as child Parts.  Trainers
    preserve that structure, namespace *every* id so separate rollouts cannot
    alias, optionally merge trainer-owned routing control onto the root, and then
    append their own generation fork(s). ``require_single_input_part`` lets serial
    runners reject trees they cannot yet return intact instead of silently dropping
    descendants.
    """
    if not isinstance(inputs, Sample):
        raise TypeError(f"{caller}: expected Sample input, got {type(inputs).__name__}.")
    if not inputs.parts:
        raise ValueError(f"{caller}: input Sample must contain at least one Part.")
    root_text = inputs.parts[0].primitives.get("text")
    if not isinstance(root_text, Texts):
        raise TypeError(
            f"{caller}: input Sample root requires primitives['text']: Texts; "
            f"got {type(root_text).__name__ if root_text is not None else 'None'}."
        )
    generated = [i for i, part in enumerate(inputs.parts) if part.is_gen]
    if generated:
        raise ValueError(f"{caller}: data-source Sample must be input-only; generated Parts at {generated}.")
    if require_single_input_part and len(inputs.parts) != 1:
        raise ValueError(f"{caller}: this trainer requires exactly one input Part; got {len(inputs.parts)}.")

    present = {key for part in inputs.parts for key in part.primitives}
    unsupported = present - set(allowed_primitives)
    if unsupported:
        raise ValueError(f"{caller}: unsupported input primitive keys: {sorted(unsupported)}")

    namespaced = inputs.map_sample_ids(lambda sample_id: f"r{rollout_id}:{sample_id}")
    if root_control is None:
        return namespaced
    root = namespaced.parts[0]
    root = replace(root, control={**root.control, **root_control})
    return namespaced.with_parts([root, *namespaced.parts[1:]])


def build_sampling_dict(sampling_cfg: DictConfig) -> Dict[str, BaseSamplingParams]:
    """Instantiate a Hydra ``sampling`` config into the modality-keyed runtime dict.

    The trainer's ``sampling_params`` is a ``Dict[str, BaseSamplingParams]`` keyed
    by modality (each modality's params ride on its gen Part at request build).
    Two config shapes are accepted so the flat single-modality recipes need no rewrite:

    - **Flat** (``sampling: {_target_: …DiffusionSamplingParams, …}``) — one
      params object, wrapped under its modality key (``"ar"`` for
      ``ARSamplingParams``, else ``"diffusion"``).
    - **Composed** (``sampling: {ar: {_target_: …}, diffusion: {_target_: …}}``)
      — each entry instantiated under its key. (Hydra does not recurse into a
      ``_target_``-less mapping, so we instantiate per entry.)
    """
    if "_target_" in sampling_cfg:
        obj = instantiate(sampling_cfg)
        return {"ar" if isinstance(obj, ARSamplingParams) else "diffusion": obj}
    return {key: instantiate(sub) for key, sub in sampling_cfg.items()}


def init_transfer_queue(cfg: DictConfig) -> Optional[dict]:
    """Driver-side TransferQueue bootstrap for ``transport_kind=transfer_queue``.

    Spins up the TransferQueue controller + storage backend from the ``cfg.transfer_queue``
    block and returns the **actor** handoff to pass to ``DevicePool(tq_handoff=...)`` —
    each Worker builds its own queue client from it (see ``build_transport``). The driver
    also creates its own client and installs a driver ``TQTransport``: reward/advantage
    materialization runs on the driver and hydrates TQ refs via
    ``TQTensorHandle.local() -> TensorTransportRuntime.current()``. The TransferQueue is a
    GLOBAL backend with no per-ref owning worker to RPC, so without a driver transport that
    ``.local()`` would raise "no TensorTransport installed". ``install()`` binds the runtime
    process-globally, keeping the controller/backend actors alive. Returns ``None`` for
    non-tq backends (colocate/gpu).
    """
    if cfg.get("transport_kind", "colocate_store") not in ("transfer_queue", "tq"):
        return None
    from unirl.distributed.tensor import TensorTransportRuntime
    from unirl.distributed.tensor.backend.transfer_queue import TransferQueueRuntime
    from unirl.distributed.tensor.backend.transfer_queue.runtime import _DEFAULT_PARTITION_ID
    from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport

    rt = TransferQueueRuntime().install()
    handoffs = rt.init(cfg)
    if handoffs is None:
        raise RuntimeError(
            "transport_kind='transfer_queue' requires a `transfer_queue:` config block, e.g.\n"
            "  transfer_queue:\n"
            "    _target_: unirl.distributed.tensor.backend.transfer_queue.simple.SimpleBackend\n"
            "    num_units: 16\n    unit_size: 1024"
        )
    controller_handoff, actor_handoff = handoffs
    # Driver client + transport: driver-side reward/advantage hydration resolves TQ refs
    # through the process TensorTransport (TQTensorHandle.local() -> .current()).
    rt.create_client("Driver", controller_handoff, sync=False)
    TensorTransportRuntime.install(TQTransport(rt, partition_id=_DEFAULT_PARTITION_ID))
    return actor_handoff


class BaseTrainer:
    """Owns a DevicePool. Subclasses use ``placement(self.pool, ...)`` to
    instantiate their ``Remote`` roles inside ``__init__`` / ``setup``.

    Also owns the (rank-0/driver) Weights & Biases logger shared by every
    trainer. Subclasses call :meth:`_init_wandb` once at the top of ``train``
    (it always builds a logger — a no-op null-object when reporting is off),
    then ``self.wandb_logger.log_rollout_step(...)`` / ``log_progress(...)``
    after each ``train_step``, and :meth:`_finish_wandb` in a ``finally``.
    """

    def __init__(
        self,
        *,
        cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
    ) -> None:
        # Device topology and tensor transport are driven entirely by top-level
        # cfg keys (num_devices / devices_per_node / workers_per_device /
        # transport_kind / transfer_queue), so the base owns the whole pool +
        # TransferQueue bootstrap here. Subclasses never thread these through:
        # they hand us ``cfg`` and get the configured pool for free.
        self.num_devices = cfg.num_devices
        self.pool = DevicePool(
            num_devices=cfg.num_devices,
            devices_per_node=int(cfg.get("devices_per_node", 8)),
            workers_per_device=int(cfg.get("workers_per_device", 1)),
            transport_kind=cfg.get("transport_kind", "colocate_store"),
            tq_handoff=init_transfer_queue(cfg),
            # Default 1 keeps every existing trainer's single-threaded-actor
            # semantics byte-identical; the agentic rollout's rank-0 coordinator
            # opts in (>=3) so it can run generate + its own drain + serve
            # next_task pulls concurrently on a threaded Worker (LIN-519/LIN-522).
            worker_max_concurrency=int(cfg.get("worker_max_concurrency", 1)),
        )
        self.pool.setup()

        # Driver/rank-0 wandb logger. Starts as a disabled null-object so trainers
        # can call ``self.wandb_logger.X(...)`` without guards even before
        # _init_wandb runs; _init_wandb replaces it with the configured (possibly
        # live) logger. Disabled => wandb methods no-op, log_progress still prints.
        # The optimizer-step counter now lives on the logger.
        from unirl.utils.wandb_logger import UniRLWandBLogger

        self.logging_cfg = logging_cfg
        self.wandb_logger = UniRLWandBLogger(enabled=False)
        # Driver-side state from a resumed checkpoint's trainer_state.json
        # (wandb run id / step axis); populated by maybe_load_checkpoint,
        # consumed by _init_wandb. Empty for fresh runs.
        self._resume_state: Dict[str, Any] = {}

        # Reclaim per-rollout transport buffers after every train_step, centrally,
        # so each subclass train loop doesn't have to remember to.
        self._install_train_step_reset_hook()

        # Time the standard step collaborators (rollout / weight_sync / reward /
        # stack) and surface them as perf/<phase>_time_s, centrally, so every
        # trainer gets step attribution without per-trainer edits. The machinery
        # lives with the rest of the logging stack in wandb_logger.
        from unirl.utils.wandb_logger import install_phase_timing

        install_phase_timing(self)

        # verl-parity memory monitoring (perf/max_memory_* + [mem] boundary
        # lines). Only constructed here — the collaborators to wrap don't exist
        # until the subclass __init__ finishes, so install() runs in _init_wandb.
        # None when disabled (logging.memory.enabled=false / UNIRL_MEM_MONITOR=0).
        from unirl.utils.memory_monitor import install_memory_monitoring

        self._memory_monitor = install_memory_monitoring(self)

    # ---- transport buffer reclaim (shared by all v2 trainers) --------------

    def _install_train_step_reset_hook(self) -> None:
        """Wrap ``train_step`` so :meth:`_reset_transport_buffers` runs after each call.

        Only installed for the transfer_queue backend; colocate/gpu_store keep their
        ``train_step`` untouched (their reclaim is a no-op anyway). Every v2 trainer has
        its own ``train`` loop but they all drive one ``train_step`` per rollout, so this
        is the single seam that reclaims per-rollout TQ buffers without per-trainer edits.
        The reset fires once ``train_step`` returns — rewards/advantages materialized, no
        live ``TensorRef`` ref into the queue's RDMA buffers remaining.
        """
        if self.pool.transport_kind not in ("transfer_queue", "tq"):
            return
        inner = getattr(self, "train_step", None)
        if not callable(inner):
            return

        @functools.wraps(inner)
        def _train_step(*args, **kwargs):
            result = inner(*args, **kwargs)
            self._reset_transport_buffers()
            return result

        self.train_step = _train_step

    def _reset_transport_buffers(self) -> None:
        """Reclaim per-rollout mooncake zero-copy buffers (no-op for other backends)."""
        self.pool.reset_transfer_queue_buffers()

    # ---- wandb logging (shared by all v2 trainers) -------------------------

    def _init_wandb(self, *, num_rollouts: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Build the (rank-0/driver) wandb logger from the optional ``logging`` block.

        The single logger factory shared by every trainer. ALWAYS assigns
        ``self.wandb_logger`` — a live run when ``report_to_wandb`` is on and a
        ``project_name`` is set, otherwise a disabled null-object whose wandb
        methods no-op (so trainers call ``self.wandb_logger.X(...)`` without
        guards, while ``log_progress`` still prints). The whole ``train`` loop
        runs on the driver, so ``rank=0``.

        Reads (all under the ``logging`` block, all optional): ``report_to_wandb``,
        ``project_name``, ``run_name``, ``entity`` (falls back to ``WANDB_ENTITY``),
        ``tags`` (list or comma-separated string), ``logging_dir``, and the media
        knobs ``log_media`` / ``media_max_items`` / ``media_log_interval``. Enabling
        reporting inherently requires a successful wandb init (it raises on
        failure) — there is no opt-out flag.
        """
        from unirl.utils.wandb_logger import init_logger

        cfg = self.logging_cfg or {}
        report = bool(cfg.get("report_to_wandb", False)) and bool(cfg.get("project_name"))

        raw_tags = cfg.get("tags")
        if isinstance(raw_tags, str):
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        elif raw_tags:
            tags = [str(t).strip() for t in raw_tags if str(t).strip()]
        else:
            tags = None

        sampling_params = getattr(self, "sampling_params", None)
        run_config: Dict[str, Any] = {
            "num_devices": self.num_devices,
            "batch_size": getattr(self, "batch_size", None),
            "num_rollouts": num_rollouts,
            "samples_per_prompt": total_samples_per_prompt(sampling_params) if sampling_params else None,
        }
        if extra:
            run_config.update(extra)

        project = cfg.get("project_name")
        self.wandb_logger = init_logger(
            project=str(project) if project else None,
            run_name=cfg.get("run_name"),
            config=run_config,
            log_dir=cfg.get("logging_dir"),
            rank=0,
            tags=tags,
            entity=(cfg.get("entity") or os.environ.get("WANDB_ENTITY") or None),
            log_media=bool(cfg.get("log_media", False)),
            media_max_items=int(cfg.get("media_max_items", 8)),
            media_log_interval=int(cfg.get("media_log_interval", 1)),
            enabled=report,
            run_id=self._resume_state.get("wandb_run_id"),
            optimizer_step=int(self._resume_state.get("optimizer_step") or 0),
        )
        if self.wandb_logger.initialized:
            logger.info("WandB initialized: project=%s run=%s", project, cfg.get("run_name"))

        # Every trainer calls _init_wandb at the top of train(): the subclass
        # __init__ has finished (collaborators exist) and the live logger is in
        # place — wrap the hand-off boundaries with memory probes here, once.
        if self._memory_monitor is not None:
            self._memory_monitor.install(self)

    def _drop_decoded(
        self,
        sample: Sample,
        *,
        rollout_id: int,
    ) -> None:
        """Upload media previews (if due this rollout) then free decoded payloads.

        Two jobs at the single pre-train chokepoint every trainer hits — and
        both FINISH here, so no decoded payload (PIL images / raw video tensors)
        ever rides into the ``train_track`` dispatch (each gen ``Part`` is
        DP_SCATTER-serialized to the training workers right after this call):

        1. **Media logging (driver-side).** When the logger wants media this
           rollout (``UniRLWandBLogger.should_log_media``), take each gen Part's
           inbound ``media_preview`` or build one from the still-live
           ``primitives`` (``build_media_preview_for_part`` hydrates a single DP
           shard), cap to ``media_max_items``, and upload it at the same
           ``rollout/step`` value :meth:`UniRLWandBLogger.log_rollout_step` uses,
           so the panels align. Captions default to the frontier-aligned prompt
           texts (``Sample.conditioning``).
        2. **Free the per-rollout payloads.** ``primitives`` (generated
           Images/Videos/Texts/Audios) is consumed upstream by reward scoring and never
           read by training (which uses only segment/conditions/advantages);
           ``media_preview`` was just uploaded (or skipped off-cadence). Clearing
           the primitive map, its metadata, and the preview before ``train_track``
           releases the driver-held
           TensorStore handles before the optimizer-step memory peak.

        Call after scoring / advantages (and any decoded-reading debug dump),
        immediately before dispatching to ``train_track``.
        """
        from unirl.types.primitives import Images, Texts

        gen_parts = sample.gen_parts()
        wb = self.wandb_logger
        if wb is not None and wb.should_log_media(rollout_id):
            from unirl.types.media_preview import build_media_preview_for_part

            multi = len(gen_parts) > 1
            # Frontier-aligned conditioning: captions (the prompt Texts) + the
            # it2i source image (the chained image input Part), row-aligned 1:1
            # with the frontier gen samples — exactly what the preview pairs.
            cond = sample.conditioning()
            default_prompts = next((list(c.texts) for c in cond if isinstance(c, Texts)), None)
            input_image = next((c for c in cond if isinstance(c, Images)), None)
            for part in gen_parts:
                name = "ar" if isinstance(part.sampling_params, ARSamplingParams) else "diffusion"
                preview = part.media_preview
                if preview is None and part.primitives:
                    # default_prompts is frontier-aligned (Sample.conditioning), so caption
                    # only the frontier gen Part; a non-frontier image Part (none today — the
                    # non-frontier AR Part is text → returns None) would otherwise be paired
                    # with the wrong-length caption list.
                    preview = build_media_preview_for_part(
                        part=part,
                        max_items=wb.media_max_items,
                        prompts=default_prompts if part is sample.parts[-1] else None,
                        input_image=input_image,
                    )
                if preview is None:
                    continue
                if len(preview) > wb.media_max_items:
                    preview = preview.slice(0, wb.media_max_items)
                key = f"rollout/{name}/generated_media" if multi else "rollout/generated_media"
                wb.log_generated_media(rollout_id + 1, preview, key=key)

        for part in gen_parts:
            part.primitives = {}
            part.primitive_metadata = {}
            part.media_preview = None

    def _wait_for_checkpoints(self, *, timeout: Optional[float] = None) -> None:
        """Flush a pending backend checkpoint before worker teardown.

        ``timeout`` bounds the underlying ``ray.get`` — passed on the exception
        path so a worker wedged in an NCCL collective can't hang the flush
        forever; ``None`` (the default, e.g. the final-save drain) waits
        indefinitely.
        """
        backend = getattr(self, "backend", None)
        if backend is None:
            return
        if timeout is None:
            backend.wait_for_checkpoint()
        else:
            backend.wait_for_checkpoint(_ray_get_timeout=timeout)

    def _cleanup_weight_sync(self, *, timeout: Optional[float] = None) -> None:
        """Let transports remove run-scoped artifacts before workers are killed.

        ``cleanup`` is a BROADCAST dispatch, so like the checkpoint flush it can
        wedge on a stuck worker; ``timeout`` bounds its ``ray.get`` on the
        exception path (``None`` waits indefinitely).
        """
        weight_sync = getattr(self, "weight_sync", None)
        cleanup = getattr(weight_sync, "cleanup", None)
        if not callable(cleanup):
            return
        if timeout is None:
            cleanup()
        else:
            cleanup(_ray_get_timeout=timeout)

    def _finish_wandb(self) -> None:
        """Flush pending work, clean transport artifacts, and close wandb."""
        active_exception = sys.exc_info()[0] is not None
        # On the exception path, bound the flush's ray.get: a worker wedged in an NCCL
        # collective (e.g. one that OOM'd mid-backward) can't service it, so an
        # unbounded flush would hang forever and mask the primary exception. A healthy
        # crash (Ctrl-C, a driver-side error) still drains the in-flight async DCP save
        # -- well within the bound -- so the last checkpoint is not left half-written.
        # On the success path, flush unbounded.
        timeout = _TEARDOWN_FLUSH_TIMEOUT_S if active_exception else None
        try:
            self._wait_for_checkpoints(timeout=timeout)
            self._cleanup_weight_sync(timeout=timeout)
        except Exception:
            if not active_exception:
                raise
            # GetTimeoutError (a wedged worker) or any flush error: keep the primary
            # exception, just record the failed best-effort flush.
            logger.exception("Failed to flush checkpoint/weight-sync state during trainer teardown")
        finally:
            if self.wandb_logger is not None:
                self.wandb_logger.finish()

    # ---- checkpointing (shared by single-backend trainers) -----------------

    def maybe_save_checkpoint(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str = "auto",
    ) -> None:
        """Save every ``save_interval`` rollouts (and on the last one).

        ``save_interval <= 0`` disables saving. Writes the backend state under
        ``<save_dir>/checkpoint-<step>/`` (``save_dir`` defaults to
        ``./checkpoints``). The backend's ``checkpoint_format`` selects either
        a legacy ``checkpoint.pt`` or reshardable DCP shards; ``save_mode="auto"``
        keeps only LoRA keys when LoRA is active and writes full checkpoints
        otherwise.
        Paths resolve to absolute here, on the driver — the backend runs in
        Ray workers whose CWD differs from the driver's.
        """
        if save_interval <= 0:
            return
        step = rollout_id + 1
        # Save on the interval, and always on the final rollout.
        if step % save_interval != 0 and step < num_rollouts:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step}")
        logger.info("Saving checkpoint at rollout %d/%d -> %s", step, num_rollouts, path)
        # Checkpoint saving gathers a full state_dict — a memory spike worth
        # bracketing (the one hand-off boundary outside the per-step phases).
        if self._memory_monitor is not None:
            self._memory_monitor.boundary("ckpt_save:begin", self.backend)
        self.backend.save(path, step=step, mode=save_mode)
        if self._memory_monitor is not None:
            self._memory_monitor.boundary("ckpt_save:end", self.backend)
        # Driver-owned state rides beside the worker-written checkpoint data:
        # the wandb run id + train/ step axis let a resume append to the SAME
        # wandb run instead of starting a fresh, misaligned one.
        trainer_state_path = os.path.join(path, "trainer_state.json")
        trainer_state_tmp = f"{trainer_state_path}.tmp"
        with open(trainer_state_tmp, "w") as f:
            json.dump({"wandb_run_id": self.wandb_logger.run_id, "optimizer_step": self.wandb_logger.optimizer_step}, f)
        os.replace(trainer_state_tmp, trainer_state_path)
        # An async DCP save (checkpoint_format="dcp" + checkpoint_async) writes
        # its shards on a background thread, normally drained by the next save.
        # The final checkpoint has no next save, so block until it is on disk
        # before train() returns and the workers are torn down.
        if step >= num_rollouts:
            self._wait_for_checkpoints()

    def maybe_load_checkpoint(self, load_dir: Optional[str], *, num_rollouts: Optional[int] = None) -> int:
        """Restore training state from ``load_dir``; return the rollout step to resume from.

        Returns 0 for a fresh run (``load_dir`` empty) or a checkpoint that
        predates step recording. Restores model/optimizer/scheduler plus the
        optimizer-step counter; the trainer loop continues from the returned
        step. Resolved to an absolute path on the driver (worker CWDs differ).
        """
        if not load_dir:
            return 0
        load_dir = os.path.abspath(load_dir)
        logger.info("Loading checkpoint from %s", load_dir)
        result = self.backend.load(load_dir)
        if isinstance(result, list):  # BROADCAST dispatch collects one result per worker
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
