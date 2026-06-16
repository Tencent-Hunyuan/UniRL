import functools
import json
import logging
import os
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from unirl.distributed.group.device_pool import DevicePool

logger = logging.getLogger(__name__)


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
            "samples_per_prompt": getattr(sampling_params, "samples_per_prompt", None),
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

    def _drop_decoded(
        self,
        req: Any,
        resp: Any,
        *,
        rollout_id: int,
        media_prompts: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Upload media previews (if due this rollout) then free ``decoded``.

        Two jobs at the single pre-train chokepoint every trainer hits — and
        both FINISH here, so no preview payload (PIL images / raw video
        tensors) ever rides into the ``train_track`` dispatch (the track is
        DP_SCATTER-serialized to the training workers right after this call):

        1. **Media logging (driver-side).** When the logger wants media this
           rollout (``UniRLWandBLogger.should_log_media``), take each track's
           inbound ``media_preview`` (populated upstream by an actor-side
           collector — none exist today) or build one from the still-live
           ``decoded`` (``build_media_preview_for_track`` hydrates a single DP
           shard), cap to ``media_max_items``, and upload it immediately at the
           same ``rollout/step`` value :meth:`UniRLWandBLogger.log_rollout_step`
           uses, so the panels align. ``media_prompts`` supplies per-track,
           sample-aligned captions for multi-track recipes whose ``req`` text is
           shorter than the expanded track.
        2. **Free the per-rollout payloads.** ``decoded`` (generated
           Images/Videos/Texts) is consumed upstream by
           ``reward.score_and_attach`` and never read by training (which uses
           only segment/conditions/advantages); ``media_preview`` was just
           uploaded (or skipped — off-cadence rollouts drop it unlogged).
           Nulling both on the driver ``resp`` before ``train_track`` releases
           the driver-held TensorStore handles before the optimizer-step memory
           peak and keeps the training dispatch free of logging payloads.

        Call after scoring / advantages (and any decoded-reading debug dump),
        immediately before dispatching to ``train_track``.
        """
        wb = self.wandb_logger
        if wb is not None and wb.should_log_media(rollout_id):
            from unirl.types.media_preview import build_media_preview_for_track

            prompts_by_track = media_prompts or {}
            multi = len(resp.tracks) > 1
            for name, track in resp.tracks.items():
                preview = track.media_preview
                if preview is None and track.decoded is not None:
                    preview = build_media_preview_for_track(
                        req=req,
                        track=track,
                        max_items=wb.media_max_items,
                        prompts=prompts_by_track.get(name),
                    )
                if preview is None:
                    continue
                if len(preview) > wb.media_max_items:
                    preview = preview.slice(0, wb.media_max_items)
                key = f"rollout/{name}/generated_media" if multi else "rollout/generated_media"
                wb.log_generated_media(rollout_id + 1, preview, key=key)

        for track in resp.tracks.values():
            track.decoded = None
            track.media_preview = None

    def _finish_wandb(self) -> None:
        """Close the wandb run if one is open."""
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

        ``save_interval <= 0`` disables saving. Writes the backend state to
        ``<save_dir>/checkpoint-<step>/checkpoint.pt`` (``save_dir`` defaults
        to ``./checkpoints``; ``save_mode="auto"`` keeps only LoRA keys when
        LoRA is active and writes full checkpoints otherwise).
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
        self.backend.save(path, step=step, mode=save_mode)
        # Driver-owned state rides beside the worker-written checkpoint.pt:
        # the wandb run id + train/ step axis let a resume append to the SAME
        # wandb run instead of starting a fresh, misaligned one.
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump({"wandb_run_id": self.wandb_logger.run_id, "optimizer_step": self.wandb_logger.optimizer_step}, f)

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
