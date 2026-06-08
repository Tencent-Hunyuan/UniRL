import functools
import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from omegaconf import DictConfig

from unirl.distributed.group.device_pool import DevicePool

if TYPE_CHECKING:
    from unirl.train.stack import TrainStepResult

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
    from unirl.distributed.tensor.backend.transfer_queue import TransferQueueRuntime
    from unirl.distributed.tensor.backend.transfer_queue.runtime import _DEFAULT_PARTITION_ID
    from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport
    from unirl.distributed.tensor.transport import TensorTransportRuntime

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

    Also owns the optional (rank-0/driver) Weights & Biases logger shared by
    every v2 trainer. Subclasses call :meth:`_init_wandb` once at the top of
    ``train``, :meth:`_log_rollout` after each ``train_step`` (a no-op when
    reporting is off), and :meth:`_finish_wandb` in a ``finally``.
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

        # Optional wandb logging (driver/rank-0). Stays ``None`` until
        # _init_wandb runs, and remains ``None`` when the ``logging`` block is
        # absent or ``report_to_wandb`` is off — every _log_rollout then no-ops.
        self.logging_cfg = logging_cfg
        self.wandb_logger = None
        self._optimizer_step = 0

        # Reclaim per-rollout transport buffers after every train_step, centrally,
        # so each subclass train loop doesn't have to remember to.
        self._install_train_step_reset_hook()

    # ---- transport buffer reclaim (shared by all v2 trainers) --------------

    def _install_train_step_reset_hook(self) -> None:
        """Wrap ``train_step`` so :meth:`_reset_transport_buffers` runs after each call.

        Only installed for the transfer_queue backend; colocate/gpu_store keep their
        ``train_step`` untouched (their reclaim is a no-op anyway). Every v2 trainer has
        its own ``train`` loop but they all drive one ``train_step`` per rollout, so this
        is the single seam that reclaims per-rollout TQ buffers without per-trainer edits.
        The reset fires once ``train_step`` returns — rewards/advantages materialized, no
        live ``TensorMeta`` ref into the queue's RDMA buffers remaining.
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

    def _init_wandb(self, *, num_rollouts: int, extra: Optional[Dict[str, Any]] = None) -> None:
        """Open the (rank-0/driver) wandb run from the optional ``logging`` block.

        No-op when the block is absent or ``report_to_wandb`` is off — every
        subsequent :meth:`_log_rollout` then short-circuits. The whole ``train``
        loop runs on the driver, so ``rank=0``.
        """
        cfg = self.logging_cfg
        if cfg is None:
            return
        if not bool(cfg.get("report_to_wandb", False)) or not cfg.get("project_name"):
            return

        from unirl.utils.wandb_logger import init_logger

        raw_tags = cfg.get("tags")
        tags = [str(t) for t in raw_tags] if raw_tags else None
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
            project=str(project),
            run_name=cfg.get("run_name"),
            config=run_config,
            rank=0,
            tags=tags,
            entity=cfg.get("entity") or None,
        )
        if self.wandb_logger.initialized:
            logger.info("WandB initialized: project=%s run=%s", project, cfg.get("run_name"))

    @staticmethod
    def _drop_decoded(resp: Any) -> None:
        """Free the reward-only ``decoded`` payload before training.

        ``decoded`` (generated Images/Videos/Texts) is consumed upstream by
        ``reward.score_and_attach`` and is never read by training (which uses
        only segment/conditions/advantages). Nulling it on the driver ``resp``
        before ``train_track`` releases the driver-held TensorStore handles so
        the (often GPU-resident, trainside) storage can free before the
        optimizer-step memory peak — worker-side nulling alone leaves the driver
        refs alive through ``_log_rollout``. ``media_preview`` is left intact
        for logging. Call after scoring / advantages (and any decoded-reading
        debug dump), immediately before dispatching to ``train_track``.
        """
        for track in resp.tracks.values():
            track.decoded = None

    def _log_rollout(
        self,
        rollout_id: int,
        results: Union["TrainStepResult", Dict[str, "TrainStepResult"]],
        resp: Any,
        *,
        step_time_s: Optional[float] = None,
    ) -> None:
        """Log one rollout's metrics to wandb. No-op when reporting is off.

        ``rollout/*`` carries reward/advantage (and, for AR tracks, response-length)
        distribution stats (single-track keys unprefixed; multi-track auto-prefixed
        by track name). ``train/*``
        carries optimizer scalars + algorithm metrics — per-track namespaced
        (``<track>/<key>``) when ``results`` is a ``{track: TrainStepResult}``
        dict, as for the PE trainer. ``perf/rollout_time_s`` is the optional
        wall-clock for the step.
        """
        wb = self.wandb_logger
        if wb is None or not wb.initialized:
            return

        from unirl.utils.wandb_logger import aggregate_stage_results
        from unirl.utils.wandb_metrics import compute_rollout_resp_metrics

        step = rollout_id + 1
        # ``trunc_len`` (AR ``max_new_tokens``) keys ``rollout/trunc_ratio``; trainers
        # without AR sampling params (diffusion) give ``None`` → trunc_ratio is skipped
        # while any AR track's response-length stats still log.
        trunc_len = getattr(getattr(self, "sampling_params", None), "max_new_tokens", None)
        wb.log_rollout(step, compute_rollout_resp_metrics(resp=resp, trunc_len=trunc_len))

        if isinstance(results, dict):
            # PE multi-track: one optimizer step, metrics namespaced by track name.
            train_metrics: Dict[str, Any] = {
                f"{name}/{key}": value
                for name, result in results.items()
                for key, value in aggregate_stage_results([result]).items()
            }
            if any(bool(r.has_backward) for r in results.values()):
                self._optimizer_step += 1
                wb.log_step(self._optimizer_step, train_metrics)
        else:
            self._log_train_per_step(results)

        if step_time_s is not None:
            wb.log_perf(step, {"rollout_time_s": float(step_time_s)})

    def _log_train_per_step(self, result: "TrainStepResult") -> None:
        """Log the ``train/`` panel one wandb point PER optimizer step.

        With ``num_updates_per_batch > 1`` the train stack attaches each optimizer
        step's own metrics on ``result.per_update`` (one Mapping per update, in
        order). We emit one ``log_step`` per update at its own ``_optimizer_step``
        with the metrics **unprefixed**, so each metric (e.g. ``train/ratio_mean``)
        stays a single per-step series — truthful and never averaged across updates,
        so it reads as a sawtooth (the on-policy step 1.0, the off-policy steps
        drifting). One series regardless of ``num_updates_per_batch`` (no
        ``update{i}/`` proliferation). A single update logs the aggregate once.
        """
        from unirl.utils.wandb_logger import aggregate_stage_results

        wb = self.wandb_logger
        if wb is None:
            return
        if len(result.per_update) > 1:
            for metrics in result.per_update:
                self._optimizer_step += 1
                wb.log_step(self._optimizer_step, dict(metrics))
        elif result.has_backward:
            self._optimizer_step += 1
            wb.log_step(self._optimizer_step, dict(aggregate_stage_results([result])))

    def _finish_wandb(self) -> None:
        """Close the wandb run if one is open."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish()
