import functools
import json
import logging
import os
import sys
from dataclasses import replace
from typing import Any, Dict, List, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig

from unirl.distributed.group.device_pool import DevicePool
from unirl.rollout.manager import required_worker_concurrency
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import ARSamplingParams, BaseSamplingParams, total_samples_per_prompt

logger = logging.getLogger(__name__)

# Bound exception-path flushes so wedged workers cannot mask the primary error.
_TEARDOWN_FLUSH_TIMEOUT_S = float(os.environ.get("UNIRL_TEARDOWN_FLUSH_TIMEOUT_S", "120"))


def resolve_worker_max_concurrency(
    cfg: DictConfig,
    *,
    per_worker_inflight: Optional[int] = None,
) -> int:
    """Resolve actor concurrency, deriving it from rollout lanes when omitted."""
    configured = cfg.get("worker_max_concurrency")
    if configured is not None:
        return int(configured)

    effective_inflight = per_worker_inflight
    if effective_inflight is None:
        effective_inflight = cfg.get("per_worker_inflight")
    return 1 if effective_inflight is None else required_worker_concurrency(effective_inflight)


def resolve_worker_concurrency_by_device(
    cfg: DictConfig,
    *,
    rollout_concurrency: int,
) -> Optional[List[int]]:
    """Keep a separate-layout train slab serial when rollout concurrency is derived."""
    if cfg.get("worker_max_concurrency") is not None or cfg.get("per_worker_inflight") is None:
        return None
    layout = cfg.get("layout")
    if layout is not None and str(layout) != "separate":
        return None
    train_fraction = float(cfg.get("train_fraction", 0.5))
    num_devices = int(cfg.num_devices)
    train_devices_float = train_fraction * num_devices
    train_devices = int(round(train_devices_float))
    if abs(train_devices_float - train_devices) > 1e-9 or not 0 < train_devices < num_devices:
        return None
    return [1] * train_devices + [rollout_concurrency] * (num_devices - train_devices)


def prepare_input_sample(
    inputs: Sample,
    rollout_id: int,
    *,
    allowed_primitives: set[str],
    caller: str,
    root_control: Optional[Dict[str, Any]] = None,
    require_single_input_part: bool = False,
) -> Sample:
    """Prepare a data-source input tree for one rollout without rebuilding it."""
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
    root = namespaced.parts[0]
    metadata = root.metadata or [{} for _ in root.sample_ids]
    if len(metadata) != len(root.sample_ids):
        raise ValueError(f"{caller}: root metadata has {len(metadata)} rows for {len(root.sample_ids)} root samples.")
    root = replace(root, metadata=[{**(row or {}), "rollout_id": int(rollout_id)} for row in metadata])
    if root_control is not None:
        root = replace(root, control={**root.control, **root_control})
    return namespaced.with_parts([root, *namespaced.parts[1:]])


def build_sampling_dict(sampling_cfg: DictConfig) -> Dict[str, BaseSamplingParams]:
    """Instantiate a Hydra ``sampling`` config into the modality-keyed runtime dict."""
    if "_target_" in sampling_cfg:
        obj = instantiate(sampling_cfg)
        return {"ar" if isinstance(obj, ARSamplingParams) else "diffusion": obj}
    return {key: instantiate(sub) for key, sub in sampling_cfg.items()}


def unwrap_replicated_int(value: object, *, name: str) -> int:
    """Normalize a BROADCAST return and verify all worker replicas agree."""
    if isinstance(value, (list, tuple)):
        if not value or any(not isinstance(item, int) for item in value):
            raise TypeError(f"{name} returned invalid worker values: {value!r}")
        if any(item != value[0] for item in value[1:]):
            raise RuntimeError(f"{name} disagrees across workers: {value!r}")
        return value[0]
    if not isinstance(value, int):
        raise TypeError(f"{name} returned {type(value).__name__}, expected int")
    return value


def init_transfer_queue(cfg: DictConfig) -> Optional[dict]:
    """Driver-side TransferQueue bootstrap for ``transport_kind=transfer_queue``."""
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
    rt.create_client("Driver", controller_handoff, sync=False)
    TensorTransportRuntime.install(TQTransport(rt, partition_id=_DEFAULT_PARTITION_ID))
    return actor_handoff


class BaseTrainer:
    """Owns a DevicePool. Subclasses use ``placement(self.pool, ...)`` to"""

    def __init__(
        self,
        *,
        cfg: DictConfig,
        logging_cfg: Optional[DictConfig] = None,
        per_worker_inflight: Optional[int] = None,
    ) -> None:
        self.num_devices = cfg.num_devices
        resolved_concurrency = resolve_worker_max_concurrency(
            cfg,
            per_worker_inflight=per_worker_inflight,
        )
        per_device_concurrency = (
            resolve_worker_concurrency_by_device(
                cfg,
                rollout_concurrency=resolved_concurrency,
            )
            if per_worker_inflight is None
            else None
        )
        self.pool = DevicePool(
            num_devices=cfg.num_devices,
            devices_per_node=int(cfg.get("devices_per_node", 8)),
            workers_per_device=int(cfg.get("workers_per_device", 1)),
            transport_kind=cfg.get("transport_kind", "colocate_store"),
            tq_handoff=init_transfer_queue(cfg),
            worker_max_concurrency=(resolved_concurrency if per_device_concurrency is None else per_device_concurrency),
        )
        self.pool.setup()

        from unirl.utils.wandb_logger import UniRLWandBLogger

        self.logging_cfg = logging_cfg
        self.wandb_logger = UniRLWandBLogger(enabled=False)
        self._resume_state: Dict[str, Any] = {}

        self._install_train_step_reset_hook()

        from unirl.utils.wandb_logger import install_phase_timing

        install_phase_timing(self)

        from unirl.utils.memory_monitor import install_memory_monitoring

        self._memory_monitor = install_memory_monitoring(self)

    def _install_train_step_reset_hook(self) -> None:
        """Wrap ``train_step`` so :meth:`_reset_transport_buffers` runs after each call."""
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

    def _init_wandb(self, *, num_rollouts: Optional[int] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        """Build the (rank-0/driver) wandb logger from the optional ``logging`` block."""
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

        if self._memory_monitor is not None:
            self._memory_monitor.install(self)

    def _drop_decoded(
        self,
        sample: Sample,
        *,
        rollout_id: int,
    ) -> None:
        """Upload media previews (if due this rollout) then free decoded payloads."""
        wb = self.wandb_logger
        if wb is not None and wb.should_log_media(rollout_id):
            self._upload_media_previews(sample, rollout_id + 1, prefix="rollout", step_key="rollout/step")

        for part in sample.gen_parts():
            part.primitives = {}
            part.primitive_metadata = {}
            part.media_preview = None

    def _upload_media_previews(self, sample: Sample, step: int, *, prefix: str, step_key: str) -> None:
        """Upload one preview grid per generated track of ``sample`` under ``prefix``."""
        from unirl.types.media_preview import build_media_preview_for_part
        from unirl.types.primitives import Images, Texts

        wb = self.wandb_logger
        gen_parts = sample.gen_parts()
        multi = len(gen_parts) > 1
        cond = sample.conditioning()
        default_prompts = next((list(c.texts) for c in cond if isinstance(c, Texts)), None)
        input_image = next((c for c in cond if isinstance(c, Images)), None)
        for part in gen_parts:
            name = "ar" if isinstance(part.sampling_params, ARSamplingParams) else "diffusion"
            preview = part.media_preview
            if preview is None and part.primitives:
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
            key = f"{prefix}/{name}/generated_media" if multi else f"{prefix}/generated_media"
            wb.log_generated_media(step, preview, key=key, step_key=step_key)

    def _log_eval_media(self, sample: Sample, step: int, *, prefix: str = "eval") -> None:
        """Upload the eval preview grid for one scored eval chunk."""
        wb = self.wandb_logger
        if wb is None or not wb.should_log_eval_media():
            return
        self._upload_media_previews(sample, step, prefix=prefix, step_key="eval/step")

    def _wait_for_checkpoints(self, *, timeout: Optional[float] = None) -> None:
        """Flush a pending backend checkpoint before worker teardown."""
        backend = getattr(self, "backend", None)
        if backend is None:
            return
        if timeout is None:
            backend.wait_for_checkpoint()
        else:
            backend.wait_for_checkpoint(_ray_get_timeout=timeout)

    def _cleanup_weight_sync(self, *, timeout: Optional[float] = None) -> None:
        """Let transports remove run-scoped artifacts before workers are killed."""
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
        # Bound teardown flushes so wedged workers cannot mask the primary exception.
        timeout = _TEARDOWN_FLUSH_TIMEOUT_S if active_exception else None
        try:
            self._wait_for_checkpoints(timeout=timeout)
            self._cleanup_weight_sync(timeout=timeout)
        except Exception:
            if not active_exception:
                raise
            logger.exception("Failed to flush checkpoint/weight-sync state during trainer teardown")
        finally:
            if self.wandb_logger is not None:
                self.wandb_logger.finish()

    def maybe_save_checkpoint(
        self,
        rollout_id: int,
        num_rollouts: int,
        *,
        save_interval: int,
        save_dir: Optional[str],
        save_mode: str = "auto",
    ) -> None:
        """Save every ``save_interval`` rollouts (and on the last one)."""
        if save_interval <= 0:
            return
        step = rollout_id + 1
        if step % save_interval != 0 and step < num_rollouts:
            return
        base_dir = os.path.abspath(save_dir) if save_dir else os.path.join(os.getcwd(), "checkpoints")
        path = os.path.join(base_dir, f"checkpoint-{step}")
        logger.info("Saving checkpoint at rollout %d/%d -> %s", step, num_rollouts, path)
        if self._memory_monitor is not None:
            self._memory_monitor.boundary("ckpt_save:begin", self.backend)
        self.backend.save(path, step=step, mode=save_mode)
        if self._memory_monitor is not None:
            self._memory_monitor.boundary("ckpt_save:end", self.backend)
        trainer_state_path = os.path.join(path, "trainer_state.json")
        trainer_state_tmp = f"{trainer_state_path}.tmp"
        with open(trainer_state_tmp, "w") as f:
            json.dump({"wandb_run_id": self.wandb_logger.run_id, "optimizer_step": self.wandb_logger.optimizer_step}, f)
        os.replace(trainer_state_tmp, trainer_state_path)
        if step >= num_rollouts:
            self._wait_for_checkpoints()

    def maybe_load_checkpoint(self, load_dir: Optional[str], *, num_rollouts: Optional[int] = None) -> int:
        """Restore training state from ``load_dir``; return the rollout step to resume from."""
        if not load_dir:
            return 0
        load_dir = os.path.abspath(load_dir)
        logger.info("Loading checkpoint from %s", load_dir)
        result = self.backend.load(load_dir)
        if isinstance(result, list):
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
