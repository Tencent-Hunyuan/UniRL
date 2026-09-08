"""DevicePool — global GPU device pool."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Set

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from unirl.distributed.group.handle import Handle
from unirl.distributed.group.worker import Worker
from unirl.distributed.utils import get_node_ip_and_port

logger = logging.getLogger(__name__)

_ROLE_TEARDOWN_TIMEOUT_S = 45.0


class DevicePool:
    """Global GPU device pool for an RL training run."""

    def __init__(
        self,
        num_devices: int,
        devices_per_node: int = 8,
        workers_per_device: int = 1,
        transport_kind: str = "colocate_store",
        tq_handoff: Optional[dict] = None,
        worker_max_concurrency: int | Sequence[int] = 1,
    ) -> None:
        if num_devices % devices_per_node != 0:
            raise ValueError(f"num_devices ({num_devices}) must be divisible by devices_per_node ({devices_per_node})")
        self.num_devices = num_devices
        self.devices_per_node = devices_per_node
        self.workers_per_device = workers_per_device
        self.transport_kind = transport_kind or "colocate_store"
        if self.transport_kind in ("colocate_store", "colocate") and self.workers_per_device != 1:
            raise ValueError(
                f"colocate_store supports one worker per device (got workers_per_device="
                f"{self.workers_per_device}); use transport_kind='gpu_store' for colocated multi-slot."
            )
        self.tq_handoff = tq_handoff

        if isinstance(worker_max_concurrency, int):
            per_device_concurrency = [max(1, worker_max_concurrency)] * num_devices
        else:
            per_device_concurrency = [max(1, value) for value in worker_max_concurrency]
            if len(per_device_concurrency) != num_devices:
                raise ValueError(
                    "worker_max_concurrency must provide one value per device: "
                    f"{len(per_device_concurrency)} != {num_devices}"
                )
        self._worker_concurrency_by_device = per_device_concurrency
        self.max_worker_concurrency = max(per_device_concurrency)

        self.workers: List[ray.actor.ActorHandle] = []

        self._tw_by_device: Dict[int, Any] = {}

        self._master_addr: Optional[str] = None
        self._master_port: Optional[str] = None

        self._pgs: List = []
        self._next_device: int = 0

        self._claimed: Set[int] = set()

        self._worker_id_to_device_id: Dict[str, int] = {}
        self._worker_id_to_slot: Dict[str, int] = {}
        self._device_to_workers: Dict[int, List] = {}  # device_id → [slot0, slot1, ...]
        self._worker_by_id: Dict[str, Any] = {}  # worker_id → handle

    @property
    def num_gpus(self) -> int:
        return self.num_devices

    @property
    def transport_cls(self) -> type:
        """The TensorTransport subclass for the configured kind (no live instance)."""
        kind = self.transport_kind
        if kind in ("colocate_store", "colocate"):
            from unirl.distributed.tensor.backend.colocate_store.transport import ColocateStoreTransport

            return ColocateStoreTransport
        if kind in ("gpu_store", "gpu"):
            from unirl.distributed.tensor.backend.gpu_store.transport import GPUStoreTransport

            return GPUStoreTransport
        if kind in ("transfer_queue", "tq"):
            from unirl.distributed.tensor.backend.transfer_queue.transport import TQTransport

            return TQTransport
        raise ValueError(f"unknown transport kind {kind!r}")

    def setup(self) -> None:
        """Create PlacementGroups, Worker actors, and (for worker-local backends) NCCL."""
        from unirl.distributed.tensor import WorkerLocalTransport

        self._create_placement_groups()
        self._create_workers()
        if self.num_devices > 1 and issubclass(self.transport_cls, WorkerLocalTransport):
            self._setup_nccl()

    def _create_placement_groups(self) -> None:
        """Create one STRICT_PACK PlacementGroup per node."""
        num_nodes = self.num_devices // self.devices_per_node
        extra_cpu = 1 if self.transport_kind in ("gpu_store", "gpu") else 0
        bundles = [{"GPU": 1, "CPU": self.workers_per_device + extra_cpu} for _ in range(self.devices_per_node)]
        pgs = [placement_group(bundles, strategy="STRICT_PACK") for _ in range(num_nodes)]
        ray.get([pg.ready() for pg in pgs])
        self._pgs = pgs

    def _create_workers(self) -> None:
        """Create slot0 Worker per device. Slot1+ are created lazily."""
        master_addr, master_port = get_node_ip_and_port(self._pgs[0], bundle_index=0)
        self._master_addr, self._master_port = master_addr, str(master_port)
        env_vars_base = {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "WORLD_SIZE": str(self.num_devices),
        }
        for device_id in range(self.num_devices):
            self._device_to_workers[device_id] = []
            env_vars = {**env_vars_base, "RANK": str(device_id)}
            w = self._spawn_worker(device_id, slot=0, env_vars=env_vars)
            self.workers.append(w)

    def _spawn_worker(self, device_id: int, slot: int, env_vars: dict = None) -> ray.actor.ActorHandle:
        """Spawn a Worker actor and register it in internal mappings."""
        if self.transport_kind in ("transfer_queue", "tq") and self.tq_handoff is None:
            raise RuntimeError(
                "transport_kind='transfer_queue' requires tq_handoff "
                "(the driver's TransferQueueRuntime.init() actor handoff)."
            )
        worker_id = f"dw{device_id}" if slot == 0 else f"dw{device_id}_s{slot}"
        pg = self._pgs[device_id // self.devices_per_node]
        bundle_index = device_id % self.devices_per_node
        num_gpus = 1 / self.workers_per_device

        options = dict(
            num_gpus=num_gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_index,
            ),
        )
        max_concurrency = self._worker_concurrency_by_device[device_id]
        if max_concurrency > 1:
            options["max_concurrency"] = max_concurrency
        if env_vars:
            options["runtime_env"] = {"env_vars": env_vars}

        w = (
            ray.remote(Worker)
            .options(**options)
            .remote(
                device_id=device_id,
                slot=slot,
                nccl_rank=device_id if slot == 0 else None,
                world_size=self.num_devices,
                transport_kind=self.transport_kind,
                tq_handoff=self.tq_handoff,
            )
        )
        self._device_to_workers[device_id].append(w)
        self._worker_by_id[worker_id] = w
        self._worker_id_to_device_id[worker_id] = device_id
        self._worker_id_to_slot[worker_id] = slot

        if self.transport_kind in ("gpu_store", "gpu"):
            tw = self._get_or_create_tw(device_id)
            ray.get(w.set_tensor_worker.remote(tw))
            ray.get(w.build_and_install_transport.remote())
        return w

    def _get_or_create_tw(self, device_id: int) -> ray.actor.ActorHandle:
        """Create (once) the per-GPU TensorWorker actor for the gpu_store backend."""
        tw = self._tw_by_device.get(device_id)
        if tw is not None:
            return tw
        from unirl.distributed.tensor.backend.gpu_store.worker import TensorWorker

        pg = self._pgs[device_id // self.devices_per_node]
        bundle_index = device_id % self.devices_per_node
        cvd = ray.get(self.slot0_worker(device_id).get_cuda_visible_devices.remote())
        tw = (
            ray.remote(TensorWorker)
            .options(
                num_gpus=0,
                runtime_env={
                    "env_vars": {
                        "MASTER_ADDR": self._master_addr,
                        "MASTER_PORT": self._master_port,
                        "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                        "CUDA_VISIBLE_DEVICES": cvd,
                    }
                },
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=bundle_index,
                ),
            )
            .remote(device_id=device_id)
        )
        self._tw_by_device[device_id] = tw
        return tw

    def _get_or_create_worker(self, device_id: int, slot: int) -> ray.actor.ActorHandle:
        """Return the worker for (device_id, slot), creating it lazily if needed."""
        workers = self._device_to_workers[device_id]
        if slot < len(workers):
            return workers[slot]
        assert slot == len(workers), (
            f"Must create slots in order: device={device_id} has {len(workers)} slot(s), requested slot={slot}"
        )
        assert slot < self.workers_per_device, f"slot={slot} >= workers_per_device={self.workers_per_device}"
        return self._spawn_worker(device_id, slot)

    def _setup_nccl(self) -> None:
        """Initialize NCCL ProcessGroup on all slot0 workers."""
        slot0_workers = [self._device_to_workers[d][0] for d in range(self.num_devices)]
        ray.get([w.setup_global_pg.remote() for w in slot0_workers])

    def slot0_worker(self, device_id: int) -> ray.actor.ActorHandle:
        """Return the slot0 worker handle for device_id."""
        return self._device_to_workers[device_id][0]

    def get_workers(self, device_ids: List[int], slot: int = 0) -> List[ray.actor.ActorHandle]:
        """Return worker handles for each device_id at the given slot."""
        return [self._device_to_workers[d][slot] for d in device_ids]

    def all_workers(self) -> List[ray.actor.ActorHandle]:
        """Every created worker handle across all slots (slot0 + lazily-created slot1+)."""
        return [w for workers in self._device_to_workers.values() for w in workers]

    def reset_transfer_queue_buffers(self) -> None:
        """Reclaim mooncake zero-copy buffer free-lists across workers + driver."""
        if self.transport_kind not in ("transfer_queue", "tq"):
            return
        from unirl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime

        rt = TransferQueueRuntime.current()
        if rt is None:
            return
        rt.reset_actors_zero_copy_buffer_free(self.all_workers())
        if rt.backend is not None and rt.backend.manager_type == "MooncakeStorageManager":
            rt.reset_zero_copy_buffer_free()

    def get_worker(self, worker_id: str) -> ray.actor.ActorHandle:
        """Return the worker handle for the given worker_id."""
        try:
            return self._worker_by_id[worker_id]
        except KeyError:
            raise KeyError(f"Unknown worker_id '{worker_id}'. Known: {sorted(self._worker_by_id)}")

    def device_id_of(self, worker_id: str) -> int:
        """Return the device_id for a worker_id (e.g. 'dw3' → 3, 'dw3_s1' → 3)."""
        try:
            return self._worker_id_to_device_id[worker_id]
        except KeyError:
            raise KeyError(f"Unknown worker_id '{worker_id}'. Known: {sorted(self._worker_id_to_device_id)}")

    def slot_of(self, worker_id: str) -> int:
        """Return the slot for a worker_id (e.g. 'dw3' → 0, 'dw3_s1' → 1)."""
        try:
            return self._worker_id_to_slot[worker_id]
        except KeyError:
            raise KeyError(f"Unknown worker_id '{worker_id}'.")

    def allocate(self, n: int) -> List[int]:
        """Auto-allocate n devices sequentially. Returns device_ids."""
        if self._next_device + n > self.num_devices:
            raise ValueError(
                f"Cannot allocate {n} devices: only "
                f"{self.num_devices - self._next_device} remaining "
                f"(total={self.num_devices}, allocated={self._next_device})"
            )
        ids = list(range(self._next_device, self._next_device + n))
        self._next_device += n
        return ids

    def create_remote(
        self,
        role_cls,
        device_ids=None,
        n_gpus: int = None,
        role_name: str = None,
        init_kwargs: dict = None,
        slot_id: Optional[int] = None,
    ) -> Handle:
        """Create a Handle for role_cls on this pool."""
        from unirl.distributed.group.placement import current_placement

        if device_ids is None:
            scope = current_placement()
            if scope is not None:
                device_ids, auto_slot = scope.assign()
                if slot_id is None:
                    slot_id = auto_slot
            elif n_gpus is not None:
                device_ids = self.allocate(n_gpus)
            else:
                device_ids = list(range(self.num_devices))
        if slot_id is None:
            slot_id = 0

        for d in list(device_ids):
            self._get_or_create_worker(d, slot_id)

        return Handle(
            role_cls,
            self,
            device_ids=device_ids,
            slot_id=slot_id,
            role_name=role_name,
            init_kwargs=init_kwargs,
        )

    def shutdown(self) -> None:
        """Release roles, then kill all Worker actors and remove PlacementGroups."""
        self._release_roles()
        for tw in self._tw_by_device.values():
            ray.kill(tw, no_restart=True)
        for w in self._worker_by_id.values():
            ray.kill(w, no_restart=True)
        for w in self._worker_by_id.values():
            try:
                ray.get(w.get_gpu_count.remote(), timeout=10)
            except Exception:
                logger.debug("Worker did not respond after ray.kill during DevicePool shutdown.", exc_info=True)
        self.workers.clear()
        self._worker_by_id.clear()
        self._device_to_workers.clear()
        self._worker_id_to_device_id.clear()
        self._worker_id_to_slot.clear()
        self._tw_by_device.clear()
        self._next_device = 0
        self._claimed.clear()
        for pg in self._pgs:
            ray.util.remove_placement_group(pg)
        self._pgs = []

    def _release_roles(self) -> None:
        """Give every worker a chance to close its roles before we kill it."""
        if not self._worker_by_id:
            return
        pending = {wid: w.teardown.remote() for wid, w in self._worker_by_id.items()}
        deadline = time.monotonic() + _ROLE_TEARDOWN_TIMEOUT_S
        for wid, ref in pending.items():
            try:
                ray.get(ref, timeout=max(0.0, deadline - time.monotonic()))
            except Exception:
                logger.warning("Worker %s did not release its roles before shutdown; killing it", wid)
