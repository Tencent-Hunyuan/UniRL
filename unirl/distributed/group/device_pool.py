"""DevicePool — global GPU device pool.

Creates N Worker Ray actors (one per GPU), auto-connects NCCL.
Uses one STRICT_PACK PlacementGroup per node (devices_per_node bundles each),
guaranteeing intra-node locality while supporting multi-node clusters.
Handles request GPU slots from DevicePool via create_remote().

workers_per_device controls how many Worker processes share each physical GPU:
  1 (default): one Worker per GPU, backward-compatible
  2:           slot0 = primary (NCCL), slot1 = colocated (IPC only, lazy-created)

Slot1+ workers are created lazily on first create_remote(slot_id=1) call.
PlacementGroup bundles reserve CPU quota for all slots upfront.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from unirl.distributed.group.handle import Handle
from unirl.distributed.group.worker import Worker
from unirl.distributed.utils import get_node_ip_and_port

logger = logging.getLogger(__name__)


class DevicePool:
    """Global GPU device pool for an RL training run.

    Creates one STRICT_PACK PlacementGroup per node (devices_per_node cards each),
    so intra-node devices are always co-located while multi-node placement works.

    Example — 64 devices, 8 per node (8 machines):
        pool = DevicePool(num_devices=64, devices_per_node=8)
        pool.setup()
        # → 8 STRICT_PACK PGs of 8 bundles each

    Device ids are global (0-63); mapping is:
        node        = device_id // devices_per_node
        bundle      = device_id %  devices_per_node
    """

    def __init__(
        self,
        num_devices: int,
        devices_per_node: int = 8,
        workers_per_device: int = 1,
        transport_kind: str = "colocate_store",
        tq_handoff: Optional[dict] = None,
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
        # Driver's TransferQueue actor handoff; required when transport_kind is
        # transfer_queue (fanned to each Worker before it builds its transport).
        self.tq_handoff = tq_handoff

        # slot0 workers indexed by device_id (backward-compatible)
        self.workers: List[ray.actor.ActorHandle] = []

        # Per-GPU TensorWorker actors (gpu_store backend only), keyed by device_id.
        self._tw_by_device: Dict[int, Any] = {}

        # MASTER_ADDR/PORT forwarded to TensorWorker actors so their cross-GPU NCCL
        # PG can bind. Set in _create_workers; mirrors the slot0 Worker env injection.
        self._master_addr: Optional[str] = None
        self._master_port: Optional[str] = None

        self._pgs: List = []
        self._next_device: int = 0

        # Devices reserved by a top-level placement scope. Subsequent sibling
        # scopes carve from the complement; nested scopes carve from their
        # parent's slab and do not touch this set.
        self._claimed: Set[int] = set()

        # Internal mappings (private — use public methods from outside)
        self._worker_id_to_device_id: Dict[str, int] = {}
        self._worker_id_to_slot: Dict[str, int] = {}
        self._device_to_workers: Dict[int, List] = {}  # device_id → [slot0, slot1, ...]
        self._worker_by_id: Dict[str, Any] = {}  # worker_id → handle

    # Keep num_gpus as a read-only alias for backward compatibility
    @property
    def num_gpus(self) -> int:
        return self.num_devices

    @property
    def transport_cls(self) -> type:
        """The TensorTransport subclass for the configured kind (no live instance).

        The controller (Handle) reads class-level policy off this — ``localize``
        (locality + cross-worker transfer) and, via ``issubclass(..,
        WorkerLocalTransport)``, whether to register decref GC. Worker-local
        backends each implement ``localize``; GLOBAL (transfer_queue) inherits
        the identity ``localize``.
        """
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
        # GLOBAL transports (transfer_queue) resolve refs from any process and have no
        # cross-worker NCCL; only worker-local backends (colocate/gpu) need the global PG.
        if self.num_devices > 1 and issubclass(self.transport_cls, WorkerLocalTransport):
            self._setup_nccl()

    def _create_placement_groups(self) -> None:
        """Create one STRICT_PACK PlacementGroup per node.

        Each bundle reserves CPU quota for all slots on that GPU upfront,
        even though slot1+ workers are created lazily.
        """
        num_nodes = self.num_devices // self.devices_per_node
        # gpu_store adds one CPU per bundle for the per-GPU TensorWorker actor.
        extra_cpu = 1 if self.transport_kind in ("gpu_store", "gpu") else 0
        bundles = [{"GPU": 1, "CPU": self.workers_per_device + extra_cpu} for _ in range(self.devices_per_node)]
        pgs = [placement_group(bundles, strategy="STRICT_PACK") for _ in range(num_nodes)]
        ray.get([pg.ready() for pg in pgs])
        self._pgs = pgs

    def _create_workers(self) -> None:
        """Create slot0 Worker per device. Slot1+ are created lazily.

        MASTER_ADDR/PORT are resolved from bundle 0 of PG 0 (node 0).
        """
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

        # gpu_store defers its transport build: the shared per-GPU TensorWorker is
        # created after the Worker exists, so inject it then build. colocate and
        # transfer_queue build in Worker.__init__ (deps ready at construction).
        if self.transport_kind in ("gpu_store", "gpu"):
            tw = self._get_or_create_tw(device_id)
            ray.get(w.set_tensor_worker.remote(tw))
            ray.get(w.build_and_install_transport.remote())
        return w

    def _get_or_create_tw(self, device_id: int) -> ray.actor.ActorHandle:
        """Create (once) the per-GPU TensorWorker actor for the gpu_store backend.

        One TensorWorker per physical GPU, shared by all slots on that GPU. Pinned
        to the device's PG bundle (num_gpus=0; the bundle's GPU is already reserved
        by the Worker fractions and shared via CUDA_VISIBLE_DEVICES).
        """
        tw = self._tw_by_device.get(device_id)
        if tw is not None:
            return tw
        from unirl.distributed.tensor.backend.gpu_store.worker import TensorWorker

        pg = self._pgs[device_id // self.devices_per_node]
        bundle_index = device_id % self.devices_per_node
        # The TensorWorker shares the bundle's physical GPU with the slot0 Worker(s).
        # With num_gpus=0 Ray hides all GPUs from the actor (CUDA_VISIBLE_DEVICES="");
        # disable that override and mirror the slot0 Worker's CUDA_VISIBLE_DEVICES so the
        # TW sees exactly that one GPU as cuda:0. MASTER_ADDR/PORT let its cross-GPU NCCL
        # ProcessGroup bind (slot0 Workers get these via _spawn_worker's env_vars).
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
        """Return the worker for (device_id, slot), creating it lazily if needed.

        Slots must be created in order (0, 1, 2, ...) to keep
        _device_to_workers[device_id] indices consistent.
        """
        workers = self._device_to_workers[device_id]
        if slot < len(workers):
            return workers[slot]
        assert slot == len(workers), (
            f"Must create slots in order: device={device_id} has {len(workers)} slot(s), requested slot={slot}"
        )
        assert slot < self.workers_per_device, f"slot={slot} >= workers_per_device={self.workers_per_device}"
        return self._spawn_worker(device_id, slot)

    def _setup_nccl(self) -> None:
        """Initialize NCCL ProcessGroup on all slot0 workers.

        Always uses ProcessGroupNCCL directly (never dist.init_process_group)
        so the global dist state is not polluted by the controller's NCCL PG.
        """
        slot0_workers = [self._device_to_workers[d][0] for d in range(self.num_devices)]
        ray.get([w.setup_global_pg.remote() for w in slot0_workers])

    # ── Public interface ──

    def slot0_worker(self, device_id: int) -> ray.actor.ActorHandle:
        """Return the slot0 worker handle for device_id."""
        return self._device_to_workers[device_id][0]

    def get_workers(self, device_ids: List[int], slot: int = 0) -> List[ray.actor.ActorHandle]:
        """Return worker handles for each device_id at the given slot.

        The slot must already exist (created via create_remote or _get_or_create_worker).
        """
        return [self._device_to_workers[d][slot] for d in device_ids]

    def all_workers(self) -> List[ray.actor.ActorHandle]:
        """Every created worker handle across all slots (slot0 + lazily-created slot1+).

        ``self.workers`` is slot0-only; multi-slot transfer_queue workers each hold their
        own TQ client, so per-process fan-outs (e.g. buffer reclaim) must reach every slot.
        """
        return [w for workers in self._device_to_workers.values() for w in workers]

    def reset_transfer_queue_buffers(self) -> None:
        """Reclaim mooncake zero-copy buffer free-lists across workers + driver.

        Called once per rollout at a quiescent boundary. No-op unless the active backend
        is transfer_queue. The worker fan-out (``reset_actors_zero_copy_buffer_free``)
        no-ops internally for non-mooncake backends, and the driver's own reset is gated on
        the mooncake manager_type here — so this is safe to call unconditionally (e.g. for
        SimpleBackend runs, which register no zero-copy buffers).
        """
        if self.transport_kind not in ("transfer_queue", "tq"):
            return
        from unirl.distributed.tensor.backend.transfer_queue.runtime import TransferQueueRuntime

        rt = TransferQueueRuntime.current()
        if rt is None:
            return
        rt.reset_actors_zero_copy_buffer_free(self.all_workers())
        if rt.backend is not None and rt.backend.manager_type == "MooncakeStorageManager":
            rt.reset_zero_copy_buffer_free()  # the driver client's own registered buffers

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
        """Create a Handle for role_cls on this pool.

        Inside a ``placement(...)`` block, ``device_ids`` and ``slot_id`` are
        sourced from the active scope; users typically pass nothing but
        ``role_cls``. Outside a scope, ``device_ids`` or ``n_gpus`` must be
        provided.

        Args:
            role_cls:      Remote subclass to register.
            device_ids:    Explicit GPU indices. Overrides the active scope.
            n_gpus:        Auto-allocate this many GPUs sequentially. Used
                           only when ``device_ids`` is None and no scope is
                           active.
            role_name:     Optional name. If None, auto-generated.
            init_kwargs: Dict of kwargs forwarded to role_cls.__init__.
            slot_id:       Worker slot on each device. Defaults to the active
                           scope's choice, or 0 when no scope is active.

        Returns:
            Handle bound to this pool.
        """
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
                # Implicit defaults: full pool, slot 0. Bare back-to-back
                # create_remote calls colocate (shared workers). Does NOT
                # touch _claimed — bare calls opt out of scope tracking.
                device_ids = list(range(self.num_devices))
        if slot_id is None:
            slot_id = 0

        # Ensure slot workers exist for all requested devices
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
        """Kill all Worker actors and remove PlacementGroups."""
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
