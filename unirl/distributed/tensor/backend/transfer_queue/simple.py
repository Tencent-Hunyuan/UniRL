"""AsyncSimpleStorageManager backend: in-memory storage units fan-out via Ray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from unirl.config.require import require
from unirl.distributed.tensor.backend.transfer_queue.base import Backend

if TYPE_CHECKING:
    from ray.actor import ActorHandle


@dataclass
class SimpleBackendConfig:
    """In-memory async storage backend configuration.

    Defaults are sized for typical SD3-GRPO single-step volume; override only
    if you've measured a need. For production-tunable sizing, use Mooncake.
    """

    num_units: int = 16  # in-process Ray storage actors
    unit_size: int = 1024  # per-unit item capacity

    def __post_init__(self) -> None:
        require(
            self.num_units > 0,
            f"SimpleBackendConfig.num_units must be > 0; got {self.num_units!r}",
        )
        require(
            self.unit_size > 0,
            f"SimpleBackendConfig.unit_size must be > 0; got {self.unit_size!r}",
        )


class SimpleBackend(Backend):
    """Drives ``AsyncSimpleStorageManager`` storage-unit fan-out."""

    manager_type = "AsyncSimpleStorageManager"

    def __init__(self, *, num_units: int, unit_size: int) -> None:
        self._num_units = int(num_units)
        self._unit_size = int(unit_size)
        self._storage_units: Dict[int, "ActorHandle"] = {}

    def bootstrap(self, *, controller_info: Any) -> dict:
        # Deferred upstream import: schema registration must work without the
        # transfer_queue runtime lib (compose tests don't need it).
        from transfer_queue import SimpleStorageUnit, get_placement_group, process_zmq_server_info

        placement_group = get_placement_group(self._num_units, num_cpus_per_actor=1)
        for rank in range(self._num_units):
            self._storage_units[rank] = SimpleStorageUnit.options(
                placement_group=placement_group,
                placement_group_bundle_index=rank,
            ).remote(storage_unit_size=self._unit_size)

        return {
            "manager_type": self.manager_type,
            "controller_info": controller_info,
            "storage_unit_infos": process_zmq_server_info(self._storage_units),
        }


__all__ = ["SimpleBackend", "SimpleBackendConfig"]
