"""MooncakeStorageManager backend: zero-copy RDMA via Mooncake."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from unirl.config.require import require
from unirl.distributed.tensor.backend.transfer_queue.base import Backend


@dataclass
class MooncakeZeroCopyConfig:
    """Zero-copy buffer sizing for the Mooncake backend.

    ``single_controller_*`` fields override ``tensor_buffer_size_gb`` /
    ``bytes_buffer_size_gb`` on the controller-only client (see
    ``MooncakeBackend.specialize_for_controller``).
    """

    enable: bool = True
    tensor_buffer_size_gb: float = 2.0
    bytes_buffer_size_gb: float = 2.0
    single_controller_tensor_buffer_size_gb: float = 10.0
    single_controller_bytes_buffer_size_gb: float = 10.0
    manager_merge_to_tensordict: bool = False


@dataclass
class MooncakeBackendConfig:
    """Mooncake storage backend configuration (with zero-copy embedded)."""

    metadata_server: str = ""
    master_server_address: str = ""
    global_segment_size_gb: int = 64
    local_buffer_size_gb: int = 10
    client_name: str = "MooncakeStorageClient"
    protocol: str = "rdma"
    # ``device_name`` is auto-discovered at runtime by ``runtime.create_client``
    # (sysfs walk → comma-separated HCA list → Mooncake's per-process PIX
    # selection via ``MC_ENABLE_DEST_DEVICE_AFFINITY=1``). Override at the CLI
    # with ``transfer_queue.device_name=<name>`` only for ops debugging.
    device_name: Optional[str] = None
    zero_copy: MooncakeZeroCopyConfig = field(default_factory=MooncakeZeroCopyConfig)

    def __post_init__(self) -> None:
        require(
            bool(self.metadata_server),
            "MooncakeBackendConfig.metadata_server must be set "
            "(e.g. 'http://<head_ip>:<port>/metadata') — points at the "
            "external mooncake_master HTTP metadata endpoint",
        )
        require(
            bool(self.master_server_address),
            "MooncakeBackendConfig.master_server_address must be set "
            "(e.g. '<head_ip>:<rpc_port>') — points at the external "
            "mooncake_master RPC endpoint",
        )
        require(
            self.protocol in ("rdma", "tcp"),
            f"MooncakeBackendConfig.protocol must be 'rdma' or 'tcp'; got {self.protocol!r}",
        )
        require(
            self.global_segment_size_gb > 0,
            f"MooncakeBackendConfig.global_segment_size_gb must be > 0; got {self.global_segment_size_gb!r}",
        )
        require(
            self.local_buffer_size_gb > 0,
            f"MooncakeBackendConfig.local_buffer_size_gb must be > 0; got {self.local_buffer_size_gb!r}",
        )


def _zero_copy_to_dict(zero_copy: Any) -> Dict[str, Any]:
    """Pull zero_copy fields into a plain dict regardless of source shape.

    ``hydra.utils.instantiate`` may pass a DictConfig, a dataclass, or a plain
    dict depending on conversion settings; attribute access works on all three.
    """
    return {
        "enable": bool(zero_copy.enable),
        "tensor_buffer_size_gb": float(zero_copy.tensor_buffer_size_gb),
        "bytes_buffer_size_gb": float(zero_copy.bytes_buffer_size_gb),
        "single_controller_tensor_buffer_size_gb": float(zero_copy.single_controller_tensor_buffer_size_gb),
        "single_controller_bytes_buffer_size_gb": float(zero_copy.single_controller_bytes_buffer_size_gb),
        "manager_merge_to_tensordict": bool(zero_copy.manager_merge_to_tensordict),
    }


class MooncakeBackend(Backend):
    """Pure-client Mooncake backend; storage segments live on the upstream service."""

    manager_type = "MooncakeStorageManager"

    def __init__(
        self,
        *,
        metadata_server: str,
        master_server_address: str,
        global_segment_size_gb: int,
        local_buffer_size_gb: int,
        client_name: str,
        protocol: str,
        device_name: Optional[str],
        zero_copy: Any,
    ) -> None:
        self._metadata_server = str(metadata_server)
        self._master_server_address = str(master_server_address)
        self._global_segment_size_gb = int(global_segment_size_gb)
        self._local_buffer_size_gb = int(local_buffer_size_gb)
        self._client_name = str(client_name)
        self._protocol = str(protocol)
        # ``None`` flows through to the handoff; ``runtime.create_client``
        # substitutes a comma-list of available HCAs at process init.
        self._device_name = device_name if device_name is None else str(device_name)
        self._zero_copy = _zero_copy_to_dict(zero_copy)

    def bootstrap(self, *, controller_info: Any) -> dict:
        return {
            "manager_type": self.manager_type,
            "controller_info": controller_info,
            "metadata_server": self._metadata_server,
            "master_server_address": self._master_server_address,
            "global_segment_size_gb": self._global_segment_size_gb,
            "local_buffer_size_gb": self._local_buffer_size_gb,
            "client_name": self._client_name,
            "protocol": self._protocol,
            "device_name": self._device_name,
            "zero_copy": dict(self._zero_copy),
            # Pure-client mode: derived byte counts the upstream library reads.
            "global_segment_size": int(self._global_segment_size_gb * 1024**3),
            "local_buffer_size": int(self._local_buffer_size_gb * 1024**3),
        }

    def specialize_for_controller(self, actor_handoff: dict) -> dict:
        handoff = dict(actor_handoff)
        zc = dict(actor_handoff["zero_copy"])
        zc["tensor_buffer_size_gb"] = zc["single_controller_tensor_buffer_size_gb"]
        zc["bytes_buffer_size_gb"] = zc["single_controller_bytes_buffer_size_gb"]
        handoff["zero_copy"] = zc
        return handoff


__all__ = ["MooncakeBackend", "MooncakeBackendConfig", "MooncakeZeroCopyConfig"]
