"""v2 full-weight IPC sync (COLOCATE, same-node)."""

from __future__ import annotations

import asyncio
import logging
import os
import socket
import threading
import time
import uuid
from datetime import timedelta
from enum import Enum
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.distributed.weight_sync.transfer.vllm_native_protocol import (
    VLLM_NATIVE_PROTOCOL_VERSION,
    VLLM_NATIVE_TRANSPORT,
    structural_manifest,
    validate_publication_result,
)

logger = logging.getLogger(__name__)


class _IPCWeightSyncEngine(Enum):
    VLLM_NATIVE_WTE = "vllm_native_wte"
    LEGACY_ZMQ = "legacy_zmq"


class IPCWeightSync(FullWeightSync):
    """Colocated IPC sync using vLLM's native WTE or legacy ZMQ."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 2048,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        use_shm: bool = False,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
        gpu_memory_headroom_mb: int = 2048,
        control_timeout_s: float = 900.0,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        if int(gpu_memory_headroom_mb) < 0:
            raise ValueError("gpu_memory_headroom_mb must be >= 0")
        control_timeout_s = float(control_timeout_s)
        if control_timeout_s <= 0:
            raise ValueError("control_timeout_s must be > 0")
        self._rollout = rollout
        self._use_shm = bool(use_shm)
        self._weight_sync_engine = self._resolve_weight_sync_engine()
        if self._weight_sync_engine is _IPCWeightSyncEngine.VLLM_NATIVE_WTE:
            receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
            timeout_for = getattr(getattr(receiver, "cfg", None), "timeout_for", None)
            if not callable(timeout_for):
                raise RuntimeError("vLLM native IPC sync requires rollout command timeouts")
            weight_update_timeout_s = float(timeout_for("update_native_weights"))
            if control_timeout_s <= weight_update_timeout_s:
                raise ValueError(
                    "control_timeout_s must exceed the vLLM weight-update timeout so non-sender "
                    f"ranks cannot time out first; got {control_timeout_s} <= {weight_update_timeout_s}"
                )
        self._gpu_memory_headroom = int(gpu_memory_headroom_mb) << 20
        self._control_timeout = timedelta(seconds=control_timeout_s)
        self._control_group = None
        self._next_model_version = 1

    def _receiver_component_name(self) -> str:
        receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
        return str(receiver.component_name())

    def _resolve_weight_sync_engine(self) -> _IPCWeightSyncEngine:
        if self._receiver_component_name() == "vllm":
            return _IPCWeightSyncEngine.VLLM_NATIVE_WTE
        return _IPCWeightSyncEngine.LEGACY_ZMQ

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def uses_gpu_streaming(self) -> bool:
        """Whether this resolved IPC engine needs Actor shards during sync."""
        return self._weight_sync_engine is _IPCWeightSyncEngine.VLLM_NATIVE_WTE

    def setup(self, transport, device, rank_info, dist_env=None, get_sibling=None) -> None:
        super().setup(
            transport=transport,
            device=device,
            rank_info=rank_info,
            dist_env=dist_env,
            get_sibling=get_sibling,
        )
        if self._weight_sync_engine is _IPCWeightSyncEngine.VLLM_NATIVE_WTE:
            if self._dist_ready():
                import torch.distributed as dist

                self._control_group = dist.new_group(backend="gloo", timeout=self._control_timeout)
            configure_sleep = getattr(self._rollout, "set_weight_sync_sleep_level", None)
            if not callable(configure_sleep):
                raise RuntimeError("vLLM native IPC sync requires configurable sleep level")
            configure_sleep(2, preserve_next_sleep=True)

    @staticmethod
    def _dist_ready() -> bool:
        try:
            import torch.distributed as dist

            return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    @property
    def _global_rank(self) -> int:
        if self._dist_ready():
            import torch.distributed as dist

            return int(dist.get_rank())
        return int(self.rank_info.rank) if self.rank_info is not None else 0

    @property
    def _world_size(self) -> int:
        if self._dist_ready():
            import torch.distributed as dist

            return int(dist.get_world_size())
        return int(self.rank_info.world_size) if self.rank_info is not None else 1

    def _all_gather(self, value: Any) -> list[Any]:
        if not self._dist_ready():
            return [value]
        import torch.distributed as dist

        values: list[Any] = [None] * self._world_size
        dist.all_gather_object(values, value, group=self._control_group)
        return values

    def _broadcast_from_rank_zero(self, value: Any) -> Any:
        if not self._dist_ready():
            return value
        import torch.distributed as dist

        values = [value if self._global_rank == 0 else None]
        dist.broadcast_object_list(values, src=0, group=self._control_group)
        return values[0]

    def _consensus(self, error: Optional[BaseException], phase: str) -> None:
        """Raise one rank-local error consistently before the next collective."""
        reports = self._all_gather(
            {
                "rank": self._global_rank,
                "phase": str(phase),
                "error": None if error is None else f"{type(error).__name__}: {error}",
            }
        )
        failures = [report for report in reports if report.get("error")]
        if failures:
            first = failures[0]
            message = f"vLLM native IPC {first['phase']} failed on rank {first['rank']}: {first['error']}"
            if error is not None:
                raise RuntimeError(message) from error
            raise RuntimeError(message)

    def _target_topology(self) -> dict[str, Any]:
        target: Optional[dict[str, Any]] = None
        error: Optional[BaseException] = None
        if self._global_rank == 0:
            try:
                receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
                fanout = int(getattr(receiver, "weight_payload_fanout", 0))
                device_uuids = list(getattr(receiver, "ipc_worker_device_uuids", ()))
                if fanout <= 0 or len(device_uuids) != fanout:
                    raise RuntimeError(f"invalid vLLM native IPC topology fanout={fanout}, UUIDs={device_uuids!r}")
                target = {"tp_world_size": fanout, "device_uuids": device_uuids}
            except BaseException as exc:
                error = exc
        self._consensus(error, "topology-discovery")
        return dict(self._broadcast_from_rank_zero(target))

    def _run_preflight(
        self,
        *,
        tp_world_size: int,
        worker_device_uuids: list[str],
        expected_metadata: list[dict[str, Any]],
    ) -> list[str]:
        import torch

        try:
            local = {
                "rank": self._global_rank,
                "host": socket.gethostname(),
                "device_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
                "error": None,
            }
        except BaseException as exc:
            local = {
                "rank": self._global_rank,
                "host": socket.gethostname(),
                "device_uuid": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        topology = self._all_gather(local)
        error: Optional[BaseException] = None
        try:
            failures = [item for item in topology if item.get("error")]
            if failures:
                raise RuntimeError(f"rank {failures[0]['rank']}: {failures[0]['error']}")
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("every Actor rank must see exactly one CUDA device")
            if self._world_size != int(tp_world_size):
                raise RuntimeError(f"Actor world size {self._world_size} != vLLM TP world {tp_world_size}")
            visible = [
                token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()
            ]
            if len(visible) != 1:
                raise RuntimeError(f"Actor rank must have one CUDA_VISIBLE_DEVICES token, got {visible!r}")
            hosts = [str(item["host"]) for item in topology]
            actor_uuids = [str(item["device_uuid"]) for item in topology]
            if len(set(hosts)) != 1:
                raise RuntimeError(f"native CUDA IPC requires one node, got hosts={hosts!r}")
            if actor_uuids != list(worker_device_uuids):
                raise RuntimeError(
                    f"Actor/vLLM physical UUID mismatch: actors={actor_uuids!r}, workers={worker_device_uuids!r}"
                )

            largest = max(int(item["nbytes"]) for item in expected_metadata)
            if largest > int(self._bucket_bytes):
                raise RuntimeError(
                    f"largest export tensor is {largest / 2**20:.1f} MiB, larger than "
                    f"native packed IPC buffer {self._bucket_bytes / 2**20:.1f} MiB"
                )
            layer_bytes: dict[str, int] = {}
            for item in expected_metadata:
                name = str(item["name"])
                if "model.layers." in name:
                    suffix = name.split("model.layers.", 1)[1]
                    owner = f"model.layers.{suffix.split('.', 1)[0]}"
                else:
                    owner = name
                layer_bytes[owner] = layer_bytes.get(owner, 0) + int(item["nbytes"])
            largest_layer = max(layer_bytes.values(), default=largest)
            required = 3 * int(self._bucket_bytes) + largest_layer + self._gpu_memory_headroom
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if int(free_bytes) < required:
                raise RuntimeError(
                    f"free GPU memory {int(free_bytes) / 2**30:.2f} GiB is below "
                    f"the conservative vLLM native IPC estimate {required / 2**30:.2f} GiB"
                )
            logger.info(
                "vLLM native IPC preflight rank=%d free=%.2f/%.2f GiB "
                "required=%.2f GiB largest_tensor=%.2f GiB largest_layer=%.2f GiB",
                self._global_rank,
                int(free_bytes) / 2**30,
                int(total_bytes) / 2**30,
                required / 2**30,
                largest / 2**30,
                largest_layer / 2**30,
            )
        except BaseException as exc:
            error = exc
        self._consensus(error, "topology-memory-preflight")
        return [str(item["device_uuid"]) for item in topology]

    @staticmethod
    def _shape_metadata(name: str, shape: tuple[int, ...], dtype: Any) -> dict[str, Any]:
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return {
            "name": str(name),
            "shape": [int(dim) for dim in shape],
            "dtype": str(dtype),
            "numel": numel,
            "nbytes": numel * int(dtype.itemsize),
        }

    def _expected_tensor_metadata(self) -> list[dict[str, Any]]:
        """Build an independent canonical Qwen3-MoE export plan."""
        model = self._backend.model
        if self._lora_merged or hasattr(model, "peft_config"):
            raise NotImplementedError("vLLM native IPC supports full fine-tuning only")
        if self._expert_weight_transform is not None:
            raise NotImplementedError("external expert export transforms are unsupported")
        if self._name_remap:
            raise NotImplementedError("Qwen3-MoE schema validation does not support name_remap")
        config = getattr(model, "config", None)
        if getattr(config, "model_type", None) != "qwen3_moe":
            raise NotImplementedError("vLLM native IPC supports qwen3_moe only")

        skip_lm_head = bool(getattr(config, "tie_word_embeddings", False))
        metadata: list[dict[str, Any]] = []
        for raw_name, tensor in model.state_dict().items():
            name = str(raw_name)
            if skip_lm_head and name == "lm_head.weight":
                continue
            dtype = self._wire_dtype if self._wire_dtype is not None and tensor.is_floating_point() else tensor.dtype
            shape = tuple(int(dim) for dim in tensor.shape)
            outputs: list[tuple[str, tuple[int, ...]]]
            if name.endswith(".mlp.experts.gate_up_proj") and len(shape) == 3:
                experts, doubled_intermediate, hidden = shape
                if doubled_intermediate % 2:
                    raise RuntimeError(f"invalid packed gate/up shape for {name!r}: {shape!r}")
                intermediate = doubled_intermediate // 2
                base = name.removesuffix(".gate_up_proj")
                outputs = [
                    (f"{base}.{expert_id}.{projection}.weight", (intermediate, hidden))
                    for expert_id in range(experts)
                    for projection in ("gate_proj", "up_proj")
                ]
            elif name.endswith(".mlp.experts.down_proj") and len(shape) == 3:
                experts, hidden, intermediate = shape
                base = name.removesuffix(".down_proj")
                outputs = [
                    (f"{base}.{expert_id}.down_proj.weight", (hidden, intermediate)) for expert_id in range(experts)
                ]
            else:
                outputs = [(name, shape)]
            planned = [self._shape_metadata(out_name, out_shape, dtype) for out_name, out_shape in outputs]
            # All canonical outputs from one state-dict entry share one FSDP
            # materialization; gate failures before advancing to the next entry.
            planned[-1]["_materialization_boundary"] = True
            metadata.extend(planned)

        names = {str(item["name"]) for item in metadata}
        required = {"model.embed_tokens.weight", "model.norm.weight"}
        if not skip_lm_head:
            required.add("lm_head.weight")
        for layer in range(int(config.num_hidden_layers)):
            prefix = f"model.layers.{layer}"
            required.update(
                {
                    f"{prefix}.self_attn.q_proj.weight",
                    f"{prefix}.self_attn.k_proj.weight",
                    f"{prefix}.self_attn.v_proj.weight",
                    f"{prefix}.self_attn.o_proj.weight",
                    f"{prefix}.mlp.gate.weight",
                    f"{prefix}.input_layernorm.weight",
                    f"{prefix}.post_attention_layernorm.weight",
                }
            )
            required.update(
                f"{prefix}.mlp.experts.{expert}.{projection}.weight"
                for expert in range(int(config.num_experts))
                for projection in ("gate_proj", "up_proj", "down_proj")
            )
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Qwen3-MoE export schema is incomplete; missing={missing[:16]}")
        structural_manifest(metadata)
        return metadata

    def _poison_rollout(self, reason: str) -> None:
        try:
            self._rollout.poison(reason=reason)
        except BaseException:
            logger.exception("Failed to poison vLLM native IPC runtime")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Select the IPC protocol from the concrete rollout engine."""
        if self._weight_sync_engine is _IPCWeightSyncEngine.VLLM_NATIVE_WTE:
            self._sync_with_vllm_native_engine()
            return
        self._sync_with_legacy_zmq()

    def _sync_with_vllm_native_engine(self) -> None:
        """Publish through vLLM's native packed-IPC weight-transfer engine."""
        import torch
        from vllm.distributed.weight_transfer import ParamMeta

        from unirl.distributed.weight_sync.transfer.vllm_native_engine import (
            VLLMNativeIPCTrainerEngine,
            VLLMNativeWeightSource,
            VLLMNativeWeightSyncClient,
        )

        started = time.perf_counter()
        expected_metadata: Optional[list[dict[str, Any]]] = None
        metadata_error: Optional[BaseException] = None
        try:
            expected_metadata = self._expected_tensor_metadata()
        except BaseException as exc:
            metadata_error = exc
        self._consensus(metadata_error, "expected-schema")
        assert expected_metadata is not None
        expected_manifest = structural_manifest(expected_metadata)

        target = self._target_topology()
        tp_world_size = int(target["tp_world_size"])
        actor_uuids = self._run_preflight(
            tp_world_size=tp_world_size,
            worker_device_uuids=list(target["device_uuids"]),
            expected_metadata=expected_metadata,
        )
        publication_id = str(self._broadcast_from_rank_zero(uuid.uuid4().hex if self._global_rank == 0 else None))
        header = {
            "protocol_version": VLLM_NATIVE_PROTOCOL_VERSION,
            "transport": VLLM_NATIVE_TRANSPORT,
            "publication_id": publication_id,
            "model_version": int(self._next_model_version),
            "tp_world_size": tp_world_size,
            "expected_manifest": expected_manifest,
            "device_uuids": actor_uuids,
        }

        source = VLLMNativeWeightSource(
            metadata=[
                ParamMeta(
                    name=str(item["name"]),
                    dtype=getattr(torch, str(item["dtype"]).removeprefix("torch.")),
                    shape=tuple(int(dim) for dim in item["shape"]),
                )
                for item in expected_metadata
            ],
            iterator_factory=lambda: self._iter_full_tensors(),
        )
        client = VLLMNativeWeightSyncClient(
            rollout=self._rollout,
            header=header,
            flush_cache=self._flush_cache,
        )
        engine = VLLMNativeIPCTrainerEngine(
            client=client,
            source=source,
            rank=self._global_rank,
            packed_buffer_size_bytes=self._bucket_bytes,
            consensus=self._consensus,
            materialization_boundaries=[
                index for index, item in enumerate(expected_metadata) if item.get("_materialization_boundary") is True
            ],
        )

        def _verify_actor_source() -> None:
            actual = structural_manifest(source.actual_metadata)
            reports = self._all_gather({"rank": self._global_rank, "manifest": actual})
            mismatches = [report for report in reports if report["manifest"] != expected_manifest]
            if mismatches:
                raise RuntimeError(
                    f"vLLM native IPC Actor manifest mismatch: expected={expected_manifest!r}, actual={mismatches!r}"
                )

        success = False
        try:
            engine.initialize()
            engine.send_weights(
                weight_version=str(self._next_model_version),
                before_finish=_verify_actor_source,
            )
            release_error: Optional[BaseException] = None
            if self._global_rank == 0:
                try:
                    client.release_ipc_imports()
                except BaseException as exc:
                    release_error = exc
            self._consensus(release_error, "native-ipc-release")
            result = client.last_result if self._global_rank == 0 else None
            result = self._broadcast_from_rank_zero(result)
            validate_publication_result(
                header,
                result,
                fanout=tp_world_size,
                flush_cache=self._flush_cache,
            )
            self._next_model_version += 1
            success = True
            logger.info(
                "vLLM native IPC TP%d committed version=%d tensors=%d bytes=%d elapsed=%.2fs",
                tp_world_size,
                int(header["model_version"]),
                int(expected_manifest["tensor_count"]),
                int(expected_manifest["byte_count"]),
                time.perf_counter() - started,
            )
        finally:
            if not success and self._global_rank == 0:
                self._poison_rollout(f"vLLM native IPC publication {publication_id} failed")
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _sync_with_legacy_zmq(self) -> None:
        """Pump full weights to the co-located engine over per-stage sockets."""
        rank_info = getattr(self, "rank_info", None)
        if int(getattr(rank_info, "tp_size", 1) or 1) > 1:
            raise NotImplementedError(
                "IPCWeightSync does not support tp_size>1: it opens only the "
                "local_rank=0 receiver socket. Use CkptEngineIPCWeightSync for "
                "SGLang TP or keep this vLLM-Omni path at tp_size=1."
            )

        from unirl.distributed.weight_sync.transfer.bucketed_transfer import (
            BucketedWeightSender,
        )
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import zmq_handle

        replica_rank = self._my_rank  # distinct per colocate engine → unique socket

        try:
            tp_per_stage = {int(stage_id): int(tp_size) for stage_id, tp_size in self._rollout.tp_per_stage().items()}
        except (AttributeError, NotImplementedError):
            tp_per_stage = {0: 1}
        if not tp_per_stage:
            tp_per_stage = {0: 1}
        unsupported_tp = {stage_id: tp_size for stage_id, tp_size in tp_per_stage.items() if tp_size > 1}
        if unsupported_tp:
            raise NotImplementedError(
                "IPCWeightSync does not support tp_size>1 because it only sends "
                f"to local_rank=0; stage TP layout={unsupported_tp}."
            )
        stage_ids = sorted(tp_per_stage)

        recv_error: dict = {}

        def _spawn_receivers() -> None:
            try:
                self._rollout.update_weights_from_ipc(
                    peft_config=None,
                    base_sync_done=False,
                    use_shm=self._use_shm,
                    replica_rank=replica_rank,
                    track_prefix=self._track_prefix,
                )
            except Exception as exc:  # surface, don't let the pump hang forever
                recv_error["exc"] = exc

        thread = threading.Thread(target=_spawn_receivers, daemon=True)
        thread.start()
        try:
            for sid in stage_ids:
                handle = zmq_handle(replica_rank=replica_rank, stage_id=int(sid), local_rank=0)
                sender = BucketedWeightSender(
                    zmq_handle=handle,
                    bucket_size_mb=self._bucket_bytes // (1024 * 1024),
                    use_shm=self._use_shm,
                )
                asyncio.run(sender.async_send_weights(self._iter_full_tensors()))
        finally:
            thread.join()
        if "exc" in recv_error:
            raise RuntimeError("IPCWeightSync: rollout receiver failed") from recv_error["exc"]

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def cleanup(self) -> None:
        self._release_resources()

    def shutdown(self) -> None:
        self._release_resources()

    def _release_resources(self) -> None:
        group, self._control_group = self._control_group, None
        if group is not None:
            import torch.distributed as dist

            try:
                dist.destroy_process_group(group)
            except Exception:
                pass


__all__ = ["IPCWeightSync"]
