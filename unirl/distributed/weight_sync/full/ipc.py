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
)

logger = logging.getLogger(__name__)


def _apply_name_remap(name: str, rules: Dict[str, Optional[str]]) -> Optional[str]:
    """Apply FullWeightSync's ordered first-match name rewrite locally."""
    for key, value in rules.items():
        prefix, _, suffix = key.partition("*")
        if not (name.startswith(prefix) and name.endswith(suffix) and len(name) >= len(prefix) + len(suffix)):
            continue
        if value is None:
            return None
        value_prefix, _, value_suffix = value.partition("*")
        middle_end = len(name) - len(suffix) if suffix else len(name)
        return value_prefix + name[len(prefix) : middle_end] + value_suffix
    return name


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
        verify: bool = False,
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
        if float(control_timeout_s) <= 0:
            raise ValueError("control_timeout_s must be > 0")
        self._rollout = rollout
        self._rank_local_receiver = bool(getattr(rollout, "rank_local_weight_transfer", False))
        self._use_shm = bool(use_shm)
        self._verify = bool(verify)
        self._weight_sync_engine = self._resolve_weight_sync_engine()
        if self._verify and self._weight_sync_engine is _IPCWeightSyncEngine.LEGACY_ZMQ:
            raise ValueError("IPCWeightSync verify=True requires a native vLLM/vLLM-Omni receiver")
        self._gpu_memory_headroom = int(gpu_memory_headroom_mb) << 20
        self._control_timeout = timedelta(seconds=float(control_timeout_s))
        self._control_group = None
        self._next_model_version = 1

    def _receiver_component_name(self) -> str:
        receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
        return str(receiver.component_name())

    def _resolve_weight_sync_engine(self) -> _IPCWeightSyncEngine:
        receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
        native_capability = bool(getattr(receiver, "supports_native_weight_transfer", False))
        native_requested = bool(getattr(receiver, "native_weight_transfer_requested", False))
        component = self._receiver_component_name()
        if native_requested and not native_capability:
            raise RuntimeError(
                "The Omni deploy config requests native weight transfer, but the running runtime does not expose "
                "the native WTE API. Install vLLM-Omni with the native layerwise reload support (upstream #7686 "
                "or v0.32.0rc1+) instead of silently falling back to legacy ZMQ."
            )
        if component == "vllm" or native_capability:
            selected = _IPCWeightSyncEngine.VLLM_NATIVE_WTE
        else:
            selected = _IPCWeightSyncEngine.LEGACY_ZMQ
        logger.info(
            "IPCWeightSync selected engine=%s component=%s native_requested=%s native_capability=%s verify=%s",
            selected.value,
            component,
            native_requested,
            native_capability,
            self._verify,
        )
        return selected

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
            if callable(configure_sleep):
                configure_sleep(2, preserve_next_sleep=True)

    @property
    def _global_rank(self) -> int:
        if self._dist_ready():
            import torch.distributed as dist

            return int(dist.get_rank())
        return self._my_rank

    @property
    def _world_size(self) -> int:
        if self._dist_ready():
            import torch.distributed as dist

            return int(dist.get_world_size())
        return self.rank_info.world_size if self.rank_info is not None else 1

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
        if self._rank_local_receiver or self._global_rank == 0:
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
        return dict(target if self._rank_local_receiver else self._broadcast_from_rank_zero(target))

    def _run_preflight(
        self,
        *,
        tp_world_size: int,
        worker_device_uuids: list[str],
        expected_metadata: list[dict[str, Any]],
    ) -> None:
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
            expected_fanout = 1 if self._rank_local_receiver else self._world_size
            if expected_fanout != int(tp_world_size):
                raise RuntimeError(f"Expected receiver fanout {expected_fanout}, got vLLM TP world {tp_world_size}")
            visible = [
                token.strip() for token in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if token.strip()
            ]
            if len(visible) != 1:
                raise RuntimeError(f"Actor rank must have one CUDA_VISIBLE_DEVICES token, got {visible!r}")
            hosts = [str(item["host"]) for item in topology]
            actor_uuids = [str(item["device_uuid"]) for item in topology]
            if len(set(hosts)) != 1:
                raise RuntimeError(f"native CUDA IPC requires one node, got hosts={hosts!r}")
            expected_uuids = [actor_uuids[self._global_rank]] if self._rank_local_receiver else actor_uuids
            if expected_uuids != list(worker_device_uuids):
                raise RuntimeError(
                    f"Actor/vLLM physical UUID mismatch: actors={actor_uuids!r}, workers={worker_device_uuids!r}"
                )

            largest = max(int(item["nbytes"]) for item in expected_metadata)
            if largest > self._bucket_bytes:
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
            required = 3 * self._bucket_bytes + largest_layer + self._gpu_memory_headroom
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if free_bytes < required:
                raise RuntimeError(
                    f"free GPU memory {free_bytes / 2**30:.2f} GiB is below "
                    f"the conservative vLLM native IPC estimate {required / 2**30:.2f} GiB"
                )
            logger.info(
                "vLLM native IPC preflight rank=%d free=%.2f/%.2f GiB "
                "required=%.2f GiB largest_tensor=%.2f GiB largest_layer=%.2f GiB",
                self._global_rank,
                free_bytes / 2**30,
                total_bytes / 2**30,
                required / 2**30,
                largest / 2**30,
                largest_layer / 2**30,
            )
        except BaseException as exc:
            error = exc
        self._consensus(error, "topology-memory-preflight")

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

    def _expected_tensor_metadata(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build an export plan for fused-MoE or ordinary state-dict models."""
        model = self._backend.model
        if self._lora_merged or hasattr(model, "peft_config"):
            raise NotImplementedError("vLLM native IPC supports full fine-tuning only")
        if self._expert_weight_transform is not None:
            raise NotImplementedError("external expert export transforms are unsupported")
        config = getattr(model, "config", None)
        model_type = getattr(config, "model_type", None)

        from unirl.utils.peft_merge import raw_state_dict_metadata

        metadata: list[dict[str, Any]] = []
        for raw_name, tensor in raw_state_dict_metadata(model, dtype=self._wire_dtype):
            if raw_name.endswith((".lora_A", ".lora_B")):
                continue
            name = _apply_name_remap(raw_name, self._name_remap)
            if name is not None:
                metadata.append(self._shape_metadata(name, tuple(tensor.shape), tensor.dtype))
        if not metadata:
            raise RuntimeError(f"native IPC model {model_type!r} has an empty export")
        if model_type != "qwen3_moe":
            return metadata, structural_manifest(metadata)
        if self._name_remap:
            raise NotImplementedError("Qwen3-MoE schema validation does not support name_remap")

        skip_lm_head = bool(getattr(config, "tie_word_embeddings", False))
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
        return metadata, structural_manifest(metadata)

    def _verify_native_receiver(self, expected_checksums: dict[str, str]) -> None:
        """Compare sender fingerprints with every TP1 native worker read-back."""
        if not expected_checksums:
            raise RuntimeError("vLLM native IPC verification produced no sender checksums")
        receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
        checksum_fn = getattr(receiver, "param_checksums", None)
        if not callable(checksum_fn):
            raise RuntimeError(
                "vLLM native IPC verification requested, but the rollout engine does not expose param_checksums"
            )
        topology_fn = getattr(receiver, "tp_per_stage", None)
        topology = topology_fn() if callable(topology_fn) else {}
        loaded = checksum_fn(names=list(expected_checksums))
        if not isinstance(loaded, dict) or not loaded:
            raise RuntimeError("vLLM native IPC verification returned no worker checksums")
        loaded_stage_ids = {int(stage_id) for stage_id in loaded}
        normalized_topology = {int(stage_id): int(tp) for stage_id, tp in topology.items()}
        if any(normalized_topology.get(stage_id, 1) != 1 for stage_id in loaded_stage_ids):
            raise NotImplementedError(
                f"vLLM native IPC checksum verification currently requires TP1 workers; got topology={topology!r}"
            )
        for stage_id, per_rank in loaded.items():
            if not isinstance(per_rank, (list, tuple)) or len(per_rank) != 1:
                raise RuntimeError(
                    f"vLLM native IPC verification expected one TP1 read-back for stage {stage_id}, got {per_rank!r}"
                )
            actual = per_rank[0]
            if actual != expected_checksums:
                actual_map = actual if isinstance(actual, dict) else {}
                missing = sorted(set(expected_checksums) - set(actual_map))
                extra = sorted(set(actual_map) - set(expected_checksums))
                mismatched = sorted(
                    name
                    for name in set(expected_checksums).intersection(actual_map)
                    if actual_map[name] != expected_checksums[name]
                )
                raise RuntimeError(
                    f"vLLM native IPC verification failed on stage {stage_id}: "
                    f"missing={missing[:8]} extra={extra[:8]} mismatched={mismatched[:8]}"
                )
        logger.info(
            "vLLM native IPC receiver verification passed: stages=%d tensors=%d",
            len(loaded),
            len(expected_checksums),
        )

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
        expected_manifest: Optional[dict[str, Any]] = None
        metadata_error: Optional[BaseException] = None
        try:
            expected_metadata, expected_manifest = self._expected_tensor_metadata()
        except BaseException as exc:
            metadata_error = exc
        self._consensus(metadata_error, "expected-schema")
        assert expected_metadata is not None and expected_manifest is not None
        expected_checksums: dict[str, str] = {}

        def _iter_verified_tensors():
            from unirl.distributed.weight_sync.transfer.checksum import fingerprint_tensor

            for name, tensor in self._iter_full_tensors():
                if self._verify:
                    expected_checksums[name] = fingerprint_tensor(tensor)
                yield name, tensor

        target = self._target_topology()
        tp_world_size = int(target["tp_world_size"])
        self._run_preflight(
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
            "device_uuids": list(target["device_uuids"]),
        }

        source = VLLMNativeWeightSource(
            metadata=[
                ParamMeta(
                    name=item["name"],
                    dtype=getattr(torch, item["dtype"].removeprefix("torch.")),
                    shape=tuple(item["shape"]),
                )
                for item in expected_metadata
            ],
            iterator_factory=_iter_verified_tensors,
        )
        client = VLLMNativeWeightSyncClient(
            rollout=self._rollout,
            header=header,
            flush_cache=self._flush_cache,
        )
        engine = VLLMNativeIPCTrainerEngine(
            client=client,
            source=source,
            is_sender=self._rank_local_receiver or self._global_rank == 0,
            packed_buffer_size_bytes=self._bucket_bytes,
            consensus=self._consensus,
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
            if self._verify:
                verification_error: Optional[BaseException] = None
                if engine.is_sender:
                    try:
                        self._verify_native_receiver(expected_checksums)
                    except BaseException as exc:
                        verification_error = exc
                self._consensus(verification_error, "native-receiver-verify")
            release_error: Optional[BaseException] = None
            if engine.is_sender:
                try:
                    client.release_ipc_imports()
                except BaseException as exc:
                    release_error = exc
            self._consensus(release_error, "native-ipc-release")
            self._next_model_version += 1
            success = True
            logger.info(
                "vLLM native IPC TP%d committed version=%d tensors=%d bytes=%d elapsed=%.2fs",
                tp_world_size,
                header["model_version"],
                expected_manifest["tensor_count"],
                expected_manifest["byte_count"],
                time.perf_counter() - started,
            )
        finally:
            if not success and engine.is_sender:
                self._poison_rollout(f"vLLM native IPC publication {publication_id} failed")
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _sync_with_legacy_zmq(self) -> None:
        """Pump full weights to the co-located engine over per-stage sockets."""
        ri = self.rank_info
        if ri is not None and ri.tp_size > 1:
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
                handle = zmq_handle(replica_rank=replica_rank, stage_id=sid, local_rank=0)
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
                logger.debug("Failed to destroy IPCWeightSync control process group", exc_info=True)


__all__ = ["IPCWeightSync"]
