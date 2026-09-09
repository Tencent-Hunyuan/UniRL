"""vLLM worker extension for UniRL full-weight updates."""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Mapping, Optional


class UniRLWeightSyncExtension:
    """Deserialize a UniRL tensor bucket inside each TP worker."""

    def unirl_before_sleep(self) -> None:
        """Drop model caches whose CuMem mappings will be released."""
        import torch

        torch.cuda.synchronize()
        model = self.model_runner.get_model()
        for module in model.modules():
            cleanup = getattr(module, "_unirl_before_sleep", None)
            if callable(cleanup):
                cleanup()
        torch.cuda.synchronize()

    def unirl_update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: Optional[List[str]] = None,
        payloads: Optional[List[str]] = None,
        header: Optional[Mapping[str, Any]] = None,
        tp_world_size: Optional[int] = None,
        load_format: Optional[str] = None,
    ) -> dict:
        if load_format not in (None, "flattened_bucket"):
            raise ValueError(f"unsupported direct-vLLM weight load format {load_format!r}")
        if payloads is not None and serialized_named_tensors is not None and payloads != serialized_named_tensors:
            raise ValueError("direct-vLLM received conflicting payloads and serialized_named_tensors")
        wire_payloads = list(payloads if payloads is not None else serialized_named_tensors or [])
        if not wire_payloads:
            raise ValueError("direct-vLLM weight update received no payloads")

        actual_tp_world, tp_rank = self._unirl_tp_shape(tp_world_size, len(wire_payloads))
        self._unirl_tp_world_size = actual_tp_world
        self._unirl_tp_rank = tp_rank
        if len(wire_payloads) != actual_tp_world:
            raise ValueError(f"direct-vLLM payload count {len(wire_payloads)} != TP world size {actual_tp_world}")
        if tp_rank < 0 or tp_rank >= actual_tp_world:
            raise ValueError(f"direct-vLLM TP rank {tp_rank} outside world size {actual_tp_world}")

        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
            monkey_patch_torch_reductions,
        )

        if header is None:
            # Legacy CUDA-IPC payloads encode CUDA storage rebuild metadata.
            # Receipt-aware FSDP payloads are CPU-backed and must not install
            # the CUDA tuple-rewrite hook.
            monkey_patch_torch_reductions()
        if header is not None:
            from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
                validate_bucket_header,
            )

            validate_bucket_header(header, payload_count=len(wire_payloads), tp_world_size=actual_tp_world)
            self._unirl_validate_bucket_sequence(header)

        # Every worker receives an independent CPU shared-memory payload with
        # identical tensor bytes. The vLLM loader copies and TP-slices it onto
        # the worker's own CUDA device.
        payload = wire_payloads[tp_rank]
        decoded = MultiprocessingSerializer.deserialize(payload)
        flattened_tensor = decoded["flattened_tensor"]
        if header is not None and flattened_tensor.device.type != "cpu":
            raise ValueError(f"FSDP-vLLM receipt-aware payload must be CPU-backed, got {flattened_tensor.device}")
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor,
            metadata=decoded["metadata"],
        )
        named_tensors = bucket.reconstruct_tensors()
        if header is not None:
            self._unirl_validate_named_tensors(header, named_tensors)

        expected_names = (
            [str(metadata["name"]) for metadata in header["tensors"]]
            if header is not None
            else [str(name) for name, _ in named_tensors]
        )
        consumed: list[str] = []

        def tracked_tensors():
            for name, tensor in named_tensors:
                consumed.append(str(name))
                yield name, tensor

        model = self.model_runner.get_model()
        loaded = model.load_weights(tracked_tensors())
        import torch

        torch.cuda.synchronize()
        consumed_counts = Counter(consumed)
        expected_counts = Counter(expected_names)
        missing = sorted((expected_counts - consumed_counts).elements())
        unexpected = sorted((consumed_counts - expected_counts).elements())
        duplicate = sorted(name for name, amount in consumed_counts.items() if amount > 1)
        if header is not None:
            previously_consumed = set(getattr(self, "_unirl_pending_consumed_names", set()))
            duplicate = sorted(set(duplicate) | (set(consumed) & previously_consumed))
        loaded_names = self._unirl_normalize_loaded_names(loaded)
        loaded_missing: list[str] = []
        loaded_unexpected: list[str] = []
        if header is not None:
            pending_loaded = set(getattr(self, "_unirl_pending_loaded_names", set()))
            pending_loaded.update(loaded_names)
            expected_model_names = {str(name) for name, _parameter in model.named_parameters()}
            if bool(header["is_last"]):
                loaded_missing = sorted(expected_model_names - pending_loaded)
                loaded_unexpected = sorted(pending_loaded - expected_model_names)
        receipt = {
            "rank": tp_rank,
            "tp_rank": tp_rank,
            "tp_world_size": actual_tp_world,
            "sync_id": str(header["sync_id"]) if header is not None else None,
            "fingerprint": str(header["fingerprint"]) if header is not None else None,
            "index": int(header["index"]) if header is not None else None,
            "count": int(header["count"]) if header is not None else None,
            "is_last": bool(header["is_last"]) if header is not None else True,
            "consumed": consumed,
            "loaded": loaded_names,
            "missing": missing,
            "unexpected": unexpected,
            "duplicate": duplicate,
            "loaded_missing": loaded_missing,
            "loaded_unexpected": loaded_unexpected,
            "model_version": (
                int(header["model_version"])
                if header is not None
                else int(getattr(self, "_unirl_committed_weight_version", 0))
            ),
            "committed": header is None,
            # Keep the old summary fields for callers that only log counts.
            "num_tensors": len(named_tensors),
            "num_loaded": len(loaded_names),
            "consumed_tensor_count": len(consumed),
            "committed_model_version": int(getattr(self, "_unirl_committed_weight_version", 0)),
        }
        if header is not None and not (missing or unexpected or duplicate or loaded_missing or loaded_unexpected):
            self._unirl_pending_consumed_names.update(consumed)
            self._unirl_pending_loaded_names.update(loaded_names)
            self._unirl_pending_next_index = int(header["index"]) + 1
            self._unirl_pending_ready_to_commit = bool(header["is_last"])
        return receipt

    def unirl_commit_weight_version(self, *, sync_id: str, model_version: int) -> dict:
        """Commit only a completely loaded last bucket after driver TP consensus."""
        pending_sync_id = getattr(self, "_unirl_pending_sync_id", None)
        pending_version = int(getattr(self, "_unirl_pending_version", -1))
        ready = bool(getattr(self, "_unirl_pending_ready_to_commit", False))
        if str(sync_id) != pending_sync_id or int(model_version) != pending_version or not ready:
            raise RuntimeError(
                "direct-vLLM cannot commit incomplete/mismatched weight publication: "
                f"requested=({sync_id!r}, {model_version}), "
                f"pending=({pending_sync_id!r}, {pending_version}), ready={ready}"
            )
        committed = int(getattr(self, "_unirl_committed_weight_version", 0))
        if int(model_version) <= committed:
            raise RuntimeError(
                f"direct-vLLM weight model_version {model_version} is not newer than committed {committed}"
            )

        self._unirl_committed_weight_version = int(model_version)
        self._unirl_pending_sync_id = None
        self._unirl_pending_version = None
        self._unirl_pending_count = None
        self._unirl_pending_next_index = 0
        self._unirl_pending_ready_to_commit = False
        self._unirl_pending_consumed_names = set()
        self._unirl_pending_loaded_names = set()
        return {
            "rank": int(getattr(self, "_unirl_tp_rank", getattr(self, "rank", 0))),
            "tp_rank": int(getattr(self, "_unirl_tp_rank", getattr(self, "rank", 0))),
            "sync_id": str(sync_id),
            "model_version": int(model_version),
            "committed_model_version": int(model_version),
            "committed": True,
        }

    @staticmethod
    def _unirl_normalize_loaded_names(loaded: Any) -> list[str]:
        if loaded is None:
            return []
        if isinstance(loaded, str):
            return [loaded]
        try:
            return sorted(str(name) for name in loaded)
        except TypeError:
            return [str(loaded)]

    def _unirl_tp_shape(self, expected_world: Optional[int], payload_count: int) -> tuple[int, int]:
        world: Optional[int] = None
        rank: Optional[int] = None
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            world = int(get_tensor_model_parallel_world_size())
            rank = int(get_tensor_model_parallel_rank())
        except (ImportError, RuntimeError, AssertionError):
            pass
        if world is None:
            parallel_config = getattr(self, "parallel_config", None)
            configured = getattr(parallel_config, "tensor_parallel_size", None)
            world = int(configured) if configured is not None else int(expected_world or payload_count)
        if rank is None:
            rank = int(getattr(self, "rank", 0))
        if expected_world is not None and int(expected_world) != world:
            raise ValueError(f"direct-vLLM caller TP world {expected_world} != worker TP world {world}")
        return world, rank

    def _unirl_validate_bucket_sequence(self, header: Mapping[str, Any]) -> None:
        sync_id = str(header["sync_id"])
        index = int(header["index"])
        count = int(header["count"])
        model_version = int(header["model_version"])
        committed = int(getattr(self, "_unirl_committed_weight_version", 0))

        if index == 0:
            active_sync_id = getattr(self, "_unirl_pending_sync_id", None)
            if active_sync_id is not None:
                raise ValueError(
                    f"direct-vLLM cannot replace incomplete publication {active_sync_id!r} with {sync_id!r}"
                )
            if model_version <= committed:
                raise ValueError(
                    f"direct-vLLM publication model_version {model_version} must exceed committed version {committed}"
                )
            self._unirl_pending_sync_id = sync_id
            self._unirl_pending_version = model_version
            self._unirl_pending_count = count
            self._unirl_pending_next_index = 0
            self._unirl_pending_ready_to_commit = False
            self._unirl_pending_consumed_names = set()
            self._unirl_pending_loaded_names = set()
        pending = (
            getattr(self, "_unirl_pending_sync_id", None),
            int(getattr(self, "_unirl_pending_version", -1)),
            int(getattr(self, "_unirl_pending_count", -1)),
            int(getattr(self, "_unirl_pending_next_index", -1)),
        )
        expected = (sync_id, model_version, count, index)
        if pending != expected:
            raise ValueError(f"direct-vLLM out-of-sequence FSDP bucket: pending={pending}, received={expected}")

    @staticmethod
    def _unirl_validate_named_tensors(
        header: Mapping[str, Any],
        named_tensors: List[tuple[str, Any]],
    ) -> None:
        from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
            tensor_sha256,
        )

        metadata = header["tensors"]
        if len(named_tensors) != len(metadata):
            raise ValueError(f"direct-vLLM decoded tensor count {len(named_tensors)} != header count {len(metadata)}")
        for position, ((name, tensor), expected) in enumerate(zip(named_tensors, metadata, strict=True)):
            actual = {
                "name": str(name),
                "shape": [int(dim) for dim in tensor.shape],
                "dtype": str(tensor.dtype),
                "numel": int(tensor.numel()),
                "sha256": tensor_sha256(tensor),
            }
            wanted = {field: expected[field] for field in ("name", "shape", "dtype", "numel", "sha256")}
            if actual != wanted:
                raise ValueError(
                    f"direct-vLLM decoded tensor #{position} failed header validation: "
                    f"expected={wanted!r}, actual={actual!r}"
                )


__all__ = ["UniRLWeightSyncExtension"]
