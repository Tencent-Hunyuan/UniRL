"""Thin adapters around vLLM 0.27's native packed-IPC weight-transfer engine."""

from __future__ import annotations

import base64
import pickle
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Optional

import torch
from torch.multiprocessing.reductions import reduce_tensor
from vllm.distributed.weight_transfer import ParamMeta, VLLMWeightSyncClient, WeightSource
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerWeightTransferEngine

ConsensusFn = Callable[[Optional[BaseException], str], None]


class VLLMNativeWeightSource(WeightSource):
    """Expose UniRL's lazy canonical exporter through vLLM's native source API."""

    def __init__(
        self,
        *,
        metadata: Sequence[ParamMeta],
        iterator_factory: Callable[[], Iterator[tuple[str, torch.Tensor]]],
    ) -> None:
        self._metadata = list(metadata)
        self._iterator_factory = iterator_factory
        self.actual_metadata: list[dict[str, Any]] = []

    def metadata(self) -> list[ParamMeta]:
        return list(self._metadata)

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        self.actual_metadata.clear()
        for name, tensor in self._iterator_factory():
            self.actual_metadata.append(
                {
                    "name": str(name),
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                    "nbytes": int(tensor.numel()) * int(tensor.element_size()),
                }
            )
            yield name, tensor


class VLLMNativeWeightSyncClient(VLLMWeightSyncClient):
    """Adapt UniRL's spawned rollout runtime to vLLM's native client API."""

    def __init__(
        self,
        *,
        rollout: Any,
        header: dict[str, Any],
        flush_cache: bool,
    ) -> None:
        self._rollout = rollout
        self._header = dict(header)
        self._flush_cache = bool(flush_cache)
        self.last_result: Optional[dict[str, Any]] = None

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        self._rollout.init_native_weight_transfer(init_info=dict(init_info))

    def start_weight_update(self) -> None:
        self._rollout.start_native_weight_update(header=self._header)

    def update_weights(self, update_info: dict[str, Any]) -> None:
        payload = dict(update_info)
        ipc_handles = payload.pop("ipc_handles", None)
        if ipc_handles is not None:
            payload["ipc_handles_pickled"] = base64.b64encode(pickle.dumps(ipc_handles)).decode("ascii")
        self._rollout.update_native_weights(update_info=payload)

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self.last_result = self._rollout.finish_native_weight_update(
            header=self._header,
            flush_cache=self._flush_cache,
            weight_version=weight_version,
        )

    def release_ipc_imports(self) -> None:
        """Collect worker imports after the trainer releases its IPC export."""
        self._rollout.release_native_ipc()


class VLLMNativeIPCTrainerEngine(IPCTrainerWeightTransferEngine):
    """Bounded wrapper over vLLM's native packed-IPC trainer engine."""

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        rank: int,
        packed_buffer_size_bytes: int,
        consensus: ConsensusFn,
        materialization_boundaries: Sequence[int],
    ) -> None:
        super().__init__(
            client=client,
            source=source,
            is_sender=int(rank) == 0,
            packed=True,
            packed_buffer_size_bytes=int(packed_buffer_size_bytes),
        )
        self._consensus = consensus
        self._materialization_boundaries = frozenset(int(index) for index in materialization_boundaries)

    def initialize(self) -> None:
        error: Optional[BaseException] = None
        if self.is_sender:
            try:
                self.client.init_weight_transfer_engine({"packed": True})
            except BaseException as exc:
                error = exc
        self._consensus(error, "vllm-native-init")

    def send_weights(
        self,
        weight_version: str | None = None,
        before_finish: Callable[[], None] | None = None,
    ) -> None:
        start_error: Optional[BaseException] = None
        if self.is_sender:
            try:
                self.client.start_weight_update()
            except BaseException as exc:
                start_error = exc
        self._consensus(start_error, "vllm-native-start")

        try:
            self._send_planned_packed()

            verify_error: Optional[BaseException] = None
            if before_finish is not None:
                try:
                    before_finish()
                except BaseException as exc:
                    verify_error = exc
            self._consensus(verify_error, "vllm-native-source-verify")

            finish_error: Optional[BaseException] = None
            if self.is_sender:
                try:
                    self.client.finish_weight_update(weight_version)
                except BaseException as exc:
                    finish_error = exc
            self._consensus(finish_error, "vllm-native-finish")
        finally:
            torch.cuda.ipc_collect()

    def _send_planned_packed(self) -> None:
        metadata = self.source.metadata()
        plans = self._plan_chunks(metadata, self.packed_buffer_size_bytes)
        source_iter = iter(self.source)
        expected_boundaries = self._validate_materialization_boundaries(
            self._materialization_boundaries,
            tensor_count=len(metadata),
        )
        ipc_buffer = None
        ipc_args = None
        allocation_error: Optional[BaseException] = None
        try:
            ipc_buffer = torch.empty(
                self.packed_buffer_size_bytes,
                dtype=torch.uint8,
                device=f"cuda:{self.device_index}",
            )
            _, ipc_args = reduce_tensor(ipc_buffer)
        except BaseException as exc:
            allocation_error = exc
        self._consensus(allocation_error, "vllm-native-buffer-allocation")
        assert ipc_buffer is not None and ipc_args is not None

        consumed = 0
        for chunk_index, chunk in enumerate(plans):
            names: list[str] = []
            shapes: list[list[int]] = []
            dtype_names: list[str] = []
            tensor_sizes: list[int] = []
            offset = 0
            chunk_end = consumed + len(chunk) - 1
            for expected in chunk:
                tensor_index = consumed
                tensor = None
                flat = None
                tensor_info: Optional[tuple[str, list[int], str, int]] = None
                materialize_error: Optional[BaseException] = None
                try:
                    name, tensor = next(source_iter)
                    self._validate_tensor(expected, name, tensor)
                    flat = tensor.detach().contiguous().view(torch.uint8).view(-1)
                    size = int(flat.numel())
                    ipc_buffer[offset : offset + size].copy_(flat, non_blocking=True)
                    tensor_info = (
                        str(name),
                        [int(dim) for dim in tensor.shape],
                        str(tensor.dtype).removeprefix("torch."),
                        size,
                    )
                except BaseException as exc:
                    materialize_error = exc
                finally:
                    # Do not retain one completed full tensor while the source
                    # enters the next FSDP materialization collective.
                    del flat
                    del tensor

                is_source_boundary = tensor_index in expected_boundaries
                is_chunk_boundary = tensor_index == chunk_end
                if materialize_error is not None or is_source_boundary or is_chunk_boundary:
                    # A source boundary precedes the next FSDP collective; a
                    # chunk boundary precedes the default-group handle gather.
                    # Every rank must report local failure before either gate.
                    if materialize_error is None:
                        try:
                            torch.cuda.current_stream().synchronize()
                        except BaseException as exc:
                            materialize_error = exc
                    next_gate = self._next_consensus_gate(
                        tensor_index,
                        source_boundaries=expected_boundaries,
                        chunk_end=chunk_end,
                    )
                    self._consensus(materialize_error, f"vllm-native-materialize-{next_gate}")

                assert tensor_info is not None
                name, shape, dtype_name, size = tensor_info
                names.append(name)
                shapes.append(shape)
                dtype_names.append(dtype_name)
                tensor_sizes.append(size)
                offset += size
                consumed += 1

            merged_handle = self._all_gather_and_merge_handles([{self.gpu_uuid: ipc_args}])[0]
            transfer_error: Optional[BaseException] = None
            try:
                self._do_send(
                    names=names,
                    dtype_names=dtype_names,
                    shapes=shapes,
                    ipc_handles=merged_handle,
                    tensor_sizes=tensor_sizes,
                )
            except BaseException as exc:
                transfer_error = exc
            self._consensus(transfer_error, f"vllm-native-transfer-{chunk_index}")

        if consumed != len(metadata):
            raise RuntimeError(f"vLLM native IPC source consumed {consumed} tensors, expected {len(metadata)}")

    @staticmethod
    def _validate_materialization_boundaries(boundaries: frozenset[int], *, tensor_count: int) -> frozenset[int]:
        if tensor_count <= 0:
            raise RuntimeError("vLLM native IPC weight source contains no tensors")
        if not boundaries or max(boundaries) != tensor_count - 1 or min(boundaries) < 0:
            raise ValueError(
                "vLLM native IPC materialization boundaries must include the final tensor "
                f"{tensor_count - 1}, got {sorted(boundaries)!r}"
            )
        if any(index >= tensor_count for index in boundaries):
            raise ValueError(
                f"vLLM native IPC materialization boundary exceeds tensor count {tensor_count}: {sorted(boundaries)!r}"
            )
        return boundaries

    @staticmethod
    def _next_consensus_gate(
        tensor_index: int,
        *,
        source_boundaries: frozenset[int],
        chunk_end: int,
    ) -> int:
        return min([chunk_end, *(index for index in source_boundaries if index >= tensor_index)])

    @staticmethod
    def _plan_chunks(
        metadata: Sequence[ParamMeta],
        buffer_size_bytes: int,
    ) -> list[list[ParamMeta]]:
        chunks: list[list[ParamMeta]] = []
        current: list[ParamMeta] = []
        current_bytes = 0
        for item in metadata:
            size = int(item.dtype.itemsize)
            for dim in item.shape:
                size *= int(dim)
            if size > int(buffer_size_bytes):
                raise ValueError(
                    f"Tensor {item.name!r} requires {size} bytes, larger than vLLM native "
                    f"packed IPC buffer {buffer_size_bytes}"
                )
            if current and current_bytes + size > int(buffer_size_bytes):
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(item)
            current_bytes += size
        if current:
            chunks.append(current)
        if not chunks:
            raise RuntimeError("vLLM native IPC weight source contains no tensors")
        return chunks

    @staticmethod
    def _validate_tensor(expected: ParamMeta, name: str, tensor: torch.Tensor) -> None:
        actual = (str(name), tuple(int(dim) for dim in tensor.shape), tensor.dtype)
        wanted = (expected.name, tuple(expected.shape), expected.dtype)
        if actual != wanted:
            raise RuntimeError(f"vLLM native IPC source order/schema mismatch: expected={wanted!r}, actual={actual!r}")
        if tensor.device.type != "cuda":
            raise RuntimeError(f"vLLM native IPC source tensor {name!r} must be CUDA-backed, got {tensor.device}")


__all__ = [
    "VLLMNativeIPCTrainerEngine",
    "VLLMNativeWeightSource",
    "VLLMNativeWeightSyncClient",
]
