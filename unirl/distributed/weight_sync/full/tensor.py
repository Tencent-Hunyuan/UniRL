"""v2 full-weight tensor-payload sync (COLOCATE)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync


class TensorWeightSync(FullWeightSync):
    """Colocate full-weight sync via serialized tensor payloads."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
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
        self._rollout = rollout

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Serialize each bucket and load it into the local engine."""
        import torch

        ri = self.rank_info
        tp_size = int(ri.tp_size) if ri is not None else 1
        is_tp_zero = ri is None or ri.tp_rank == 0

        receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
        rollout_mod = type(receiver).__module__
        use_sglang = "sglang" in rollout_mod and "vllm" not in rollout_mod
        if use_sglang:
            try:
                from sglang.srt.utils.common import MultiprocessingSerializer
                from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
                from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
            except ImportError as exc:
                raise ImportError(
                    "TensorWeightSync requires SGLang's native serializer, reductions, "
                    "and tensor bucket classes for an SGLang rollout; falling back to "
                    "sgl_compat would produce a SafeUnpickler-incompatible payload."
                ) from exc
        if not use_sglang:
            from unirl.distributed.weight_sync.transfer.sgl_compat import (
                FlattenedTensorBucket,
                MultiprocessingSerializer,
                monkey_patch_torch_reductions,
            )

        monkey_patch_torch_reductions()

        dist_ready = self._dist_ready()

        for bucket, is_last in self._iter_buckets():
            by_dtype: dict = {}
            for name, tensor in bucket:
                by_dtype.setdefault(tensor.dtype, []).append((name, tensor))
            del name, tensor

            fanout = int(getattr(receiver, "weight_payload_fanout", tp_size))
            sglang_tp_fanout = use_sglang and fanout > 1
            participates_in_sglang_tp = sglang_tp_fanout and dist_ready

            if not is_tp_zero and not participates_in_sglang_tp:
                del by_dtype, bucket
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            groups = list(by_dtype.values())
            n_dtypes = len(groups)
            for i, grouped in enumerate(groups):
                flush = self._flush_cache and is_last and i == n_dtypes - 1
                payload_keepalive = None
                if sglang_tp_fanout:
                    if dist_ready:
                        local_payload, serialization_error = self._serialize_payload_or_error(
                            grouped,
                            FlattenedTensorBucket,
                            MultiprocessingSerializer,
                        )
                        payload_per_rank = self._gather_sglang_tp_payloads(
                            local_payload,
                            local_error=serialization_error,
                            rank_info=ri,
                            tp_size=tp_size,
                        )
                    else:
                        payload_per_rank, payload_keepalive = self._serialize_single_process_sglang_tp_payloads(
                            grouped,
                            fanout=fanout,
                            flat_bucket_cls=FlattenedTensorBucket,
                            serializer_cls=MultiprocessingSerializer,
                        )
                elif fanout > 0:
                    payload_per_rank = [
                        self._serialize_payload(
                            grouped,
                            FlattenedTensorBucket,
                            MultiprocessingSerializer,
                        )
                        for _ in range(fanout)
                    ]
                else:
                    payload_per_rank = []

                update_error = None
                if is_tp_zero and payload_per_rank:
                    try:
                        self._rollout.update_weights_from_tensor(
                            serialized_named_tensors=payload_per_rank,
                            load_format="flattened_bucket",
                            flush_cache=flush,
                            track_prefix=self._track_prefix,
                        )
                    except BaseException as exc:  # keep peer ranks from hanging
                        update_error = f"{type(exc).__name__}: {exc}"

                if participates_in_sglang_tp:
                    self._raise_if_sglang_tp_update_failed(update_error, rank_info=ri)
                elif update_error is not None:
                    raise RuntimeError(f"TensorWeightSync: rollout update failed: {update_error}")
                del payload_keepalive

            # Release each gathered bucket before loading the next to avoid OOMs.
            del groups, by_dtype, bucket
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

    @staticmethod
    def _serialize_payload(grouped, flat_bucket_cls, serializer_cls) -> str:
        flat = flat_bucket_cls(named_tensors=grouped)
        payload = {
            "flattened_tensor": flat.get_flattened_tensor(),
            "metadata": flat.get_metadata(),
        }
        try:
            return serializer_cls.serialize(payload, output_str=True)
        finally:
            del payload, flat

    @classmethod
    def _serialize_payload_or_error(
        cls, grouped, flat_bucket_cls, serializer_cls
    ) -> tuple[Optional[str], Optional[str]]:
        """Capture a rank-local serialization error so peers can fail together."""
        try:
            return (
                cls._serialize_payload(
                    grouped,
                    flat_bucket_cls,
                    serializer_cls,
                ),
                None,
            )
        except BaseException as exc:
            return None, f"{type(exc).__name__}: {exc}"

    @staticmethod
    def _dist_ready() -> bool:
        try:
            import torch.distributed as dist

            return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    @staticmethod
    def _gather_sglang_tp_payloads(
        local_payload: Optional[str],
        *,
        local_error: Optional[str],
        rank_info,
        tp_size: int,
    ) -> list[str]:
        import torch.distributed as dist

        if rank_info is None:
            raise RuntimeError("TensorWeightSync: distributed SGLang TP payload gather requires rank_info")
        local = {
            "rank": int(rank_info.rank),
            "dp_rank": int(rank_info.dp_rank),
            "pp_rank": int(rank_info.pp_rank),
            "tp_rank": int(rank_info.tp_rank),
            "payload": local_payload,
            "error": local_error,
        }
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        errors = [item for item in gathered if item is not None and item.get("error")]
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"TensorWeightSync: SGLang TP payload serialization failed on rank {first['rank']}: {first['error']}"
            )
        group = [
            item
            for item in gathered
            if item is not None
            and int(item["dp_rank"]) == int(rank_info.dp_rank)
            and int(item["pp_rank"]) == int(rank_info.pp_rank)
        ]
        group.sort(key=lambda item: int(item["tp_rank"]))
        tp_ranks = [int(item["tp_rank"]) for item in group]
        if len(group) != int(tp_size) or tp_ranks != list(range(int(tp_size))):
            raise RuntimeError(
                "TensorWeightSync: incomplete SGLang TP payload gather for "
                f"dp_rank={rank_info.dp_rank}, pp_rank={rank_info.pp_rank}: "
                f"expected tp ranks 0..{int(tp_size) - 1}, got {tp_ranks}"
            )
        missing_payload_ranks = [int(item["rank"]) for item in group if item.get("payload") is None]
        if missing_payload_ranks:
            raise RuntimeError(
                f"TensorWeightSync: SGLang TP payload gather returned empty payloads from ranks {missing_payload_ranks}"
            )
        return [str(item["payload"]) for item in group]

    @staticmethod
    def _raise_if_sglang_tp_update_failed(local_error: Optional[str], *, rank_info) -> None:
        import torch.distributed as dist

        local = {"rank": int(rank_info.rank) if rank_info is not None else 0, "error": local_error}
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local)
        errors = [item for item in gathered if item is not None and item.get("error")]
        if errors:
            first = errors[0]
            raise RuntimeError(f"TensorWeightSync: rollout update failed on rank {first['rank']}: {first['error']}")

    @classmethod
    def _serialize_single_process_sglang_tp_payloads(
        cls,
        grouped,
        *,
        fanout: int,
        flat_bucket_cls,
        serializer_cls,
    ) -> tuple[list[str], list]:
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() < int(fanout):
            raise RuntimeError(
                "TensorWeightSync cannot build SGLang TP payloads in this process: "
                f"fanout={fanout}, visible_cuda_devices={torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
                "Run under the distributed TP layout so each TP rank exports its local CUDA IPC payload, "
                "or use CkptEngineIPCWeightSync for SGLang TP colocate sync."
            )
        payloads: list[str] = []
        keepalive = []
        for tp_rank in range(int(fanout)):
            device = torch.device("cuda", tp_rank)
            per_rank = [(name, tensor.to(device, non_blocking=False).contiguous()) for name, tensor in grouped]
            keepalive.extend(tensor for _, tensor in per_rank)
            payloads.append(cls._serialize_payload(per_rank, flat_bucket_cls, serializer_cls))
        return payloads, keepalive


__all__ = ["TensorWeightSync"]
