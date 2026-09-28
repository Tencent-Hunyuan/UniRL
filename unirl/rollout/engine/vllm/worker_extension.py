"""Small worker-side hooks around vLLM's native weight-transfer engine."""

from __future__ import annotations

from typing import Any, Mapping


class UniRLWeightSyncExtension:
    """Expose topology, publication receipts, and IPC cleanup to UniRL."""

    def unirl_weight_sync_capabilities(self) -> dict[str, Any]:
        import torch
        from vllm import __version__ as vllm_version
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        return {
            "tp_rank": int(get_tensor_model_parallel_rank()),
            "tp_world_size": int(get_tensor_model_parallel_world_size()),
            "cuda_device_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "vllm_version": str(vllm_version),
        }

    def unirl_native_weight_sync_receipt(self, *, header: Mapping[str, Any]) -> dict[str, Any]:
        """Attest native WTE completion on one TP worker."""
        import torch
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        from unirl.distributed.weight_sync.transfer.vllm_native_protocol import (
            validate_publication_header,
        )

        tp_rank = int(get_tensor_model_parallel_rank())
        tp_world_size = int(get_tensor_model_parallel_world_size())
        validate_publication_header(header, tp_world_size=tp_world_size)
        device_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        if header["device_uuids"][tp_rank] != device_uuid:
            raise RuntimeError(
                f"vLLM native IPC TP rank {tp_rank} UUID {device_uuid} != Actor UUID {header['device_uuids'][tp_rank]}"
            )
        torch.cuda.synchronize()
        model = self.model_runner.get_model()
        resident_count = sum(1 for _name, _parameter in model.named_parameters())
        return {
            "tp_rank": tp_rank,
            "tp_world_size": tp_world_size,
            "device_uuid": device_uuid,
            "publication_id": str(header["publication_id"]),
            "model_version": int(header["model_version"]),
            "expected_manifest_fingerprint": str(header["expected_manifest"]["fingerprint"]),
            "resident_parameter_count": resident_count,
            "ready": resident_count > 0,
        }

    def unirl_native_ipc_collect(self) -> None:
        """Release cached imports after the trainer drops its IPC export."""
        import torch

        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()


__all__ = ["UniRLWeightSyncExtension"]
