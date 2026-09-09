"""vLLM worker extension for UniRL full-weight updates."""

from __future__ import annotations

from typing import List, Optional


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
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
    ) -> dict:
        if load_format not in (None, "flattened_bucket"):
            raise ValueError(f"unsupported direct-vLLM weight load format {load_format!r}")
        if not serialized_named_tensors:
            raise ValueError("direct-vLLM weight update received no payloads")

        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()
        rank = int(getattr(self, "rank", 0))
        payload = serialized_named_tensors[rank % len(serialized_named_tensors)]
        decoded = MultiprocessingSerializer.deserialize(payload)
        bucket = FlattenedTensorBucket(
            flattened_tensor=decoded["flattened_tensor"],
            metadata=decoded["metadata"],
        )
        named_tensors = bucket.reconstruct_tensors()
        model = self.model_runner.get_model()
        loaded = model.load_weights(iter(named_tensors))
        import torch

        torch.cuda.synchronize()
        return {
            "rank": rank,
            "num_tensors": len(named_tensors),
            "num_loaded": len(loaded) if loaded is not None else None,
        }


__all__ = ["UniRLWeightSyncExtension"]
