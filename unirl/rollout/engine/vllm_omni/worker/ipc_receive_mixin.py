"""Shared bucketed-IPC receive mixin for HI3 worker-extension classes."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from unirl.distributed.weight_sync.transfer.bucketed_transfer import (
    BucketedWeightReceiver,
)
from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
    UNIRL_LORA_INT_ID,
    UNIRL_LORA_NAME,
    UNIRL_LORA_PATH,
    replica_rank_from_env,
    zmq_handle,
)
from unirl.rollout.engine.vllm_omni.patches.runtime import (
    OmniTensorLoRARequest,
    VLLMOmniHijack,
)

logger = logging.getLogger(__name__)


class BucketedIPCReceiveMixin:
    """Adds ``update_weights_from_ipc`` (and the LoRA hijack install) to a vllm-omni worker via multiple inheritance."""

    def __new__(cls, **kwargs):
        VLLMOmniHijack.hijack()
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()
        return super().__new__(cls)

    def update_weights_from_ipc(
        self,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        stage_id: int = 0,
        replica_rank: Optional[int] = None,
    ) -> None:
        """Receive a state dict over the per-rank ZMQ socket."""
        if peft_config and base_sync_done:
            try:
                self.remove_lora(UNIRL_LORA_INT_ID)
            except Exception as exc:
                logger.warning(
                    "%s.remove_lora(%d) failed: %s",
                    type(self).__name__,
                    UNIRL_LORA_INT_ID,
                    exc,
                )

        if self.device is None:
            raise RuntimeError(
                f"{type(self).__name__}: worker has no `device` attribute — unexpected for a fully-initialized worker."
            )

        handle = zmq_handle(
            replica_rank=replica_rank if replica_rank is not None else replica_rank_from_env(),
            stage_id=stage_id,
            local_rank=self.local_rank,
        )
        receiver = BucketedWeightReceiver(
            zmq_handle=handle,
            device=self.device,
            use_shm=use_shm,
        )

        def _load_bucket(weights: list[tuple[str, torch.Tensor]]) -> None:
            if peft_config and base_sync_done:
                tensors = dict(weights)
                lora_request = OmniTensorLoRARequest(
                    lora_name=UNIRL_LORA_NAME,
                    lora_int_id=UNIRL_LORA_INT_ID,
                    lora_path=UNIRL_LORA_PATH,
                    peft_config=peft_config,
                    lora_tensors=tensors,
                )
                self.add_lora(lora_request)
                logger.info(
                    "%s: LoRA bucket loaded (%d tensors, adapter id=%d)",
                    type(self).__name__,
                    len(tensors),
                    UNIRL_LORA_INT_ID,
                )
            else:
                logger.debug("%s: bucket loaded (%d tensors)", type(self).__name__, len(weights))
                self.unirl_target_module().load_weights(weights)

        receiver.receive_weights(on_bucket_received=_load_bucket)

    def unirl_target_module(self) -> torch.nn.Module:
        """Return the loaded model/pipeline module owned by ``self.model_runner``."""
        runner = self.model_runner
        module = runner.model if hasattr(runner, "model") else runner.pipeline
        if module is None:
            raise RuntimeError(f"{type(self).__name__}: model_runner has no loaded model/pipeline.")
        return module

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list,
        target_modules: Optional[list] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Receive a SGLang-shape one-bag payload and load it."""
        del target_modules, flush_cache  # accepted for SGLang-shape parity
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
        )

        local_rank = self.local_rank
        if local_rank >= len(serialized_named_tensors):
            raise IndexError(
                f"{type(self).__name__}.update_weights_from_tensor: "
                f"local_rank={local_rank} but serialized_named_tensors has "
                f"only {len(serialized_named_tensors)} entries"
            )
        my_payload_str = serialized_named_tensors[local_rank]
        payload = MultiprocessingSerializer.deserialize(my_payload_str)
        bucket = FlattenedTensorBucket(
            flattened_tensor=payload["flattened_tensor"],
            metadata=payload["metadata"],
        )
        named_tensors = bucket.reconstruct_tensors()
        self.unirl_target_module().load_weights(named_tensors)
        logger.info(
            "%s: tensor-payload loaded (%d tensors, load_format=%r)",
            type(self).__name__,
            len(named_tensors),
            load_format,
        )

    def set_lora_from_tensor_dict(
        self,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
        peft_config: dict,
        lora_tensors_serialized: str,
    ) -> bool:
        """Reconstruct an ``OmniTensorLoRARequest`` from primitive args and forward to ``self.add_lora``."""
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            MultiprocessingSerializer,
        )

        lora_tensors = MultiprocessingSerializer.deserialize(lora_tensors_serialized)
        if not isinstance(lora_tensors, dict):
            raise TypeError(
                f"{type(self).__name__}.set_lora_from_tensor_dict: "
                f"deserialised lora_tensors expected dict, got "
                f"{type(lora_tensors).__name__}"
            )
        request = OmniTensorLoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
            peft_config={} if peft_config is None else dict(peft_config),
            lora_tensors=lora_tensors,
        )
        return self.add_lora(request)

    def set_lora_from_tensor_dict_copy(
        self,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
        peft_config: dict,
        lora_tensors_serialized: str,
    ) -> bool:
        """Byte-copy variant of :meth:`set_lora_from_tensor_dict` for HI3."""
        import base64
        import io

        raw = base64.b64decode(lora_tensors_serialized)
        lora_tensors = torch.load(io.BytesIO(raw), map_location="cpu")
        if not isinstance(lora_tensors, dict):
            raise TypeError(
                f"{type(self).__name__}.set_lora_from_tensor_dict_copy: "
                f"deserialised lora_tensors expected dict, got "
                f"{type(lora_tensors).__name__}"
            )
        request = OmniTensorLoRARequest(
            lora_name=lora_name,
            lora_int_id=lora_int_id,
            lora_path=lora_path,
            peft_config={} if peft_config is None else dict(peft_config),
            lora_tensors=lora_tensors,
        )
        return self.add_lora(request)

    def _unirl_loaded_param_checksums(
        self,
        names: Optional[list] = None,
    ) -> dict:
        """Full-byte SHA-256 of the worker's loaded parameters."""
        from unirl.distributed.weight_sync.transfer.checksum import (
            fingerprint_tensor,
        )

        parameter_module = self.unirl_target_module()
        target = set(names) if names else None
        out: dict = {}
        for name, p in parameter_module.named_parameters():
            if target is not None and name not in target:
                continue
            out[name] = fingerprint_tensor(p)
        return out

    def _unirl_loaded_lora_checksums(
        self,
        adapter_id: int,
        names: Optional[list] = None,
    ) -> dict:
        """Full-byte SHA-256 of the worker's loaded LoRA adapter tensors."""
        from unirl.distributed.weight_sync.transfer.checksum import (
            fingerprint_tensor,
        )

        manager = self.lora_manager if hasattr(self, "lora_manager") else self.model_runner.lora_manager
        if manager is None:
            return {}
        if hasattr(manager, "_adapter_manager"):
            manager = manager._adapter_manager
        lora_model = manager._registered_adapters.get(adapter_id)
        if lora_model is None:
            return {}
        target = set(names) if names else None
        out: dict = {}
        for layer_name, layer in lora_model.loras.items():
            if target is not None and layer_name not in target:
                continue
            per_field: dict = {}
            for field, t in (("lora_a", layer.lora_a), ("lora_b", layer.lora_b)):
                if isinstance(t, torch.Tensor):
                    per_field[field] = fingerprint_tensor(t)
                elif isinstance(t, (list, tuple)):
                    for i, sub in enumerate(t):
                        if isinstance(sub, torch.Tensor):
                            per_field[f"{field}.{i}"] = fingerprint_tensor(sub)
            out[layer_name] = per_field
        return out


__all__ = ["BucketedIPCReceiveMixin"]
