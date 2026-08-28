"""Shared bucketed-IPC receive mixin for HI3 worker-extension classes."""

from __future__ import annotations

import logging
from typing import Optional

import torch

from unirl.distributed.weight_sync.transfer.bucketed_transfer import (
    BucketedWeightReceiver,
)
from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
    DIFFRL_LORA_INT_ID,
    DIFFRL_LORA_NAME,
    DIFFRL_LORA_PATH,
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
                self.remove_lora(DIFFRL_LORA_INT_ID)
            except Exception as exc:
                logger.warning(
                    "%s.remove_lora(%d) failed: %s",
                    type(self).__name__,
                    DIFFRL_LORA_INT_ID,
                    exc,
                )

        device = getattr(self, "device", None)
        if device is None:
            raise RuntimeError(
                f"{type(self).__name__}: worker has no `device` attribute — unexpected for a fully-initialized worker."
            )

        handle = zmq_handle(
            replica_rank=int(replica_rank) if replica_rank is not None else replica_rank_from_env(),
            stage_id=int(stage_id),
            local_rank=int(getattr(self, "local_rank", 0)),
        )
        receiver = BucketedWeightReceiver(
            zmq_handle=handle,
            device=device,
            use_shm=use_shm,
        )
        receiver.receive_weights(
            on_bucket_received=lambda weights: self._diffrl_load_bucket(
                weights, peft_config=peft_config, base_sync_done=base_sync_done
            )
        )

    def _diffrl_load_bucket(
        self,
        weights: list[tuple[str, torch.Tensor]],
        peft_config: Optional[dict],
        base_sync_done: bool,
    ) -> None:
        if peft_config and base_sync_done:
            tensors = dict(weights)
            lora_request = OmniTensorLoRARequest(
                lora_name=DIFFRL_LORA_NAME,
                lora_int_id=DIFFRL_LORA_INT_ID,
                lora_path=DIFFRL_LORA_PATH,
                peft_config=peft_config,
                lora_tensors=tensors,
            )
            self.add_lora(lora_request)
            logger.info(
                "%s: LoRA bucket loaded (%d tensors, adapter id=%d)",
                type(self).__name__,
                len(tensors),
                DIFFRL_LORA_INT_ID,
            )
        else:
            logger.debug("%s: bucket loaded (%d tensors)", type(self).__name__, len(weights))
            self._diffrl_load_weights(weights)

    def _diffrl_load_weights(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Forward weights to whichever loader the underlying worker exposes."""
        runner = getattr(self, "model_runner", None)
        if runner is not None:
            for attr in ("pipeline", "model"):
                obj = getattr(runner, attr, None)
                validator = getattr(obj, "validate_weight_sync_names", None) if obj is not None else None
                if callable(validator):
                    validator(weights)
                    break

        loader = getattr(self, "load_weights", None)
        if callable(loader):
            loader(weights)
            return
        if runner is None:
            raise RuntimeError(f"{type(self).__name__}: no `load_weights` and no `model_runner`.")
        for attr in ("model", "pipeline"):
            obj = getattr(runner, attr, None)
            obj_loader = getattr(obj, "load_weights", None) if obj is not None else None
            if callable(obj_loader):
                obj_loader(weights)
                return
        raise RuntimeError(
            f"{type(self).__name__}: could not find a load_weights method on "
            f"self, model_runner.model, or model_runner.pipeline."
        )

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

        local_rank = int(getattr(self, "local_rank", 0))
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
        self._diffrl_load_weights(named_tensors)
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
        from unirl.rollout.engine.vllm_omni.patches.runtime import (
            OmniTensorLoRARequest,
        )

        lora_tensors = MultiprocessingSerializer.deserialize(lora_tensors_serialized)
        if not isinstance(lora_tensors, dict):
            raise TypeError(
                f"{type(self).__name__}.set_lora_from_tensor_dict: "
                f"deserialised lora_tensors expected dict, got "
                f"{type(lora_tensors).__name__}"
            )
        request = OmniTensorLoRARequest(
            lora_name=str(lora_name),
            lora_int_id=int(lora_int_id),
            lora_path=str(lora_path),
            peft_config=dict(peft_config or {}),
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
        from unirl.rollout.engine.vllm_omni.patches.runtime import (
            OmniTensorLoRARequest,
        )

        request = OmniTensorLoRARequest(
            lora_name=str(lora_name),
            lora_int_id=int(lora_int_id),
            lora_path=str(lora_path),
            peft_config=dict(peft_config or {}),
            lora_tensors=lora_tensors,
        )
        return self.add_lora(request)

    def _diffrl_describe_params(
        self,
        names: Optional[list] = None,
    ) -> dict:
        """Return ``{name: (shape_tuple, dtype_str)}`` for the worker's loaded model."""
        runner = getattr(self, "model_runner", None)
        if runner is None:
            return {}
        param_source = None
        for attr in ("pipeline", "model"):
            obj = getattr(runner, attr, None)
            if obj is not None and hasattr(obj, "named_parameters"):
                param_source = obj
                break
        if param_source is None:
            return {}

        target = set(names) if names else None
        out: dict = {}
        for name, p in param_source.named_parameters():
            if target is not None and name not in target:
                continue
            out[name] = (tuple(p.shape), str(p.dtype))
        return out

    def _diffrl_param_checksums(
        self,
        names: Optional[list] = None,
    ) -> dict:
        """Return ``{name: short_sha256_hex}`` for the worker's loaded model."""
        import hashlib

        runner = getattr(self, "model_runner", None)
        if runner is None:
            return {}
        param_source = None
        for attr in ("pipeline", "model"):
            obj = getattr(runner, attr, None)
            if obj is not None and hasattr(obj, "named_parameters"):
                param_source = obj
                break
        if param_source is None:
            return {}

        target = set(names) if names else None
        out: dict = {}
        for name, p in param_source.named_parameters():
            if target is not None and name not in target:
                continue
            data = p.detach().contiguous()
            hasher = hashlib.sha256()
            hasher.update(str(data.dtype).encode())
            hasher.update(str(tuple(data.shape)).encode())
            flat = data.view(torch.uint8).flatten()
            n = flat.numel()
            head = flat[: min(256, n)].cpu().numpy().tobytes()
            tail = flat[max(0, n - 256) :].cpu().numpy().tobytes()
            hasher.update(head)
            hasher.update(tail)
            hasher.update(str(n).encode())
            out[name] = hasher.hexdigest()[:16]
        return out

    def _diffrl_loaded_param_checksums(
        self,
        names: Optional[list] = None,
    ) -> dict:
        """Full-byte SHA-256 of the worker's loaded parameters."""
        from unirl.distributed.weight_sync.transfer.checksum import (
            fingerprint_tensor,
        )

        runner = getattr(self, "model_runner", None)
        if runner is None:
            return {}
        param_source = None
        for attr in ("pipeline", "model"):
            obj = getattr(runner, attr, None)
            if obj is not None and hasattr(obj, "named_parameters"):
                param_source = obj
                break
        if param_source is None:
            return {}

        target = set(names) if names else None
        out: dict = {}
        for name, p in param_source.named_parameters():
            if target is not None and name not in target:
                continue
            out[name] = fingerprint_tensor(p)
        return out

    def _diffrl_loaded_lora_checksums(
        self,
        adapter_id: int,
        names: Optional[list] = None,
    ) -> dict:
        """Full-byte SHA-256 of the worker's loaded LoRA adapter tensors."""
        from unirl.distributed.weight_sync.transfer.checksum import (
            fingerprint_tensor,
        )

        manager = getattr(self, "lora_manager", None) or getattr(
            getattr(self, "model_runner", None), "lora_manager", None
        )
        if manager is None:
            return {}
        manager = getattr(manager, "_adapter_manager", manager)
        registered = getattr(manager, "_registered_adapters", None)
        if registered is None:
            return {}
        lora_model = registered.get(int(adapter_id))
        if lora_model is None:
            return {}
        target = set(names) if names else None
        out: dict = {}
        for layer_name, layer in lora_model.loras.items():
            if target is not None and layer_name not in target:
                continue
            per_field: dict = {}
            for field in ("lora_a", "lora_b", "bias", "embeddings_tensor"):
                t = getattr(layer, field, None)
                if isinstance(t, torch.Tensor):
                    per_field[field] = fingerprint_tensor(t)
                elif isinstance(t, (list, tuple)):
                    for i, sub in enumerate(t):
                        if isinstance(sub, torch.Tensor):
                            per_field[f"{field}.{i}"] = fingerprint_tensor(sub)
            out[layer_name] = per_field
        return out


__all__ = ["BucketedIPCReceiveMixin"]
