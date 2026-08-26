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
from unirl.distributed.weight_sync.transfer.minimax_h3_lora import (
    remap_minimax_h3_lora,
    validate_minimax_h3_lora_coverage,
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
        loader = getattr(self, "load_weights", None)
        if callable(loader):
            loader(weights)
            return
        runner = getattr(self, "model_runner", None)
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
        ready_token: Optional[str] = None,
    ) -> bool:
        """Byte-copy variant of :meth:`set_lora_from_tensor_dict` for HI3."""
        import base64
        import io

        rank = int(getattr(self, "rank", getattr(self, "local_rank", 0)))
        logger.debug(
            "[LoRA-COPY] rank=%d payload_chars=%d decode_begin",
            rank,
            len(lora_tensors_serialized),
        )
        raw = base64.b64decode(lora_tensors_serialized)
        logger.debug("[LoRA-COPY] rank=%d payload_bytes=%d torch_load_begin", rank, len(raw))
        lora_tensors = torch.load(io.BytesIO(raw), map_location="cpu")
        logger.debug("[LoRA-COPY] rank=%d tensors=%d torch_load_complete", rank, len(lora_tensors))
        return self._diffrl_install_lora_tensors(
            lora_name,
            lora_int_id,
            lora_path,
            peft_config,
            lora_tensors,
            ready_token=ready_token,
        )

    def set_lora_from_tensor_file(
        self,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
        peft_config: dict,
        payload_path: str,
        ready_token: Optional[str] = None,
    ) -> bool:
        """Load a large adapter from a controller-written local payload file."""
        rank = int(getattr(self, "rank", getattr(self, "local_rank", 0)))
        logger.debug("[LoRA-FILE] rank=%d torch_load_begin path=%s", rank, payload_path)
        lora_tensors = torch.load(payload_path, map_location="cpu")
        logger.debug("[LoRA-FILE] rank=%d tensors=%d torch_load_complete", rank, len(lora_tensors))
        return self._diffrl_install_lora_tensors(
            lora_name,
            lora_int_id,
            lora_path,
            peft_config,
            lora_tensors,
            ready_token=ready_token,
        )

    def _diffrl_install_lora_tensors(
        self,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
        peft_config: dict,
        lora_tensors,
        *,
        ready_token: Optional[str],
    ) -> bool:
        rank = int(getattr(self, "rank", getattr(self, "local_rank", 0)))
        if not isinstance(lora_tensors, dict):
            raise TypeError(
                f"{type(self).__name__}: deserialised lora_tensors expected dict, got {type(lora_tensors).__name__}"
            )
        peft_config = dict(peft_config or {})
        pipeline = getattr(getattr(self, "model_runner", None), "pipeline", None)
        transformer = getattr(pipeline, "transformer", None)
        if (
            transformer is not None
            and hasattr(transformer, "blocks")
            and not hasattr(transformer, "transformer_blocks")
        ):
            lora_tensors, peft_config, renamed = remap_minimax_h3_lora(lora_tensors, peft_config)
            validate_minimax_h3_lora_coverage(
                lora_tensors,
                block_count=len(transformer.blocks),
            )
            logger.info(
                "[LoRA-REMAP] rank=%d trainer=transformer_blocks rollout=blocks renamed=%d",
                rank,
                renamed,
            )
        from unirl.rollout.engine.vllm_omni.patches.runtime import (
            OmniTensorLoRARequest,
        )

        request = OmniTensorLoRARequest(
            lora_name=str(lora_name),
            lora_int_id=int(lora_int_id),
            lora_path=str(lora_path),
            peft_config=peft_config,
            lora_tensors=lora_tensors,
        )
        # Keep replacement and installation in the same worker RPC. The
        # controller intentionally does not run a separate all-rank remove,
        # which would reintroduce a status collective around dynamic wrapping.
        self.remove_lora(int(lora_int_id))
        logger.debug("[LoRA-INSTALL] rank=%d add_lora_begin", rank)
        added = bool(self.add_lora(request))
        logger.debug("[LoRA-INSTALL] rank=%d add_lora_complete added=%s", rank, added)
        if not added:
            raise RuntimeError(f"failed to install LoRA adapter {lora_int_id} on rank {getattr(self, 'rank', '?')}")
        wrapped = len(getattr(getattr(self, "lora_manager", None), "_lora_modules", {}))
        if wrapped <= 0:
            raise RuntimeError(f"LoRA adapter {lora_int_id} wrapped zero rollout layers on rank {rank}")
        logger.info("[LoRA-INSTALL] rank=%d wrapped_layers=%d", rank, wrapped)
        if ready_token:
            marker = f"/tmp/diffrl_lora_ready_{ready_token}_{rank}"
            with open(marker, "w", encoding="utf-8") as handle:
                handle.write("ready\n")
        return True

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
        gather_all_ranks: bool = False,
    ) -> object:
        """Full-byte SHA-256 of the worker's loaded LoRA adapter tensors."""
        from unirl.distributed.weight_sync.transfer.checksum import (
            fingerprint_tensor,
        )

        manager_owner = getattr(self, "lora_manager", None) or getattr(
            getattr(self, "model_runner", None), "lora_manager", None
        )
        if manager_owner is None:
            return {}
        manager = getattr(manager_owner, "_adapter_manager", manager_owner)
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

        # Also fingerprint the active GPU buffers that Punica reads during
        # forward. Registered LoRAModel tensors can be correct while packed
        # physical layers are reset, mis-sliced, or bound to the wrong logical
        # submodule.
        lora_modules = getattr(manager, "_lora_modules", None) or getattr(manager_owner, "_lora_modules", {})
        packed_sublayers = getattr(manager, "_get_packed_sublayer_suffixes", None)
        get_lora_weights = getattr(manager, "_get_lora_weights", None)
        active_adapter_id = getattr(manager, "_active_adapter_id", None)
        if active_adapter_id is not None and int(active_adapter_id) == int(adapter_id) and callable(get_lora_weights):
            for physical_name, physical_layer in lora_modules.items():
                a_stacked = getattr(physical_layer, "lora_a_stacked", ())
                b_stacked = getattr(physical_layer, "lora_b_stacked", ())
                n_slices = int(getattr(physical_layer, "n_slices", len(a_stacked)))
                if not isinstance(a_stacked, (list, tuple)) or not isinstance(b_stacked, (list, tuple)):
                    continue
                if len(a_stacked) != n_slices or len(b_stacked) != n_slices:
                    continue
                prefix, _, packed_suffix = str(physical_name).rpartition(".")
                sub_suffixes = (
                    packed_sublayers(packed_suffix, n_slices) if n_slices > 1 and callable(packed_sublayers) else None
                )
                logical_names = (
                    [f"{prefix}.{suffix}" if prefix else suffix for suffix in sub_suffixes]
                    if sub_suffixes
                    else [str(physical_name)]
                )
                if len(logical_names) != n_slices:
                    continue

                expected_layers = [get_lora_weights(lora_model, logical_name) for logical_name in logical_names]
                if any(
                    not isinstance(getattr(layer, "lora_a", None), torch.Tensor)
                    or not isinstance(getattr(layer, "lora_b", None), torch.Tensor)
                    for layer in expected_layers
                ):
                    continue
                active_scale = float(getattr(manager, "_adapter_scales", {}).get(int(adapter_id), 1.0))
                expected_a_arg = [layer.lora_a for layer in expected_layers]
                expected_b_arg = [layer.lora_b * active_scale for layer in expected_layers]
                if n_slices == 1:
                    expected_a_arg = expected_a_arg[0]
                    expected_b_arg = expected_b_arg[0]
                if int(getattr(physical_layer, "tp_size", 1)) > 1:
                    expected_a_arg = physical_layer.slice_lora_a(expected_a_arg)
                    expected_b_arg = physical_layer.slice_lora_b(expected_b_arg)
                expected_a_slices = expected_a_arg if isinstance(expected_a_arg, list) else [expected_a_arg]
                expected_b_slices = expected_b_arg if isinstance(expected_b_arg, list) else [expected_b_arg]
                if len(expected_a_slices) != n_slices or len(expected_b_slices) != n_slices:
                    continue

                for slice_index, logical_name in enumerate(logical_names):
                    if target is not None and logical_name not in target:
                        continue
                    expected_a = expected_a_slices[slice_index]
                    expected_b = expected_b_slices[slice_index]
                    if not isinstance(expected_a, torch.Tensor) or not isinstance(expected_b, torch.Tensor):
                        continue
                    active_a = a_stacked[slice_index][
                        0,
                        0,
                        : expected_a.shape[0],
                        : expected_a.shape[1],
                    ]
                    active_b = b_stacked[slice_index][
                        0,
                        0,
                        : expected_b.shape[0],
                        : expected_b.shape[1],
                    ]
                    fields = out.setdefault(logical_name, {})
                    fields["active_expected_lora_a"] = fingerprint_tensor(expected_a)
                    fields["active_expected_lora_b"] = fingerprint_tensor(expected_b)
                    fields["active_lora_a"] = fingerprint_tensor(active_a)
                    fields["active_lora_b"] = fingerprint_tensor(active_b)
        if gather_all_ranks:
            import torch.distributed as dist

            if dist.is_initialized():
                gathered: list[object] = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, out)
                return gathered
            return [out]
        return out


__all__ = [
    "BucketedIPCReceiveMixin",
    "remap_minimax_h3_lora",
    "validate_minimax_h3_lora_coverage",
]
