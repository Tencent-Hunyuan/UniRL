"""Shared bucketed-IPC receive mixin for HI3 worker-extension classes.

Both the AR and the DiT stage extensions need the same ``update_weights_from_ipc``
implementation: open a per-rank ``BucketedWeightReceiver`` on the shared ZMQ
socket, then forward each bucket to ``self.load_weights`` (or to
``self.add_lora`` in the LoRA-sync case).

Factor it into a mixin so the AR extension can compose it with
``HI3ARWorkerExtension`` (the existing tokenizer-compat target) and the
DiT extension can compose it with vllm-omni's ``CustomPipelineWorkerExtension``.

The worker ``self`` provides:
- ``self.device`` (cuda device of this worker)
- ``self.local_rank`` (rank within the stage's TP/PP group)
- ``self.load_weights(weights)`` (per-bucket loader)
- ``self.add_lora(req)`` / ``self.remove_lora(int_id)`` (LoRA ops)
"""

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
    """Adds ``update_weights_from_ipc`` (and the LoRA hijack install) to a
    vllm-omni worker via multiple inheritance."""

    def __new__(cls, **kwargs):
        # Run the LoRA hijack once per worker subprocess so add_lora can
        # accept tensor-bag requests (matches verl-omni utils.py:40-46
        # pattern). Safe to call repeatedly — the patch is idempotent.
        VLLMOmniHijack.hijack()
        # The trainer pickles IPC handles whose rebuild fn is the vendored
        # ``_rebuild_cuda_tensor_modified``. Unpickling on this worker imports
        # the SAME function (same module path), which then calls
        # ``reductions._rebuild_cuda_tensor_original`` — only present after
        # ``monkey_patch_torch_reductions()`` has run here too. We use the
        # sglang reductions vendored under unirl because the vllm-omni venv
        # intentionally has no sglang; trainer and worker both import this one
        # module so the CUDA-IPC pickle round-trips.
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            monkey_patch_torch_reductions,
        )

        monkey_patch_torch_reductions()
        return super().__new__(cls)

    # ------------------------------------------------------------------
    # IPC receive
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        stage_id: int = 0,
        replica_rank: Optional[int] = None,
    ) -> None:
        """Receive a state dict over the per-rank ZMQ socket.

        Trainer-side counterpart is
        ``distributed.weight_sync.full.ipc.IPCWeightSync``.

        ``replica_rank`` defaults to ``replica_rank_from_env()`` (v1 behavior);
        the v2 colocated handler passes its train-rank explicitly so colocated
        engines on one node don't collide on the same ZMQ socket path (the Omni
        subprocess spawns before any per-Worker env can be injected, so env is
        not a reliable per-replica discriminator there).
        """
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

    # ------------------------------------------------------------------
    # Per-bucket dispatch — full state-dict vs LoRA tensor-bag
    # ------------------------------------------------------------------

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
        """Forward weights to whichever loader the underlying worker exposes.

        - DiT worker (``DiffusionWorker`` base of ``CustomPipelineWorkerExtension``)
          has ``self.load_weights`` directly — delegates to the pipeline.
        - AR worker (``GPUARWorker``) has no ``load_weights`` on the worker;
          the model sits at ``self.model_runner.model`` and exposes its own
          ``load_weights``.
        """
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

    # ------------------------------------------------------------------
    # SGLang-shape one-bag tensor payload
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list,
        target_modules: Optional[list] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        """Receive a SGLang-shape one-bag payload and load it.

        Picks ``serialized_named_tensors[self.local_rank]``, deserializes via
        sglang's ``MultiprocessingSerializer`` + ``FlattenedTensorBucket``,
        then forwards the reconstructed ``[(name, tensor), ...]`` to
        ``self.load_weights``. Sender is
        :class:`unirl.distributed.weight_sync.full.tensor.TensorWeightSync`.

        Runtime dep: sglang must be installed in the worker subprocess for the
        ``FlattenedTensorBucket`` dataclass to round-trip the pickle. The pod
        venv that runs the rollout actor and the worker shares this dep.
        """
        del target_modules, flush_cache  # accepted for SGLang-shape parity
        # Vendored sglang serializer/bucket (engine venv has no sglang); the
        # TensorWeightSync sender imports the SAME module so the pickle matches.
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
        # MultiprocessingSerializer.deserialize handles the base64+pickle round-trip
        # symmetric with output_str=True on the sender.
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

    # ------------------------------------------------------------------
    # LoRA tensor-bag — driver-supplied dict reconstructed worker-side
    # ------------------------------------------------------------------

    def set_lora_from_tensor_dict(
        self,
        lora_name: str,
        lora_int_id: int,
        lora_path: str,
        peft_config: dict,
        lora_tensors_serialized: str,
    ) -> bool:
        """Reconstruct an ``OmniTensorLoRARequest`` from primitive args
        and forward to ``self.add_lora``.

        Why not pass the request object directly via ``collective_rpc``:

        - vllm's wire encoder (msgspec) doesn't recognise our Struct
          subclass ``OmniTensorLoRARequest`` and decodes it positionally
          as a list, so the worker sees ``[<fields>]`` instead of an
          object with ``.lora_int_id``.
        - The inner ``lora_tensors`` dict values (``torch.Tensor``) also
          can't survive the msgpack wire — tensors come back as plain
          Python lists, and ``LoRAModel.from_lora_tensors`` then trips
          ``'list' object has no attribute 'to'``.

        So the engine ships LoRA tensors via SGLang's
        ``MultiprocessingSerializer`` (same path B.2 uses); the worker
        deserialises into a real ``dict[str, torch.Tensor]`` and rebuilds
        the request locally.
        """
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
        """Byte-copy variant of :meth:`set_lora_from_tensor_dict` for HI3.

        # DELETE-WHEN: the vLLM-Omni LoRA handle transport is TP>1-broadcast-safe
        #   — then ``set_lora_from_tensor_dict`` serves every stage and this
        #   byte-copy mate (+ engine-side ``set_lora_from_tensors_copy``) is dead.

        :meth:`set_lora_from_tensor_dict` ships a zero-copy
        ``MultiprocessingSerializer`` handle, whose one-shot ``file_descriptor``
        ``resource_sharer`` pops after the first consumer — fine for the SD3
        per-worker DP path (TP=1, single consumer) but it makes ranks 2..N of a
        TP>1 stage raise ``KeyError`` / ``EOFError`` when a single
        ``collective_rpc`` broadcasts the same handle to every worker. The HI3
        AR / DiT stages are TP>1, so the driver pushes via
        ``set_lora_from_tensors_copy``, which sends the LoRA as a *data copy*
        (``torch.save`` bytes, base64-wrapped). Each worker ``torch.load``s its
        own independent tensors, so the fan-out is unbounded. LoRA is tiny (tens
        of MB), so copying per rank is free.
        """
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

    # ------------------------------------------------------------------
    # Debug — parameter inspection (used by E2E test)
    # ------------------------------------------------------------------

    def _diffrl_describe_params(
        self,
        names: Optional[list] = None,
    ) -> dict:
        """Return ``{name: (shape_tuple, dtype_str)}`` for the worker's loaded model.

        Used by the E2E test to build synthetic state-dicts that match real
        parameter shapes (so ``load_weights`` actually mutates them).
        """
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
        """Return ``{name: short_sha256_hex}`` for the worker's loaded model.

        Used to assert that a given weight-sync transport actually mutated
        worker-side parameters. Cheap when ``names`` is provided (skips the
        rest); expensive when ``None``.

        Pulls parameters from ``self.model_runner.pipeline`` (DiT) or
        ``self.model_runner.model`` (AR) depending on which attribute exists.
        """
        import hashlib

        runner = getattr(self, "model_runner", None)
        if runner is None:
            return {}
        # DiT worker exposes a pipeline with .named_parameters via its model
        # subobject; AR worker has model_runner.model directly. Try both.
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
            # Hash a small fingerprint so we don't pay full-tensor cost for
            # huge HI3 transformer params: dtype + shape + first/last 256B + numel
            # in deterministic order. SHA over this gives a cheap stable id.
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

    # ------------------------------------------------------------------
    # Post-load value-correctness — full-byte hashes of what actually
    # landed in the model, exposed for the trainer to compare against
    # the ``compute_*_checksums`` helpers in ``weight_sync.checksum``.
    # ------------------------------------------------------------------

    def _diffrl_loaded_param_checksums(
        self,
        names: Optional[list] = None,
    ) -> dict:
        """Full-byte SHA-256 of the worker's loaded parameters.

        Called *after* ``load_weights`` returns to verify the bytes that
        landed match what the trainer intended. Trainer-side counterpart
        is :func:`unirl.distributed.weight_sync.transfer.checksum.compute_param_checksums`.

        TP note: this rank's hash is over the rank's *local* tensor.
        For TP-flat params (layer norms, scalars) every rank holds the
        same full tensor, so per-rank hashes equal the trainer's
        full-tensor hash. For TP-sharded params each rank holds a slice
        and per-rank hashes differ from the trainer's full hash;
        verification of those needs an external all-gather (currently
        deferred — the smoke test only targets TP-flat names).
        """
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
        """Full-byte SHA-256 of the worker's loaded LoRA adapter tensors.

        Walks the inner ``LoRAModelManager._registered_adapters[adapter_id].loras``
        and hashes ``lora_a`` / ``lora_b`` / ``bias`` / ``embeddings_tensor``
        when present. Packed modules store their ``lora_a`` / ``lora_b`` as a
        list of sub-tensors (one per fused projection), which are hashed
        per-shard as ``<field>.<i>``. ``lora.optimize()`` has already run by
        this point — ``lora_b`` is post-scaling — so the trainer must apply the
        matching ``alpha / r`` scaling before hashing for equality. The
        helper :func:`weight_sync.checksum.compute_lora_checksums_post_optimize`
        does that.

        Returns ``{loaded_layer_name: {field: hex, ...}}``. The layer
        name is whatever the manager stores (typically the post-rename
        name PEFTHelper produces, e.g. ``q_proj``, not the trainer's
        ``base_model.model.q_proj``); the smoke test strips the
        ``base_model.model.`` prefix before comparing.

        Worker layouts (DiT vs. AR) put the lora_manager in different
        spots; we probe both. Returns ``{}`` if the adapter id isn't
        registered on this rank — caller should treat that as a
        mismatch, not a silent skip.
        """
        from unirl.distributed.weight_sync.transfer.checksum import (
            fingerprint_tensor,
        )

        manager = getattr(self, "lora_manager", None) or getattr(
            getattr(self, "model_runner", None), "lora_manager", None
        )
        if manager is None:
            return {}
        # vLLM wraps the registry: the outer ``WorkerLoRAManager`` delegates to
        # an inner ``LoRAModelManager`` (``_adapter_manager``) that actually owns
        # ``_registered_adapters``. Reading the outer object returns an empty
        # mapping, so descend when the inner manager is present.
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
                    # Packed modules (``qkv_proj``, ``gate_up_proj``) store one
                    # sub-tensor per fused projection; ``None`` slots mark
                    # absent shards. Hash each present shard separately so the
                    # readback covers the whole fused layer, not just the first.
                    for i, sub in enumerate(t):
                        if isinstance(sub, torch.Tensor):
                            per_field[f"{field}.{i}"] = fingerprint_tensor(sub)
            out[layer_name] = per_field
        return out


__all__ = ["BucketedIPCReceiveMixin"]
