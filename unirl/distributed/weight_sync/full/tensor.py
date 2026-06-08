"""v2 full-weight tensor-payload sync (COLOCATE).

Pushes the trained FSDP full base weights into a co-located vLLM-Omni rollout
engine by serializing each bucket (SGLang ``FlattenedTensorBucket`` +
``MultiprocessingSerializer``) and handing it to the local engine sibling's
``update_weights_from_tensor`` — the engine owns the Worker→Omni-subprocess
transfer (serialize already done; ``collective_rpc`` fans to the stage workers).

Full-weight analogue of ``weight_sync/lora/local.py:LocalLoraWeightSync`` and the v2
transport-mate of v1 ``distributed/weight_sync/tensor.py``. Colocate only:
``backend`` and ``rollout`` arrive as LOCAL siblings (same Worker process), so
the v1 cross-rank ``gather_object``/gloo-subgroup logic is unnecessary — each
train rank ships to its own co-located engine, and (TP=1) the worker picks
``serialized_named_tensors[0]``.

Scope: single-node, TP=1; a single-model engine, or one child of a
``ComposedRolloutEngine`` (via ``track_prefix``). All model / sglang imports are
deferred so the driver can import this module for ``remote(...)``.
"""

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
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout  # local engine sibling (single-model, or a ComposedRolloutEngine)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Serialize each bucket and load it into the local engine.

        Runs on every train rank (``BROADCAST``); the ``raw_state_dict`` walk
        all-gathers each shard on every rank in lockstep. Each rank talks to
        its own co-located engine, so no cross-rank gather is needed.
        """
        try:
            from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        except ImportError:
            from sglang.srt.patch_torch import monkey_patch_torch_reductions
        from sglang.srt.utils import MultiprocessingSerializer

        try:
            from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
        except ImportError:
            from sglang.srt.model_executor.model_runner import FlattenedTensorBucket

        import torch

        monkey_patch_torch_reductions()

        for bucket, is_last in self._iter_buckets():
            # Group by dtype, one FlattenedTensorBucket per dtype (matches the
            # receiver's flattened_bucket load_format).
            by_dtype: dict = {}
            for name, tensor in bucket:
                # Tensors arrive already at the wire dtype: ``wire_dtype`` (sync
                # config) is applied once in the base-class walk, shard-side.
                by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            serialized = []
            for grouped in by_dtype.values():
                flat = FlattenedTensorBucket(named_tensors=grouped)
                payload = {
                    "flattened_tensor": flat.get_flattened_tensor(),
                    "metadata": flat.get_metadata(),
                }
                serialized.append(MultiprocessingSerializer.serialize(payload, output_str=True))

            n_dtypes = len(serialized)
            for i, payload in enumerate(serialized):
                # TP=1 → the worker picks serialized_named_tensors[0], so ship a
                # single-element list. flush only on the very last payload.
                self._rollout.update_weights_from_tensor(
                    serialized_named_tensors=[payload],
                    load_format="flattened_bucket",
                    flush_cache=(self._flush_cache and is_last and i == n_dtypes - 1),
                    track_prefix=self._track_prefix,
                )
            # Release the all-gathered full tensors + IPC payloads for this bucket
            # before gathering the next — else the full model (~13GB) accumulates
            # in the caching allocator and OOMs the colocated SRT server.
            del serialized, by_dtype, bucket
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.weight_version += 1


__all__ = ["TensorWeightSync"]
