"""Let sglang's LoRA B slicers take 2-D ``[total_out, rank]`` as well as 3-D ``[N, out_dim, rank]``."""

from __future__ import annotations

_SENTINEL = "_unirl_lora_slice_b_2d_tolerant"


def patch_lora_slice_2d() -> None:
    import sglang.multimodal_gen.runtime.layers.lora.linear as ll

    _patch_merged_column(ll)


def _patch_merged_column(ll) -> None:
    cls = ll.MergedColumnParallelLinearWithLoRA
    if getattr(cls.slice_lora_b_weights, _SENTINEL, False):
        return

    get_tp_rank = ll.get_tp_rank

    def slice_lora_b_weights(self, B):
        tp_rank = get_tp_rank()
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        if B.dim() == 2:
            return B[start_idx:end_idx, :]
        return B[:, start_idx:end_idx, :]

    slice_lora_b_weights._unirl_lora_slice_b_2d_tolerant = True  # type: ignore[attr-defined]
    cls.slice_lora_b_weights = slice_lora_b_weights
