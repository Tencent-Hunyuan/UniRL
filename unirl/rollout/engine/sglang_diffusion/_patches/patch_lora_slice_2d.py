"""Make sglang's Merged/QKV-parallel LoRA B slicers tolerate 2-D tensors (LIN-365).

sglang's ``MergedColumnParallelLinearWithLoRA.slice_lora_b_weights`` expects a
3-D ``[N, out_dim, rank]`` LoRA B tensor (fork's stacked format for split-fused
params; see ``patch_lora_tensors._register_lora_state_dict`` merge_index branch).
For models where diffusers PEFT delivers ``ff.linear_in`` / ``to_qkv_mlp_proj``
as a SINGLE 2-D ``[total_out, rank]`` LoRA B (the gate/up or qkv/mlp split is
internal to the weight matrix, not separate adapter keys), the 3-D slicer
crashes ``IndexError: too many indices for tensor of dimension 2``. Hit by
FLUX.2-Klein's ``Flux2FeedForward.linear_in`` on the 2nd rollout (after the 1st
weight-sync pushes the trainer LoRA tensors).

This patch makes ``MergedColumnParallelLinearWithLoRA.slice_lora_b_weights``
tolerant of both shapes:
  * 3-D ``[N, out_dim, rank]``: keep upstream behaviour (slice axis-1 by TP).
  * 2-D ``[total_out, rank]``: slice axis-0 by TP, identical to plain
    ``ColumnParallelLinearWithLoRA``. The downstream forward math
    (``input @ A.T @ B.T``) works the same for both shapes when TP=1.

For TP>1 the 2-D path treats the merged output dim as a single contiguous
shard (same as ColumnParallel). This matches diffusers' PEFT semantics. Rollout
runs TP=1 per worker (one GPU per actor), so this is sound for the LIN-365
flux2_klein path.
"""

from __future__ import annotations

_SENTINEL = "_unirl_lora_slice_b_2d_tolerant"


def patch_lora_slice_2d() -> None:
    import sglang.multimodal_gen.runtime.layers.lora.linear as ll

    _patch_merged_column(ll)


def _patch_merged_column(ll) -> None:
    cls = ll.MergedColumnParallelLinearWithLoRA
    if getattr(cls.slice_lora_b_weights, _SENTINEL, False):
        return

    # Resolve get_tp_rank symbol via the module that defines it; ll already
    # imports it at module load time (used by the existing slicer above).
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
