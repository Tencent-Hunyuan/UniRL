from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class EmaLoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    default_adapter: str = "default"
    shadow_adapter: str = "old"
    ema_decay: float = 0.001
    ema_decay_type: str = "constant"
    ema_flat_steps: int = 0
    ema_uprate: float = 0.001
    ema_uphold: float = 0.5
    timing: str = "rollout_end"


@dataclass
class EmaFullConfig:
    target_decay: float = 0.9999
    timing: str = "optimizer_step"
    shadow_prefix: str = "shadow_"


@dataclass
class FSDPConfig:
    param_dtype: str = "bf16"
    cpu_offload: bool = False
    mixed_precision: bool = True
    fsdp_mode: str = "full"
    reshard_after_forward: bool = True
    activation_checkpointing: bool = False
    use_torch_compile: bool = False
    # Opt-in no-sync gradient accumulation: defer the per-block FSDP2 gradient
    # reduce-scatter to the last micro-batch of an optimizer step, so one
    # reduce-scatter runs per step instead of one per micro-batch (the standard
    # FSDP2 no-sync pattern; a multi-node win, ~no-op over NVLink). Only takes
    # effect under ZeRO-2 (reshard_after_forward=False); ignored otherwise. NOT
    # bit-identical to the per-micro path: the deferred grads accumulate in the
    # unsharded buffers at param dtype (bf16 under mixed precision) across the
    # micro-batches, whereas per-micro sync reduces each in reduce_dtype (fp32) —
    # so reward / grad-norm parity must be confirmed before enabling in a recipe.
    defer_grad_sync: bool = False
    # Opt-in FSDP2 cross-block forward prefetch: overlap each block's all-gather
    # with compute (a multi-node win, ~no-op over NVLink). Chains the default root
    # wrap to prefetch block 0, then block i to prefetch block i+1; needs the root
    # wrap (raises if root_wrap=False). Off keeps the default wrap with no prefetch.
    forward_prefetch: bool = False
    # Optional high-precision master dtype for the TRAINABLE params (e.g. "fp32").
    # When set, fsdp_wrap keeps the trainable (LoRA) sharded master + optimizer
    # states at this dtype while MixedPrecisionPolicy(param_dtype) still casts the
    # all-gathered compute copy to param_dtype (bf16) — the standard "fp32 master +
    # bf16 compute" recipe flow_grpo relies on for stable low-gradient RL. None
    # (default) = master dtype follows param_dtype (the prior all-bf16 behavior).
    master_dtype: Optional[str] = None
    # Root fully_shard (default ON): after the per-block wrap, fully_shard the
    # model root so the leftover params (embed / final norm / lm_head) are
    # sharded + mp_policy'd like everything else instead of staying plain
    # replicated tensors (which need manual grad sync and keep full fp32
    # masters per rank). Set false for models whose stages call submodules of
    # the wrapped object directly outside a root forward (bagel's vendored
    # code) or whose wrapped object carries frozen mixed-dtype sibling
    # sub-models that must not be sharded (hunyuan_image3's VAE/ViT).
    root_wrap: bool = True
    # Checkpoint storage backend. "torch" (default) keeps the legacy path:
    # gather a full state dict to rank 0 and torch.save a single checkpoint.pt.
    # "dcp" uses torch.distributed.checkpoint sharded save/load — each rank
    # reads/writes only its own shard (no rank-0 full-tensor gather), which
    # enables meta-init bundles (80B) to checkpoint, parallelizes I/O across
    # ranks, and resumes under a different world size. load auto-detects the
    # on-disk format, so legacy checkpoint.pt dirs still resume regardless.
    checkpoint_format: str = "torch"
    # dcp only: background (async) save so the train loop is not blocked on the
    # I/O. Ignored under checkpoint_format="torch". (Wired in a later phase.)
    checkpoint_async: bool = False
    # Ulysses sequence-parallel degree (default 1 = disabled, a true no-op).
    # When >1 the VeOmni backend builds a folded dp_shard x ulysses FSDP mesh
    # (init_parallel_state(ulysses_size=sp_size, dp_size=world//sp_size)) and
    # installs the per-architecture SP patch (slice the sequence across sp_size
    # ranks, all-to-all in attention) — see unirl.train.backend.veomni.sp.
    # Must divide the world size and the model's attention head count. Only the
    # VeOmni backend honors it; FSDPBackend ignores it.
    sp_size: int = 1
    # Expert-parallel degree (default 1 = disabled); when >1 the VeOmni backend shards fused
    # experts over a separate mesh and requires the model to expose get_parallel_plan(). VeOmni only.
    ep_size: int = 1


__all__ = [
    "LoraConfig",
    "EmaLoraConfig",
    "EmaFullConfig",
    "FSDPConfig",
]
