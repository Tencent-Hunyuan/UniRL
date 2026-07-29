"""REAL UniRL VeOmniBackend EP verification.

Unlike unirl_ep_verify.py (which replicated the init call + drove the wrap/clip
functions), this drives the **actual** ``VeOmniBackend`` class end to end:

    VeOmniBackend.__init__  (init_parallel_state w/ ep -> veomni_parallelize ->
        _attach_extra_parallel_param_groups -> load_trainable_weights ->
        build optimizer/scheduler)
    -> backend.zero_grad / loss.backward / backend.optimizer_step (EP-aware clip)

on a VeOmni-patched Qwen3-MoE (meta-init + real stacked safetensors load).

Launch: torchrun --nproc_per_node=8 unirl_ep_backend_real.py <config_dir> <ep> <out.json>
"""

import json
import os
import sys
import time

import torch
import torch.distributed as dist

from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.veomni.backend import VeOmniBackend
from unirl.train.configs import FSDPConfig


class _SimpleBundle:
    """Minimal meta-init bundle: a VeOmni MoE transformer + stashed weights dir.

    Satisfies the duck-typed contract VeOmniBackend uses:
    ``.transformer`` (resolve_trainable_module fallback) and
    ``._transformer_weights_path`` (load_trainable_weights Pattern B).
    """

    def __init__(self, transformer, weights_dir):
        self.transformer = transformer
        self._transformer_weights_path = weights_dir

    def prepare_for_expert_parallel(self):
        if not callable(getattr(self.transformer, "get_parallel_plan", None)):
            raise RuntimeError("test transformer does not expose get_parallel_plan()")


def build_meta_moe(config_path):
    from unirl.models.types.meta_init import finalize_meta_init
    from unirl.train.backend.veomni import _compat

    _compat.ensure_qwen3_moe_installed()
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

    ops = OpsImplementationConfig(
        attn_implementation="flash_attention_2",
        moe_implementation="fused_triton",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="bfloat16",
        init_device="meta",
        ops_implementation=ops,
    )
    return finalize_meta_init(model, dtype=torch.bfloat16)


def validate_recovered_rope(model):
    """Assert that the backend restored finite, nonzero RoPE frequencies."""
    n = 0
    for name, module in model.named_modules():
        inv_freq = getattr(module, "inv_freq", None)
        if inv_freq is None:
            continue
        local = inv_freq.to_local() if hasattr(inv_freq, "to_local") else inv_freq
        if not bool(torch.isfinite(local).all()) or int(torch.count_nonzero(local)) == 0:
            raise RuntimeError(f"backend did not recover RoPE inv_freq for {name!r}")
        n += 1
    if n == 0:
        raise RuntimeError("test model has no RoPE inv_freq buffer to validate")
    return n


def main():
    config_dir = sys.argv[1]  # dir containing config.json + stacked model.safetensors
    ep_size = int(sys.argv[2])
    out_path = sys.argv[3]
    steps = int(os.environ.get("STEPS", "8"))
    seq_len = int(os.environ.get("SEQ", "4096"))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    rank = int(os.environ.get("RANK", "0"))

    bundle = _SimpleBundle(build_meta_moe(config_dir), config_dir)

    fsdp_cfg = FSDPConfig(param_dtype="bf16", fsdp_mode="full", reshard_after_forward=True, ep_size=ep_size)
    opt_cfg = OptimizerConfig(learning_rate=1e-4, adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8, weight_decay=0.0)
    sched_cfg = LrSchedulerConfig(type="constant", warmup_steps=0, total_steps=100)

    # ---- the REAL backend: full __init__ runs init_parallel_state(ep) + wrap + load + optimizer ----
    backend = VeOmniBackend(
        bundle=bundle,
        block_class_names=("Qwen3MoeDecoderLayer",),
        fsdp_cfg=fsdp_cfg,
        optimizer_cfg=opt_cfg,
        scheduler_cfg=sched_cfg,
        trainable_attr="transformer",
        device=device,
        rank=rank,
    )
    model = backend.model
    n_rope = validate_recovered_rope(model)

    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_enabled = ps.ep_enabled
    has_groups = hasattr(model, "_extra_parallel_param_groups")
    if rank == 0:
        print(
            f"[real] ep={ep_size} ep_enabled={ep_enabled} ep_size(ps)={ps.ep_size if ep_enabled else 1} "
            f"ep_param_groups={has_groups} rope_validated={n_rope} model={type(model).__name__}",
            flush=True,
        )

    vocab = model.config.vocab_size
    records = []
    last = None
    for step in range(steps):
        input_ids = torch.randint(0, min(vocab, 1024), (1, seq_len), device=device)
        labels = input_ids.clone()
        backend.zero_grad()
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        gn = backend.optimizer_step(max_grad_norm=1.0)  # EP-aware clip + step + sched + ema
        torch.cuda.synchronize()
        now = time.perf_counter()
        dt = None if last is None else now - last
        last = now
        peak = torch.cuda.max_memory_allocated() / 1e9
        records.append(
            {"step": step, "time_s": dt, "peak_alloc_gb": peak, "loss": float(loss.detach()), "grad_norm": float(gn)}
        )
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(
                f"[real] step {step} loss={float(loss.detach()):.4f} gn={float(gn):.4f} peak={peak:.2f}GB dt={dt}",
                flush=True,
            )

    local_peak = max(r["peak_alloc_gb"] for r in records)
    t = torch.tensor([local_peak], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    global_peak = t.item()

    if rank == 0:
        steady = [r["time_s"] for r in records if r["time_s"] is not None and r["step"] >= 3]
        med = sorted(steady)[len(steady) // 2] if steady else None
        out = {
            "ep_size": ep_size,
            "world": dist.get_world_size(),
            "ep_enabled": bool(ep_enabled),
            "ep_param_groups": bool(has_groups),
            "rope_validated": n_rope,
            "global_peak_alloc_gb": global_peak,
            "median_step_time_s": med,
            "records": records,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[real] WROTE {out_path}: ep={ep_size} peak={global_peak:.2f}GB median_step={med}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
