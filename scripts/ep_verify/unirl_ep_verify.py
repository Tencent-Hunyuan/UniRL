"""Verify UniRL's VeOmni-backend EP path on a real MoE model.

Drives the *actual* UniRL functions touched by the EP change:
  * the init_parallel_state(...) call exactly as VeOmniBackend.__init__ issues it
    (with extra_parallel_sizes=(ep,)) — i.e. FSDPConfig.ep_size in action;
  * unirl.train.backend.veomni.wrap.veomni_parallelize  (the real wrap);
  * unirl.train.backend.veomni.state.clip_grad_norm      (the EP-aware clip).

Builds a VeOmni-patched Qwen3-MoE (random init on meta — no checkpoint needed),
runs fwd/bwd/clip/step, and records per-step time + per-GPU peak memory so we can
compare ep_size=1 (pure FSDP) vs ep_size>1 (expert-parallel).

Launch: torchrun --nproc_per_node=8 unirl_ep_verify.py <config.json> <ep_size> <out.json>
"""

import json
import os
import sys
import time

import torch
import torch.distributed as dist

from unirl.train.backend.veomni.state import clip_grad_norm
from unirl.train.backend.veomni.wrap import veomni_parallelize

# --- UniRL code under test ---
from unirl.train.configs import FSDPConfig


def main():
    config_path = sys.argv[1]
    ep_size = int(sys.argv[2])
    out_path = sys.argv[3]
    sp_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    steps = int(os.environ.get("STEPS", "8"))
    seq_len = int(os.environ.get("SEQ", "4096"))

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda")

    # FSDPConfig is the real config object the backend consumes; ep_size is the
    # field added for EP.
    fsdp_cfg = FSDPConfig(
        param_dtype="bf16", fsdp_mode="full", reshard_after_forward=True, sp_size=sp_size, ep_size=ep_size
    )

    # ---- EXACT init_parallel_state call from VeOmniBackend.__init__ ----
    from unirl.train.backend.veomni import _compat

    _compat.ensure_qwen3_moe_installed()
    from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state

    if world % fsdp_cfg.sp_size != 0:
        raise ValueError("world not divisible by sp")
    if fsdp_cfg.ep_size > 1 and world % fsdp_cfg.ep_size != 0:
        raise ValueError("world not divisible by ep")
    init_parallel_state(
        dp_size=world // fsdp_cfg.sp_size,
        ulysses_size=fsdp_cfg.sp_size,
        extra_parallel_sizes=(fsdp_cfg.ep_size,),
        extra_parallel_names=("ep",),
        extra_parallel_placement_innermost=(False,),
        dp_mode="fsdp2",
        device_type="cuda",
    )
    ps = get_parallel_state()
    ep_enabled = ps.ep_enabled
    if rank == 0:
        print(
            f"[verify] world={world} sp={sp_size} ep={ep_size} ep_enabled={ep_enabled} "
            f"ep_size(ps)={ps.ep_size if ep_enabled else 1}",
            flush=True,
        )

    # ---- build VeOmni MoE model on meta (random init, no checkpoint) ----
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

    ops = OpsImplementationConfig(
        attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"),
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
    # model must expose get_parallel_plan for EP (Shard(0) experts)
    has_plan = getattr(model, "get_parallel_plan", None) is not None
    if rank == 0:
        print(f"[verify] model={type(model).__name__} has_parallel_plan={has_plan}", flush=True)

    # ---- the REAL UniRL wrap (forwards to parallelize_model_fsdp2 -> applies EP) ----
    veomni_parallelize(
        model,
        block_class_names=("Qwen3MoeDecoderLayer",),
        param_dtype=fsdp_cfg.param_dtype,
        reshard_after_forward=fsdp_cfg.reshard_after_forward,
    )
    has_ep_groups = hasattr(model, "_extra_parallel_param_groups")
    if rank == 0:
        print(f"[verify] wrapped; _extra_parallel_param_groups={has_ep_groups}", flush=True)

    # Mirror UniRL's real build_optimizer EXACTLY: a single AdamW with
    # foreach=False. The single-tensor (per-param) kernel steps each DTensor
    # independently, so EP-sharded experts (ep_fsdp mesh) and non-EP params
    # (dp_shard mesh) never get stacked across meshes — UniRL's existing
    # optimizer already handles EP with no change.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, foreach=False)

    vocab = model.config.vocab_size
    records = []
    last = None
    for step in range(steps):
        input_ids = torch.randint(0, min(vocab, 1024), (1, seq_len), device=device)
        labels = input_ids.clone()
        optimizer.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss
        loss.backward()
        gn = clip_grad_norm(model, 1.0)
        optimizer.step()
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
                f"[verify] step {step} loss={float(loss.detach()):.4f} gn={float(gn):.4f} peak={peak:.2f}GB dt={dt}",
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
            "sp_size": sp_size,
            "world": world,
            "ep_enabled": bool(ep_enabled),
            "has_parallel_plan": bool(has_plan),
            "has_ep_param_groups": bool(has_ep_groups),
            "global_peak_alloc_gb": global_peak,
            "median_step_time_s": med,
            "records": records,
        }
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[verify] WROTE {out_path}: ep={ep_size} peak={global_peak:.2f}GB median_step={med}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
