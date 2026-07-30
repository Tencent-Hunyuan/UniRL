"""REAL GRPO training-side step on an EP MoE, through the actual UniRL stack.

Pipeline (all real UniRL classes):
    Qwen3MoeBundle  (VeOmni MoE, meta-init, EP-capable)
      -> VeOmniBackend(ep_size=N)            # full __init__: EP shard + load + optimizer
      -> Qwen3ARStage(model=bundle)          # installs the replay forward
      -> GRPO(stage=...).compute_loss_and_backward(conds, segment, advantages, ...)
            == stage.replay (policy fwd on EP MoE) + PPO clip loss + backward
      -> backend.optimizer_step(max_grad_norm)   # EP-aware clip + step + sched

Rollout + reward are synthesized (random prompts / responses / advantages) — EP
only affects the TRAINING backend, not the reward signal. old_logp is seeded
from a no-grad replay so the step-0 ratio == 1 (a clean GRPO ratio).

Launch: torchrun --nproc_per_node=8 unirl_grpo_ep_real.py <stacked_cfg_dir> <ep> <out.json>
"""

import json
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

from unirl.algorithms.grpo import GRPO
from unirl.models.qwen3.ar import Qwen3ARStage
from unirl.models.qwen3.conditions import Qwen3ARConditions
from unirl.models.qwen3_moe import Qwen3MoeBundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.veomni.backend import VeOmniBackend
from unirl.train.configs import FSDPConfig
from unirl.types.conditions import TextTokenCondition
from unirl.types.segments.text import TextSegment


def main():
    cfg_dir = sys.argv[1]
    ep_size = int(sys.argv[2])
    out_path = sys.argv[3]
    steps = int(os.environ.get("STEPS", "6"))
    B = int(os.environ.get("B", "4"))  # prompts (== global batch on this single DP group)
    P = int(os.environ.get("P", "64"))  # prompt len
    R = int(os.environ.get("R", "128"))  # response len

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    rank = int(os.environ.get("RANK", "0"))

    # Stub tokenizer: vocab-4096 toy model needs pad_id < vocab (a real Qwen
    # tokenizer's pad id 151643 would index past the toy embedding).
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=0, pad_token="<pad>", eos_token="<eos>")
    bundle = Qwen3MoeBundle.from_config(pretrained_model_ckpt_path=cfg_dir, tokenizer=tok)

    fsdp_cfg = FSDPConfig(param_dtype="bf16", fsdp_mode="full", reshard_after_forward=True, ep_size=ep_size)
    backend = VeOmniBackend(
        bundle=bundle,
        block_class_names=("Qwen3MoeDecoderLayer",),
        fsdp_cfg=fsdp_cfg,
        optimizer_cfg=OptimizerConfig(
            learning_rate=1e-4, adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8, weight_decay=0.0
        ),
        scheduler_cfg=LrSchedulerConfig(type="constant", warmup_steps=0, total_steps=100),
        trainable_attr="transformer",
        device=device,
        rank=rank,
    )

    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_enabled = ps.ep_enabled

    stage = Qwen3ARStage(model=bundle)
    grpo = GRPO(stage=stage, conditions_cls=Qwen3ARConditions, clip_range=0.2, sampling_temperature=1.0)

    vocab = bundle.transformer.config.vocab_size
    hi = min(vocab, 1024)
    # Fixed synthetic rollout (same across ep sizes via seeded RNG → comparable).
    g = torch.Generator(device="cpu").manual_seed(1234)
    prompt_ids = torch.randint(0, hi, (B, P), generator=g).to(device)
    attn = torch.ones((B, P), dtype=torch.long, device=device)
    conds = {"prompt": TextTokenCondition(input_ids=prompt_ids, attention_mask=attn)}
    resp = [torch.randint(0, hi, (R,), generator=g).to(device) for _ in range(B)]

    # Seed old_logp from a no-grad replay so step-0 ratio == 1 (clean GRPO).
    seg0 = TextSegment.pack(tokens=resp, log_probs=[torch.zeros(R, device=device) for _ in range(B)])
    with torch.no_grad():
        old_flat = stage.replay(Qwen3ARConditions.from_dict(conds), segment=seg0, temperature=1.0)
    old_lists, off = [], 0
    for _ in range(B):
        old_lists.append(old_flat[off : off + R].float().cpu())
        off += R
    segment = TextSegment.pack(tokens=[r.cpu() for r in resp], log_probs=old_lists)
    advantages = torch.randn(B, generator=g)

    records = []
    last = None
    for step in range(steps):
        backend.zero_grad()
        res = grpo.compute_loss_and_backward(
            conditions=conds,
            segment=segment,
            advantages=advantages,
            training_progress=0.0,
            loss_scale=1.0,
        )
        gn = backend.optimizer_step(max_grad_norm=1.0) if res.has_backward else float("nan")
        torch.cuda.synchronize()
        now = time.perf_counter()
        dt = None if last is None else now - last
        last = now
        peak = torch.cuda.max_memory_allocated() / 1e9
        m = res.metrics
        records.append(
            {
                "step": step,
                "time_s": dt,
                "peak_alloc_gb": peak,
                "policy_loss": m.get("policy_loss"),
                "grad_norm": float(gn),
                "ratio_mean": m.get("ratio_mean"),
                "logp_absdiff": m.get("rollout_replay_logp_absdiff"),
            }
        )
        torch.cuda.reset_peak_memory_stats()
        if rank == 0:
            print(
                f"[grpo] step {step} loss={m.get('policy_loss'):.5f} gn={float(gn):.4f} "
                f"ratio_mean={m.get('ratio_mean')} peak={peak:.2f}GB dt={dt}",
                flush=True,
            )

    local_peak = max(r["peak_alloc_gb"] for r in records)
    t = torch.tensor([local_peak], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if rank == 0:
        steady = [r["time_s"] for r in records if r["time_s"] is not None and r["step"] >= 2]
        med = sorted(steady)[len(steady) // 2] if steady else None
        out = {
            "ep_size": ep_size,
            "ep_enabled": bool(ep_enabled),
            "world": dist.get_world_size(),
            "global_peak_alloc_gb": t.item(),
            "median_step_time_s": med,
            "records": records,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[grpo] WROTE {out_path}: ep={ep_size} peak={t.item():.2f}GB median_step={med}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
