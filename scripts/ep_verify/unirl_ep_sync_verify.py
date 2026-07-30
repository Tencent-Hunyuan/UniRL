"""Verify EP-aware weight sync round-trips the EP-sharded model to HF per-expert
format (no SGLang needed).

Builds the real VeOmniBackend (ep_size>1) on a MoE checkpoint, runs the ACTUAL
EP weight walk (FullWeightSync._iter_full_tensors_ep), and checks that the emitted
HF per-expert tensors (experts.{e}.gate_proj/up_proj/down_proj) exactly match the
original checkpoint's per-expert weights (cast to the train dtype). This proves the
ep all-gather + stacked->per-expert reverse-convert is correct, which is the only
EP-specific part of pushing weights into a rollout engine.

Launch: torchrun --nproc_per_node=8 unirl_ep_sync_verify.py <split_ckpt_dir> <ep>
"""

import sys
import types

import torch
import torch.distributed as dist
from safetensors import safe_open

from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.models.qwen3_moe import Qwen3MoeBundle
from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig
from unirl.train.backend.veomni.backend import VeOmniBackend
from unirl.train.configs import FSDPConfig


def main():
    ckpt = sys.argv[1]
    ep_size = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda")
    rank = int(__import__("os").environ.get("RANK", "0"))

    tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=0, pad_token="<pad>", eos_token="<eos>")
    bundle = Qwen3MoeBundle.from_config(pretrained_model_ckpt_path=ckpt, tokenizer=tok)
    backend = VeOmniBackend(
        bundle=bundle,
        block_class_names=("Qwen3MoeDecoderLayer",),
        fsdp_cfg=FSDPConfig(param_dtype="bf16", fsdp_mode="full", reshard_after_forward=True, ep_size=ep_size),
        optimizer_cfg=OptimizerConfig(
            learning_rate=1e-4, adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8, weight_decay=0.0
        ),
        scheduler_cfg=LrSchedulerConfig(type="constant", warmup_steps=0, total_steps=10),
        trainable_attr="transformer",
        device=device,
        rank=rank,
    )

    # Drive the REAL EP weight walk via a minimal FullWeightSync-shaped object.
    fake = types.SimpleNamespace(_backend=backend, _wire_dtype=None, _name_remap={})
    emitted = {}
    n_expert_keys = 0
    for name, tensor in FullWeightSync._iter_full_tensors_ep(fake):
        if ".experts." in name and any(
            name.endswith(s) for s in (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
        ):
            n_expert_keys += 1
            if rank == 0:
                emitted[name] = tensor.detach().to("cpu", torch.float32)

    if rank == 0:
        f = safe_open(f"{ckpt}/model.safetensors", framework="pt", device="cpu")
        ckpt_keys = set(f.keys())
        # Sample experts across layers/expert-ids; verify bit-exact (file cast to bf16,
        # the train dtype the backend loaded into, then back to fp32 for compare).
        import re

        layer_ids = sorted(
            {int(m.group(1)) for k in ckpt_keys if (m := re.search(r"layers\.(\d+)\.mlp\.experts\.0\.", k))}
        )
        E = 1 + max(int(m.group(1)) for k in ckpt_keys if (m := re.search(r"experts\.(\d+)\.gate_proj", k)))
        checks, ok = 0, 0
        sample_layers = [layer_ids[0], layer_ids[-1]] if layer_ids else []
        sample_experts = sorted({0, 1, E // 2, E - 1})
        for L in sample_layers:
            for e in sample_experts:
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    key = f"model.layers.{L}.mlp.experts.{e}.{proj}.weight"
                    if key not in emitted or key not in ckpt_keys:
                        print(f"[sync] MISSING {key} (emitted={key in emitted}, ckpt={key in ckpt_keys})", flush=True)
                        checks += 1
                        continue
                    ref = f.get_tensor(key).to(torch.bfloat16).to(torch.float32)  # match backend bf16 load
                    got = emitted[key]
                    checks += 1
                    if got.shape == ref.shape and torch.equal(got, ref):
                        ok += 1
                    else:
                        md = (got - ref).abs().max().item() if got.shape == ref.shape else -1
                        print(
                            f"[sync] MISMATCH {key} shape got={tuple(got.shape)} ref={tuple(ref.shape)} maxdiff={md}",
                            flush=True,
                        )
        total_expert_keys = len(layer_ids) * E * 3
        print(
            f"[sync] EP={ep_size} emitted_expert_keys={len(emitted)} (expected {total_expert_keys}); "
            f"bit-exact {ok}/{checks} sampled",
            flush=True,
        )
        print(f"[sync] RESULT: {'PASS' if ok == checks and len(emitted) == total_expert_keys else 'FAIL'}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
