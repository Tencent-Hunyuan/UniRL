#!/usr/bin/env python
"""Phase 0 probe — validate unirl's ``enable_grad()`` distributed autograd for ReFL.

ReFL / direct-reward-backprop needs gradients to flow END-TO-END across two
roles: ``policy.sample`` (the model) → ``reward.score`` (a differentiable reward)
→ ``-reward.mean()`` → back onto the policy weights. unirl already ships the
controller-side autograd that could carry this (``enable_grad()`` /
``GradContext``), but it has no consumers, no tests, and has never run a real
backward. This probe is the gate: does it land correct ``.grad`` on the policy
weights, including under FSDP and across ASYMMETRIC data-parallel layouts
(policy dp=N, reward dp=M, N != M)?

The check is numerical: a single-process plain-``torch.autograd`` reference (the
oracle) computes ``g_ref``; the distributed two-role run must reproduce it.

  - replicated policy (no FSDP): each dp worker holds the full weights and its
    own shard gradient; the mean over workers must equal ``g_ref``.
  - FSDP policy: reduce-scatter already averages across dp ranks, so the
    reconstructed full gradient (``DTensor.full_tensor``) must equal ``g_ref``.

Run ONE config per process (keeps Ray / NCCL / FSDP default-PG state clean):

    python scripts/phase0_gradctx_probe.py --policy-gpus 4 --reward-gpus 1 --fsdp 1

Exit code 0 = PASS, 1 = FAIL/error.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
import torch.nn as nn

import ray

from unirl.distributed.group.device_pool import DevicePool
from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.group.remote import Remote
from unirl.distributed.tensor.grad_context import enable_grad

# Feature width of the "image" the policy emits and the reward consumes.
DIM = 16
HID = 32
POLICY_SEED = 0
REWARD_SEED = 1
DATA_SEED = 7


# ── Deterministic, device-independent builders (identical on workers & oracle) ──


def _fill_deterministic(module: nn.Module, seed: int) -> None:
    """Overwrite every parameter with CPU-RNG values so init is bit-identical
    regardless of device or default RNG state."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(generator=g))


def build_policy(seed: int = POLICY_SEED) -> nn.Module:
    m = nn.Sequential(nn.Linear(DIM, HID), nn.SiLU(), nn.Linear(HID, DIM))
    _fill_deterministic(m, seed)
    return m


def build_reward(seed: int = REWARD_SEED) -> nn.Module:
    m = nn.Linear(DIM, 1)
    _fill_deterministic(m, seed)
    for p in m.parameters():
        p.requires_grad_(False)  # frozen reward; gradient flows through the IMAGE
    return m


def full_input(total_n: int, seed: int = DATA_SEED) -> torch.Tensor:
    """Full input batch, generated on CPU so every worker's contiguous slice
    matches the oracle's full batch row-for-row."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(total_n, DIM, generator=g)


# ── Roles ──────────────────────────────────────────────────────────────────────


class PolicyRole(Remote):
    def initialize(self, seed: int = POLICY_SEED, use_fsdp: bool = False) -> None:
        torch.cuda.set_device(self.device)
        self.use_fsdp = use_fsdp
        model = build_policy(seed).to(self.device)
        if use_fsdp:
            import torch.distributed as dist
            from torch.distributed.fsdp import fully_shard

            if not dist.is_initialized():
                # dist_env (MASTER_ADDR/PORT/RANK/WORLD_SIZE for THIS role group)
                # was pushed into os.environ by Remote.setup().
                dist.init_process_group(backend="nccl")
            fully_shard(model)  # FSDP2; default PG = this policy group
        self.model = model

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def sample(self, total_n: int, seed: int):
        """Emit this dp-shard's images. Args are scalars (broadcast); each worker
        slices its own contiguous rows by dp_rank so the merged output is the
        full, in-order batch."""
        dp_rank = self.rank_info.dp_rank
        dp_size = self.rank_info.dp_size
        shard = total_n // dp_size
        x = full_input(total_n, seed)[dp_rank * shard : (dp_rank + 1) * shard].to(self.device)
        return self.model(x)  # [shard, DIM], carries grad_fn under grad_mode

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def loss_backward(self, reward):
        """The ``-reward.mean()`` seed node. Runs a LOCAL backward to populate
        reward.grad; the empty return makes this an always-run backward node so
        GradContext chains reward.grad up through score → sample → weights."""
        loss = -(reward.to(self.device)).mean()
        loss.backward()
        return None

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.ALL)
    def grad_vector(self):
        """Per-worker flattened param grad as plain floats (no transport).
        FSDP grads are DTensors → reconstruct the full tensor."""
        out = []
        for p in self.model.parameters():
            g = p.grad
            if g is None:
                out.append(None)
                continue
            if hasattr(g, "full_tensor"):  # DTensor (FSDP2 sharded grad)
                g = g.full_tensor()
            out.append(g.detach().float().flatten().cpu().tolist())
        return out


class RewardRole(Remote):
    def initialize(self, seed: int = REWARD_SEED) -> None:
        torch.cuda.set_device(self.device)
        self.model = build_reward(seed).to(self.device)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score(self, img):
        img = img.to(self.device)
        return self.model(img).squeeze(-1)  # [shard] per-row reward, grad via img


# ── Oracle (single process, plain autograd) ─────────────────────────────────────


def oracle_grad(total_n: int) -> list:
    """g_ref = d(-reward(policy(x)).mean()) / d(policy params), flattened."""
    policy = build_policy()
    reward = build_reward()
    x = full_input(total_n)
    loss = -(reward(policy(x)).squeeze(-1)).mean()
    loss.backward()
    return [p.grad.detach().float().flatten().tolist() for p in policy.parameters()]


# ── Compare ──────────────────────────────────────────────────────────────────


def _flatten(list_of_lists) -> torch.Tensor:
    flat = []
    for v in list_of_lists:
        flat.extend(v)
    return torch.tensor(flat, dtype=torch.float64)


def run(policy_gpus: int, reward_gpus: int, total_n: int, use_fsdp: bool, tol: float) -> bool:
    tag = f"P={policy_gpus} Q={reward_gpus} n={total_n} fsdp={int(use_fsdp)}"
    print(f"\n===== CONFIG {tag} =====", flush=True)
    if total_n % policy_gpus or total_n % reward_gpus:
        print(f"SKIP: n={total_n} not divisible by P={policy_gpus} and Q={reward_gpus}", flush=True)
        return False

    ray.init(ignore_reinit_error=True)
    print("cluster resources:", ray.cluster_resources(), flush=True)
    pool = DevicePool(num_devices=8, devices_per_node=8)
    pool.setup()
    try:
        policy = pool.create_remote(PolicyRole, device_ids=list(range(policy_gpus)))
        reward = pool.create_remote(
            RewardRole, device_ids=list(range(policy_gpus, policy_gpus + reward_gpus))
        )
        policy.initialize(seed=POLICY_SEED, use_fsdp=use_fsdp)
        reward.initialize(seed=REWARD_SEED)
        print("roles initialized; running enable_grad() forward+backward...", flush=True)

        with enable_grad():
            img = policy.sample(total_n=total_n, seed=DATA_SEED)
            rew = reward.score(img)
            policy.loss_backward(rew)

        per_worker = policy.grad_vector()  # list length = policy_gpus
        print(f"got grads from {len(per_worker)} policy worker(s)", flush=True)

        # Any worker with all-None grads => gradient never reached the weights.
        for i, gw in enumerate(per_worker):
            if any(g is None for g in gw):
                print(f"FAIL: policy worker {i} has a None param.grad (graph disconnected)", flush=True)
                return False

        worker_tensors = [_flatten(gw) for gw in per_worker]
        if use_fsdp:
            # FSDP reduce-scatter already averaged across dp ranks; full_tensor on
            # any worker is the full averaged grad. Cross-check workers agree.
            g_dist = worker_tensors[0]
            spread = max((t - g_dist).abs().max().item() for t in worker_tensors) if len(worker_tensors) > 1 else 0.0
            print(f"FSDP cross-worker full_tensor spread: {spread:.3e}", flush=True)
        else:
            # Replicated: mean of per-worker shard grads == full-batch grad.
            g_dist = torch.stack(worker_tensors).mean(dim=0)

        g_ref = _flatten(oracle_grad(total_n))
        if g_dist.numel() != g_ref.numel():
            print(f"FAIL: grad size mismatch dist={g_dist.numel()} ref={g_ref.numel()}", flush=True)
            return False

        max_abs = (g_dist - g_ref).abs().max().item()
        denom = g_ref.abs().max().item() or 1.0
        max_rel = max_abs / denom
        ref_norm = g_ref.norm().item()
        print(f"oracle grad norm={ref_norm:.4f}  max_abs_diff={max_abs:.3e}  max_rel_diff={max_rel:.3e}", flush=True)

        if not torch.isfinite(g_dist).all():
            print("FAIL: distributed grad has non-finite values", flush=True)
            return False
        if ref_norm < 1e-8:
            print("FAIL: oracle grad ~0 (degenerate probe; pick a different seed)", flush=True)
            return False
        ok = max_rel < tol
        print(f"{'PASS' if ok else 'FAIL'}: {tag}  (tol={tol:.1e})", flush=True)
        return ok
    finally:
        pool.shutdown()
        ray.shutdown()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-gpus", type=int, default=4)
    ap.add_argument("--reward-gpus", type=int, default=1)
    ap.add_argument("--n", type=int, default=64, help="total batch size")
    ap.add_argument("--fsdp", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-3)
    args = ap.parse_args()
    if args.policy_gpus + args.reward_gpus > 8:
        print("policy-gpus + reward-gpus must be <= 8", flush=True)
        return 1
    try:
        ok = run(args.policy_gpus, args.reward_gpus, args.n, bool(args.fsdp), args.tol)
    except Exception:
        print("ERROR: probe raised:\n" + traceback.format_exc(), flush=True)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
