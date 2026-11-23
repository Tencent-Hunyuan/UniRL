---
name: hot-path-and-backend-changes
description: Engineering standards for operators, training backends, rollout engines, SDE/replay, weight sync, and hot-path sync discipline across UniRL. Use when adding or modifying custom operators/kernels, training backends (FSDP2/VeOmni), rollout adapters or patches (vLLM-Omni/SGLang/FastVideo/Trainside), hot-path loops, or weight/tensor transport.
---

# Hot-Path and Backend Changes

Project-wide rules for where code lives, how components compose, and what is allowed on training and inference hot paths. Read [`unirl/README.md`](../../../../unirl/README.md) first for layer directions.

## 1. Constrained Composability & Seam Placement

Components in the training loop (`rollout/`, `reward/`, `algorithms/`, `train/`) wire via Hydra `_target_` + config validation in `unirl/config/`, not cross-sibling imports.

- **Composability is bounded.** Respect dependency-extra exclusivity (`sglang` and `vllm` extras in `pyproject.toml` are mutually exclusive) and engine/backend compatibility contracts (`ComposedRolloutEngine` is specific to PE chaining; `DifferentiableReward` carries autograd graphs; `VeOmniBackend` requires meta-init bundles). Validate invalid pairs early in `unirl/config/` or component constructors.
- **Share mechanisms; keep policy local.** Mechanisms (`distributed/`, `sde/`, `types/`, `config/`, and genuinely cross-owner leaves in `utils/`) are shared. Keep model, algorithm, reward, and backend policy local until a **second independent consumer** justifies promoting a narrow helper.

| Change | Canonical Owner |
| --- | --- |
| FSDP2 sharding, offload, checkpointing, structural injection | `unirl/train/backend/fsdp/` (`FSDPBackend`) |
| Ulysses SP, EP grouped-GEMM / `"fused_triton"` MoE, expert layout | `unirl/train/backend/veomni/` (`VeOmniBackend`, `ep/experts.py` keyed by `config.model_type`), under `BaseFSDP2Backend` (`base_backend.py`) |
| Rollout adapter / wire format translation | `unirl/rollout/engine/<engine>/adapters/` behind `BaseRolloutEngine` (`base.py`) |
| Upstream engine runtime bugfix / monkeypatch | Owning engine's patch package (`vllm_omni/patches/`, `sglang_diffusion/_patches/`, `fastvideo/_patches/`) with a `DELETE-WHEN` row |
| Model-specific forward/replay kernels, packing, `x_T`/σ overrides | `unirl/models/<family>/` (`compute_mu` overrides in `unirl/sde/`) |
| SDE step kernels, σ schedule, deterministic `x_T` noise | `unirl/sde/` (`SDEStrategy`, `FlowMatchSchedulePolicy`) |
| Trainer → rollout weight transfer | `unirl/distributed/weight_sync/full/{ipc,nccl,tensor,checkpoint}` or `lora/{local,remote}` |
| Rollout → trainer tensor handles & transport | `unirl/distributed/tensor/` (`TensorRef`, `TensorHandle`, `TensorSpan`, `hydrate`) |

**Open coupling debt (do not copy or extend):**
`models/{hunyuan_image3,qwen3_moe,qwen3_5}` importing `train/backend/veomni`, `models/cosmos3` subclassing `train/sft`, `algorithms/` flipping `train/lora`, `train/backend/sharded_load.py` importing VeOmni EP Qwen helpers, `utils/peft_merge.py` unpacking model internals, `distributed/weight_sync/full/tensor.py` adapter module-name sniffing, and duplicated SDE math in `vllm_omni/pipelines/_shared/flow_match_sde_scheduler.py`.

## 2. Host-Device Sync Discipline (Training AND Inference Hot Paths)

Apply this classification to both **training** (`train/`, `algorithms/`, `models/`, `sde/`) and **rollout/reward** (`rollout/engine/`, `reward/`) hot loops (per-step, per-microbatch, per-sample, per-layer, decode/denoise loops):

- **Class A — Banned on hot paths:**
  - `.item()`, `.cpu()`, `.tolist()`, `.numpy()`, or `torch.cuda.synchronize()` inside per-sample, per-layer, per-microbatch, or per-denoise/decode step loops (including scalar metric extraction before accumulation ends).
  - Implicit host sync via GPU tensor truthiness or string formatting (`if tensor:`, `bool(t)`, `int(t)`, `float(t)`, `f"{t}"`).
  - CUDA `torch.nonzero()` or dynamic-shape boolean indexing `x[mask]` in hot GPU paths.
  - Materializing `sigmas` to Python floats ahead of the denoising loop when arithmetic stays in PyTorch (keep `fp32` GPU tensors to avoid caller-dtype / `float64` drift and per-step host-to-device transfers).
- **Class B — Budgeted (`<= 1` batched sync per step, microbatch, or termination boundary):**
  - AR decode EOS / stop-condition check (`<= 1` reduced flag or EOS check per decode step/batch) and lockstep collective termination (`all_reduce(MIN)` + single `.item()`).
  - VeOmni EP `split_sizes` (`.cpu()`/`.tolist()` once before dispatched grouped GEMM) and varlen FlashAttention `max_seqlen`.
  - Non-finite `grad_norm` step-skip check once per optimizer step (`optimizer_step`).
  - Step metrics: accumulate on GPU (`torch.stack`) and do **one** batched `.tolist()`/`.cpu()` at the end of the optimizer step.
- **Class C — Allowed lifecycle & planning:**
  - `TokenBudgetPlanner`, `sleep` / `wake_up`, `onload` / `offload`, weight sync materialization + rollout `quiesce`/ACK, DCP/torch checkpointing, startup validation, and `regen_initial_noise` CPU `fp32` generation (required for cross-engine `x_T` bit-identity).

**Multi-stream & kernel hygiene:**
- Prefer explicit `torch.cuda.Event` (`event.record(producer)` + `consumer.wait_event(event)`) plus holding live tensor references across streams over blanket `record_stream()`.
- Keep custom `torch.autograd.Function` cohesive at one operator boundary; avoid redundant `.contiguous()` / `.clone()` copies or cross-rank conditional collectives (`P0-RANK-DIVERGENCE`).

## 3. Verification Tiers (Report Honestly in `## Test Plan`)

- **Tier 0 (Always):** `SKIP=no-commit-to-branch pre-commit run --all-files --show-diff-on-failure` plus `python -m unirl.<train_entry> --config-name=<domain>/<recipe> --cfg job --resolve` on every touched recipe.
- **Tier 1 (Parity & Sync Evidence):**
  - Capture a profiler trace in the process executing the changed path (`UNIRL_PROFILE=one-update` or `train` for training compute via `unirl/utils/profiling.py`; engine-side profiling for rollout processes) to confirm no unexpected sync bubbles or throughput regressions.
  - Use `torch.cuda.set_sync_debug_mode("error")` only as a **scoped local diagnostic** around non-collective Python hot paths (it does not observe NCCL internals, C++ extensions, allocator retries, or rollout engine subprocesses).
  - For deterministic kernels, weight-load/export, and router indices verify exact parity (`max_diff == 0.0`); for BF16/distributed reductions verify `atol/rtol` parity against the baseline (with `use_reentrant=False` when activation checkpointing is touched).
- **Tier 2 (Single-GPU Multi-Rank Gloo / Single-Rank CUDA):** Multi-rank `gloo` simulation validates collective ordering, rank lockstep, and FSDP wrap/sharding shapes (alongside single-rank CUDA checks); **never** claim it validates real multi-GPU NCCL, NVLink/P2P, CUDA-IPC (`gpu_store` / `IPCWeightSync`), `DevicePool`, or multi-TP rollout engines. State this limit explicitly in `## Test Plan`.
- **Tier 3 (Real Multi-GPU / Multi-Node NCCL):** Required to validate multi-GPU NCCL collectives, IPC, cross-node RDMA weight sync, SP/EP scaling, or real end-to-end throughput; if unavailable in the current environment, write `Not run; reason: ...`.
