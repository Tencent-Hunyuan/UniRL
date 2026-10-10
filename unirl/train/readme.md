# Train Stack

> **Where it fits:** the optimizer half of the *train* step —
> rollout → reward → advantage → **train** → sync. In: a track with advantages
> (plus the algorithm's gradients). Out: updated weights (synced back to rollout in
> dedicated modes). Full map: [`../README.md`](../README.md).

## What it is

`unirl/train` is the optimizer half of the UniRL training loop. It owns the
trainable model's parameters, optimizer, scheduler, EMA shadow, and structural
injection (LoRA / DiffusionNFT / mirror), and it sequences loss → backward → optimizer step
for one rollout track at a time. The loss math itself belongs to the algorithms
module; this module never computes a loss.

## Why it exists

The split exists because two correctness invariants live on the train side, not in
the loss math, and both silently corrupt training if they slip. Per-block
`fully_shard` alone would leave the root params (embed / final norm / lm_head) as
plain replicated tensors FSDP never reduces — their replicas would drift apart
across ranks and the grad-norm would be wrong — so `fsdp_wrap` claims them into a
root `fully_shard` group by default (`root_wrap`) and fails fast when a multi-rank
run leaves a trainable param outside every group. The uniform wrap also keeps the
optimizer an all-`DTensor` param bag (a mixed `Tensor` + `DTensor` bag forces
`foreach=False` or AdamW's fused kernel raises). Centralizing wrap + step here lets
every algorithm inherit these invariants for free; the loss module legitimately
knows nothing about DTensor sharding or wrap topology.

## How it works

- **`FSDPBackend`** (`backend/fsdp/backend.py`) holds the FSDP2-wrapped module
  (`bundle.<trainable_attr>`, default `transformer`), the optimizer, scheduler, and
  EMA (when configured). Structural injection (`inject_lora` / `inject_nft` /
  `inject_mirror`) happens once at construction, *before* `fsdp_wrap`.
  `optimizer_step` is the single chokepoint: clip → step → schedule → EMA, and it
  **skips the whole step on a non-finite grad norm** (stepping would scale every
  parameter by the bad norm and poison the next rollout).
- **`TrainStack`** (`stack/base.py`) takes one backend + one `StageAlgorithm` and runs
  `train_track`: move the segment onto device → `prepare_segment_anchors` (freeze
  π_old once) → `num_updates_per_batch` optimizer steps over disjoint mini-batches,
  each a micro-batch loop of `compute_loss_and_backward`. The mini/micro slicing comes
  from one source — the injected `micro_planner` (`stack/planner/`; `CountPlanner` by
  default, `TokenBudgetPlanner` for token packing) — shared with
  `prepare_segment_anchors` (`stack/anchor.py`, also used by `UnifiedModelTrainStack`),
  so when an algorithm replays its anchor, it's recomputed at the *exact* geometry
  training uses, which is what pins the on-policy PPO ratio to 1 under bf16's
  batch-shape sensitivity. `AgenticTrainer` uses the same interface after
  concatenating every successful trajectory's generated assistant turns.
- **`UnifiedModelTrainStack`** (`unified_model_stack.py`) drives two algorithms
  (`ar` + `image`) backward-accumulating into one shared optimizer step on one
  shared backbone (HunyuanImage3).

**Extending it:** a new structural injection mode is an `inject_<mode>` alongside
`inject_lora` (`lora.py`) and `inject_nft` / `inject_mirror` (`ema.py`), called
before `fsdp_wrap`, plus a config in `configs.py`; a new optimizer or LR schedule
is a branch in `optim.py` plus fields on `OptimizerConfig` / `LrSchedulerConfig`
in `backend/base.py`; a multi-update-capable algorithm sets
`supports_multi_update = True`, declares `anchor_fields`, and exposes
`recomputes_anchor` when its anchor must follow the planned micro geometry (see
`../algorithms/README.md`).

## Choosing HSDP (hybrid sharding)

`fsdp_mode: hybrid` on `FSDPBackend` builds a
`(world / hsdp_shard_size, hsdp_shard_size)` replicate × shard mesh: parameters
shard within each `hsdp_shard_size` group, and the groups hold replicated copies
whose gradients are all-reduced. Choose it when a shard group that fits one node
over NVLink is a cheaper all-gather domain than the whole world;
`hsdp_shard_size: devices_per_node` is the usual layout (world 16, shard 8 → a
`(2, 8)` mesh). The trade is memory: per-rank weight + optimizer memory is
`world_size / hsdp_shard_size` times the `full`-mode footprint.
`resolve_fsdp_mesh_shape` ([`configs.py`](configs.py)) fails fast unless the
shard size is `>= 2`, the world is strictly larger (a one-group world is
`full`), and the world divides by the shard size.

Checked-in recipes:
[`diffusion/minimax_h3/minimax_h3_t2va_trainside_hsdp_2x8_validation`](../../examples/diffusion/minimax_h3/minimax_h3_t2va_trainside_hsdp_2x8_validation.yaml)
— a 2×8 **validation/smoke** recipe, not a production default — and
[`ar/bagel_grpo_arxivqa_mc_2x8_lora`](../../examples/ar/bagel_grpo_arxivqa_mc_2x8_lora.yaml).

## Choosing the VeOmni backend

`VeOmniBackend` ([`backend/veomni/backend.py`](backend/veomni/backend.py)) is a
per-recipe alternative to `FSDPBackend`, selected by
`backend._target_: unirl.train.backend.veomni.backend.VeOmniBackend`. It subclasses the same base, so the checkpoint/save/resume formats, the
optimizer, and the `TrainStack` contract are unchanged. Install with the
`veomni` extra (`pip install -e '.[veomni]'`; extras table in
[INSTALL.md](../../INSTALL.md)).

What it adds: Ulysses sequence parallelism (`fsdp_cfg.sp_size` — must divide
`world_size`, and the model's attention heads) and MoE expert parallelism
(`fsdp_cfg.ep_size` — must divide `world_size` and the expert count; each rank
owns `num_experts / ep_size` experts, tokens routed all-to-all into fused
grouped-GEMM). The VeOmni bundles build the transformer on the **meta device**
— VeOmni's parallelize asserts meta init, and the backend loads real weights
after sharding — so a bundle without meta-init support cannot run on this
backend.

Climb the ladder from a parity twin before the parallel variants; it lists
every checked-in VeOmni recipe:

| Step | Recipe | Exercises |
| --- | --- | --- |
| 1 | [`diffusion/sd3_trainside_veomni`](../../examples/diffusion/sd3_trainside_veomni.yaml) | parity twin of `sd3/sd3_trainside` on VeOmniBackend (`sp_size: 1`) — backend parity |
| 2 | [`ar/qwen3_grpo_4b_veomni_sp_sglang`](../../examples/ar/qwen3_grpo_4b_veomni_sp_sglang.yaml), [`ar/qwen3_drpo_4b_veomni_sp_sglang`](../../examples/ar/qwen3_drpo_4b_veomni_sp_sglang.yaml) | Ulysses SP (`sp_size: 2`) + SGLang, GRPO and DRPO |
| 3 | [`ar/qwen3_moe_grpo_30b_a3b_veomni_ep_sglang`](../../examples/ar/qwen3_moe_grpo_30b_a3b_veomni_ep_sglang.yaml), [`ar/qwen3_5_moe_grpo_35b_a3b_base_dapo_sglang`](../../examples/ar/qwen3_5_moe_grpo_35b_a3b_base_dapo_sglang.yaml), [`ar/qwen3_5_moe_grpo_35b_a3b_geo3k_mc_sglang`](../../examples/ar/qwen3_5_moe_grpo_35b_a3b_geo3k_mc_sglang.yaml) | expert parallelism (`ep_size: 8`) on 30B-A3B / 35B-A3B MoE |
| 4 | [`diffusion/qwen_image_trainside_veomni`](../../examples/diffusion/qwen_image_trainside_veomni.yaml), [`unified_model/hi3_vllmomni_veomni_ep`](../../examples/unified_model/hi3_vllmomni_veomni_ep.yaml) | Qwen-Image parity twin; HI3 unified model with EP |

v1 restrictions, all validated fail-fast in `_validate_fsdp_cfg`:
`fsdp_mode='full'` only (HSDP stays on `FSDPBackend`), `mixed_precision` must
stay enabled, and `cpu_offload` / `copy_engine_all_gather` are unsupported.
Separately, with `ep_size > 1`, full **DCP** checkpoints are rejected — the
expert split is not encoded in the DTensor placements. This is checked at save
and load time, not at startup, so a run with `checkpoint_format: dcp` fails at
its first checkpoint; keep the default `checkpoint_format: torch` (EP-aware) or
use LoRA adapter mode. Formats and export live in
[`trainer/README.md · Checkpointing`](../trainer/README.md#checkpointing).

## Gotchas

- **`num_updates_per_batch > 1` needs `supports_multi_update` *and* must evenly
  divide the per-worker batch** — otherwise the ctor or `_build_mini_batch_slices`
  raises (a ragged mini-batch would silently drop samples and desync grad-accum
  across DP ranks).
- **Multi-update membership is contiguous by default.** Set `shuffle_updates: true`
  on the stack to shuffle the full per-worker batch before it is partitioned into
  optimizer updates; `shuffle_seed` (default 0) is combined with the rollout id so
  the per-rollout ordering is reproducible across checkpoint resume. It is a no-op
  at `num_updates_per_batch: 1`, requires the default `CountPlanner`, and gathers
  each micro-batch lazily, so it adds no full-track copy.
- **`optimizer_step` silently *skips* (does not crash) on a non-finite grad norm**
  and zeroes grads — a flat loss curve with a logged warning means grads went
  non-finite.
- **Checkpointing preserves never-updated AdamW params** — export writes a param
  without state (or a never-stepped AdamW) as the step-0 zero state its first
  update would create, so torch and DCP checkpoints stay dense; load drops step-0
  entries back to lazy init. Torch-format checkpoints that omitted them still load.
- **`master_dtype` defaults to `None`, so the optimizer master follows `param_dtype`** —
  a bf16-loaded base then keeps a bf16 LoRA master and the ~1e-6 AdamW steps round
  away (the policy drifts into a degenerate reward-hack). An fp32-loaded model gets an
  fp32 master for free; a bf16 load needs `master_dtype: fp32` set explicitly. The
  ctor never warns.
- **The EP fused-expert layout registry is keyed by `config.model_type`, never by
  parameter-name suffix** (`backend/veomni/ep/experts.py`). HunyuanImage3's fused
  params end in `.experts.down_proj` too, but its `gate_and_up` halves are stored
  swapped relative to Qwen3-MoE's `gate_up`, so a suffix match would silently hand
  one model's packing semantics to another — today that only fails closed because
  HI3's *other* suffix happens not to match. An `ExpertExportLayout` pairs a model
  module's `is_fused_expert_param(name)` and
  `iter_hf_expert_tensors(name, stacked)` operations; register it in
  `_EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE` to enable full-weight sync for that family.
  Consumers outside `train/` take the transform from
  `backend.expert_weight_export_transform()` — `distributed/weight_sync/` must not
  import the layout (the core-dependency guard rejects it).
- **Advantages are not computed here** — `train` raises if
  `part.advantages is None`; the trainer must call `compute_advantages` on the
  full shard first.
- **Selective offload supports direct GPU-streaming weight sync.** The trainer
  keeps FSDP parameter shards on device while vLLM receives weights, clears
  consumed gradients and offloads optimizer state before publication, then
  offloads the model after publication commits. Calling `offload()` without
  arguments retains the legacy full-state behavior.
- **`fsdp_wrap` wraps *nothing* when no block class is discovered** — the warning
  says "root-only wrap" but `_enumerate_block_instances` returns `()`, so the
  shard/cast loops are no-ops and the model trains **unsharded and un-cast**. Pass
  `block_class_names` explicitly in the recipe.
- **`fsdp_wrap` shards the leftover params (embed / final norm / lm_head) into a
  root `fully_shard` group by default** (`root_wrap`, on) — the root group never
  reshards after forward. Set `root_wrap: false` for models whose stages call
  submodules of the wrapped object directly (bagel) or that wrap frozen mixed-dtype
  siblings (hunyuan_image3); a multi-rank run with a trainable param left outside
  every group then fails fast.
- **`defer_grad_sync` needs the last micro-batch of an optimizer step to run a
  backward** — the deferred reduce-scatter only fires inside it. If that micro
  skips backward (an all-empty micro) while earlier ones ran, `TrainStack.train`
  raises instead of silently stepping on never-synced grads (which would also
  leak the stale accumulation into the next step's reduce-scatter).
- **`fsdp_mode: no_shard` trades memory for the all-gather** — a `(world, 1)` mesh
  leaves the full model on every rank, so no parameter bytes cross ranks and only
  gradients are all-reduced (DDP). It pays off where the re-gathered bytes dwarf the
  gradient (LoRA on a frozen base re-gathers the whole backbone *every* micro-batch)
  and the model still fits unsharded — per-rank memory becomes the whole model +
  grads + optimizer state. `reshard_after_forward` is **not** a no-op here: `false`
  keeps every block's unsharded compute copy resident, which is a second full copy of
  the model whenever `param_dtype` upcasts (fp32 compute over a bf16 checkpoint), so
  leave it `true` unless that copy is cheap. `defer_grad_sync: true` then gives one
  all-reduce per optimizer step. VeOmni only supports `full`.
- **`copy_engine_all_gather: true` takes the FSDP all-gather off the SMs** — FSDPBackend
  creates the default NCCL group with the zero-CTA policy and every `fully_shard` group
  allocates its all-gather buffer from NCCL symmetric memory, so the gather runs on the
  copy engines (`cudaMemcpyBatchAsync`) instead of an `ncclDevKernel_AllGather` kernel.
  Needs PyTorch >= 2.13, NCCL >= 2.28, and a shard group that stays on one node over
  NVLink (`full` on a single node, or `hybrid` with `hsdp_shard_size:
  devices_per_node`); `no_shard` and VeOmni reject it. FSDPBackend must be what brings
  up `torch.distributed` (it binds WORLD to the rank's CUDA device so DeviceMesh splits
  the shard group from it and the policy is inherited), and
  `NCCL_CTA_POLICY` must stay unset or `2`. WORLD keeps the usual `cpu:gloo,cuda:nccl`
  pair, so in `hybrid` mode torch logs one `ProcessGroupGloo::split ... Falling back to
  default options` warning per process while splitting the gloo half; it is expected.
- **`lora_cfg.frozen_adapters` (OPD teachers) have real weights only after
  `apply_deferred_ops`** — the adapter is injected pre-wrap so meta-init bundles work, but
  its weights load after materialization; reading a teacher earlier sees a null or
  uninitialized adapter. Teachers must be plain LoRA deltas: unconverted base-rewriting
  inits (pissa, olora, ...), `modules_to_save`, `layer_replication`,
  `trainable_token_indices`, DoRA, and `bias != "none"` are rejected at startup.
- **Adapter checkpoints exclude frozen teachers, and resume requires the same teachers** —
  teachers reload from their paths and the checkpoint pins each by a content sha256, so a
  different teacher set or different weights raises on `load`. A checkpoint trained without
  teachers resumes into any teacher set.
- **AdamW takes the single-tensor path only for a *mixed* `Tensor`/`DTensor` param bag** —
  `build_optimizer` sets `foreach=False` only for the params it actually hands to AdamW, and
  only when those mix FSDP-wrapped `DTensor`s with plain `Tensor`s; every all-`DTensor`
  configuration keeps torch's default (multi-tensor on CUDA, single-tensor on CPU) instead of
  the per-parameter loop. In the FSDP backend that mix needs `training.fsdp.root_wrap=false` on
  a **single** rank: the stray-param guard in `fsdp_wrap` runs at `world_size > 1` only, so on
  one rank an unfrozen leftover (full FT of the fp32-pinned `proj_in` / `time_embedder` that
  the LoRA recipes ship frozen) puts a plain `Tensor` into a `DTensor` list and the
  multi-tensor kernels reject it with `RuntimeError: aten._foreach_lerp_.Scalar got mixed
  torch.Tensor and DTensor`. The VeOmni backend parallelizes through `parallelize_model_fsdp2`
  instead, which carries no equivalent guard — its param mix is not assumed here. Params under
  `cpu_offload` report `device.type == "cpu"`, so torch's own device gate keeps them on the
  single-tensor path, and `clip_grad_norm` catches the same mixed bag by message in
  `backend/fsdp/state.py`.

## Profiling → Perfetto

Opt-in `torch.profiler` (`unirl/utils/profiling.py`): one env var writes a gzipped Perfetto
trace to `outputs/profiler/` on rank 0 (no-op otherwise). Training compute only (not rollout).

Both capture **one snapshot of a single train step** (after a short warmup, then off — one
trace, not one per step; training keeps running). They differ in *which slice* they record:

- **`one-update`** — one optimizer update (forward + backward + optimizer) with its cross-GPU
  comm. Skips the anchor forward; small; **for compute/comm overlap**.
- **`train`** — the whole step: anchor forward (recompute old-policy log-probs; for diffusion
  it replays the denoising trajectory, so it's large) + all updates. **For step-time breakdown.**

> The unified-model stack (HI3, AR+image) only supports `train` — it fuses each step into a
> single update, so there is no per-update boundary for `one-update` to wrap (a warning is
> logged and no trace is produced).

```bash
UNIRL_PROFILE=one-update  python -m unirl.train_diffusion --config-name=<recipe> ...
UNIRL_PROFILE=train       python -m unirl.train_diffusion --config-name=<recipe> ...
```

Download the `.gz` to your local machine, then <https://ui.perfetto.dev/> → *Open trace file*.

### Configurable env vars (defaults are fine)

| var | what it does | default |
|-----|--------------|---------|
| `UNIRL_PROFILE` | `one-update` / `train` (unset = off) | off |
| `UNIRL_PROFILE_DIR` | where the trace is written (auto-created) | `outputs/profiler` |
| `UNIRL_PROFILE_RANKS` | which ranks profile: `0`, `all`, or a list like `0,8` | `0` |
| `UNIRL_PROFILE_CUDA` | record GPU kernels; `0` = CPU-only trace | `1` |
| `UNIRL_PROFILE_WARMUP` | skip the first few iterations before recording (avoids the one-time first-iter compile / alloc) — one-update: skip N updates; train: schedule warmup | `2` / `1` |
| `UNIRL_PROFILE_MEMORY` | also record CUDA memory alloc/free (memory-over-time; bigger trace) | off |
