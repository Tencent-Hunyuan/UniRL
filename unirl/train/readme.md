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

- **`FSDPBackend`** (`backend/fsdp.py`) holds the FSDP2-wrapped module
  (`bundle.<trainable_attr>`, default `transformer`), the optimizer, scheduler, and
  EMA (when configured). Structural injection (`inject_lora` / `inject_nft` /
  `inject_mirror`) happens once at construction, *before* `fsdp_wrap`.
  `optimizer_step` is the single chokepoint: clip → step → schedule → EMA, and it
  **skips the whole step on a non-finite grad norm** (stepping would scale every
  parameter by the bad norm and poison the next rollout).
- **`TrainStack`** (`stack/base.py`) takes one backend + one `StageAlgorithm` and runs
  `train_track`: move the segment onto device → `prepare_segment` (freeze π_old
  once) → `num_updates_per_batch` optimizer steps over disjoint mini-batches, each a
  micro-batch loop of `compute_loss_and_backward`. The mini/micro slicing comes from
  one source, `_optimizer_step_slices`, shared with `prepare_segment` — so when an
  algorithm replays its anchor, it's recomputed at the *exact* geometry training
  uses, which is what pins the on-policy PPO ratio to 1 under bf16's batch-shape
  sensitivity.
- **`UnifiedModelTrainStack`** (`unified_model_stack.py`) drives two algorithms
  (`ar` + `image`) backward-accumulating into one shared optimizer step on one
  shared backbone (HunyuanImage3).

**Extending it:** a new structural injection mode is an `inject_<mode>` in
`inject.py` (called before `fsdp_wrap`) plus a config in `configs.py`; a new
optimizer or LR schedule is a branch in `factories.py` plus fields on
`OptimizerConfig`/`LrSchedulerConfig`; a multi-update-capable algorithm sets
`supports_multi_update = True` and declares `anchor_fields` (see
`../algorithms/README.md`).

## Gotchas

- **`num_updates_per_batch > 1` needs `supports_multi_update` *and* must evenly
  divide the per-worker batch** — otherwise the ctor or `_build_mini_batch_slices`
  raises (a ragged mini-batch would silently drop samples and desync grad-accum
  across DP ranks).
- **`optimizer_step` silently *skips* (does not crash) on a non-finite grad norm**
  and zeroes grads — a flat loss curve with a logged warning means grads went
  non-finite.
- **`master_dtype` defaults to `None`, so the optimizer master follows `param_dtype`** —
  a bf16-loaded base then keeps a bf16 LoRA master and the ~1e-6 AdamW steps round
  away (the policy drifts into a degenerate reward-hack). An fp32-loaded model gets an
  fp32 master for free; a bf16 load needs `master_dtype: fp32` set explicitly. The
  ctor never warns.
- **Advantages are not computed here** — `train` raises if
  `resp_track.advantages is None`; the trainer must call `compute_advantages` on the
  full shard first.
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
