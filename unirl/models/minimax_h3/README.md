# MiniMax-H3

> **Where it fits:** trainside text-to-video/audio rollout and replay over one shared bundle.
> In: prompts and sampling parameters. Out: video, audio, and diffusion trajectories.
> Full map: [`../README.md`](../README.md).

## Gotchas

- **`frozen_adaln_cache: true` caches complete timestep calls, not individual timesteps.**
  The default `false` installs no hooks or child groups. Opt-in installs
  `FrozenAdaLNCache` before backend construction; after LoRA injection,
  `fsdp_wrap` calls the optional `_fsdp_shard_frozen_children` protocol (frozen
  children plus a post-wrap callback) and shards each frozen `adaln_proj`
  before checkpoint wrappers and parent blocks, using
  the same mixed-precision/offload/reshard policy. Child groups disable backward
  unsharding because their outputs have no gradient dependency. AC still wraps
  only the original blocks. The callback re-prepends key capture after FSDP
  installs its own prepended device/cast hook, preserving the original input
  dtype/device/bytes even with root wrapping. Only UniRL FSDP2 is supported;
  explicit forward prefetch and torch.compile are rejected. Keep the recipe's
  `cast_forward_inputs=false` to preserve fp32 SiLU inputs; root_wrap=true still
  casts the outer timestep MLP as it does without caching, so it is not an
  equivalent replacement for the recipe's root_wrap=false.
- **Cache hits skip the child module's entire call, not merely its linear.**
  The vendor changes only pass a per-call table object through the transformer,
  block and checkpoint paths; bypassing inside the child's forward would be
  too late to skip FSDP's pre-hook. Misses execute the original
  `silu(temb) -> cast -> linear` on the entire original GEMM shape, then reshard
  normally. AC captures the original call's table object, not mutable global
  timestep state, and hits even after other forwards or LRU eviction. Returned
  tables are clones, so downstream writes cannot corrupt stored values.
- **All ranks must enter cached model calls in lockstep.**
  Keys contain the complete original timestep bytes, dtype, device and shape,
  plus autocast mode/dtype. Reading a CUDA key adds one blocking device-to-host
  copy per model call, not per layer. A WORLD all-reduce of the local hit flag
  adds one scalar host read/synchronization; a miss on any rank makes all ranks
  compute their own full tables, preventing divergent collective sequences.
  There are two added host synchronization points per CUDA model call and none
  in AC recomputation. Locally cached entries can therefore be recomputed.
- **Only fully frozen modulation and timestep upstream modules may use this cache.**
  `time_proj`, `time_embedder` and every `adaln_proj` are checked after structural
  injection and before every model call; timesteps requiring gradients are
  rejected even on a cache hit. The current LoRA targets omit all
  three; full/mirror EMA registers only trainable parameters, and adapter EMA
  swaps only adapters, so neither changes these frozen weights. This contract
  must be rechecked if the vendor timestep path gains another dependency.
- **Frozen-weight changes require invalidation before the next forward.**
  Root `load_state_dict` and root `_apply` clear the cache; the latter includes
  dtype/device moves, materialization and FSDP offload/onload. The invalidation
  wrapper delegates to the current model class's `_apply`, not a saved pre-FSDP
  bound method, which would bypass FSDP's reshard/reset behavior. Call
  `transformer._h3_adaln_cache.clear()` on every rank before direct parameter
  writes, direct child moves, nonstandard checkpoint loading, numerical-policy
  changes (including TF32), or changes to timestep-module configuration. Do not change weights or precision between
  a checkpointed forward and its backward. H3 pipeline sleep/wake releases
  only auxiliary modules, leaves the shared transformer untouched and does not
  invalidate tables; a sleep that offloads the transformer does invalidate them.
- **Persistent table storage is capped at 32 LRU keys and 256 MiB per rank.**
  For U timesteps, L layers, width H and d bytes per output element, one key
  stores `U * L * 3 * 6 * H * d` tensor bytes, plus timestep bytes and Python
  metadata. For H=5376, L=50 and bf16, U=1/2 costs 9,676,800/19,353,600 bytes.
  A cold call temporarily holds its in-progress table; live checkpoint graphs
  also pin their call tables until backward, outside the persistent LRU budget.
  Oversized single entries are not published. Only successful complete model
  calls are published; an exception cannot publish a partial entry. Completed
  earlier steps remain valid after a later generate failure. Rank zero logs one
  cumulative hits/misses/keys/bytes summary at each generate exit; the next
  generate summary includes intervening policy, KL and backward hits.

- **`rollout_keep_unsharded_layers: N` retains only the first N transformer blocks during one generate.**
  The H3 config accepts integers in `[0, 50]`; the stage also checks the loaded
  layer count. The default `0` leaves `generate` unwrapped, without FSDP scans,
  flag changes, logging, or extra tensor/RNG operations. Opt-in requires no-grad
  generation and UniRL FSDP2 with `reshard_after_forward=true`, one FSDP group
  per selected block (excluding optional cached AdaLN child groups);
  activation-checkpoint wrappers in either order and
  `root_wrap=false` are supported. All ranks must use the same N.
  Each selected block gathers on first use, then reuses its compute parameters;
  `finally` reshards it and restores the flag before generate returns or raises.
  Nothing stays retained across VAE decode, another forward batch, sleep/wake,
  policy replay, reference KL, or backward. Eval uses the same scope.
- **Retained compute parameters consume additional device memory, not just their local shard size.**
  Rank 0's first successful enabled generate logs each selected block's actual
  unsharded parameter bytes and total GiB (including adapters/shadows, excluding
  cached AdaLN child groups, allocator padding, activations and communication buffers). The original shards
  remain resident. VAEs are already awake during denoising; size N against that
  combined peak, not the lower decode peak. Crossing decode to retain across
  batches would add these bytes to decode activations too, so is not supported.
  With L block gathers per step and S steps, each generate uses `S*(L-N)+N`
  gathers; K forward batches use `K*(S*(L-N)+N)`. With AdaLN caching, add L
  child gathers per globally cold timestep call; the remaining parent gathers
  carry fewer bytes, but their count does not decrease. `H3_DIAG_AG=1` counts actual
  launches, so retained blocks' no-op unshards do not inflate the count. Measure
  GPU headroom before increasing N; return to `0` after an OOM and restart the run.

- **H3 supports engine-grouped per-sample SDE requests.**
  `MiniMaxH3DiffusionStage.supports_order_independent_sde_noise` is true because
  `MiniMaxH3Pipeline.generate` passes sample ids as `denoise_seed_keys` to both
  stream generators. Regrouping therefore preserves each sample's step noise.
  The [FlashGRPO recipe](../../../examples/diffusion/minimax_h3/minimax_h3_t2va_flashgrpo.yaml)
  enables this path; its shared engine requirements live in
  [`trainer/README.md`](../../trainer/README.md#gotchas).
- **Sleep returns released auxiliary blocks to the CUDA driver.** With
  `rollout_resident=false`, `MiniMaxH3Pipeline.sleep` releases CUDA video/audio
  VAEs and any CUDA text encoder after decoding and inline `reward_stack` scoring,
  without retaining per-rank CPU weight copies. `vae_components_on_cpu` and
  `aux_components_on_cpu` independently control placement: CPU modules stay resident,
  so CPU VAEs do not prevent a CUDA text encoder from being released. Before the next
  rollout or eval, `wake_up` reloads released modules on their original devices;
  the shared trainable transformer stays untouched. Sleep then calls
  `torch.cuda.empty_cache()`, after the module references are dropped and whenever
  a module is recorded as released: freed blocks otherwise stay in PyTorch's
  caching allocator, where allocations made outside it (the cuBLAS handle, NCCL
  buffers) cannot use them.
- **H3 auxiliary checkpoints and construction config must stay unchanged between wakes.**
  `load_minimax_h3_vae` and `load_minimax_h3_text_encoder` in
  [`bundle.py`](bundle.py) serve both initial load and wake, preserving checkpoint
  paths, dtype, eval mode, and freezing; `from_bundle` must receive the bundle's
  construction config so reloading reproduces its weights.
- **H3's audio VAE has unregistered `weight_norm` tensors.** Moving only registered
  parameters/buffers to meta leaves those allocations alive; sleep drops every
  bundle, decode-stage, and text-embed-stage module reference, and wake rebinds
  all of them, including the text encoder's VAE lease.
- **Checkpoint overrides default only when `None`.** `vae_ckpt_path` and
  `text_encoder_ckpt_path` otherwise reach their loaders unchanged, including
  invalid empty paths.
- **A batched rollout shares one packed layout and one SDE step list.**
  `MiniMaxH3DiffusionStage.generate` accepts `[B, rows, channels]` video/audio
  latents with the same geometry, text length after trimming right-padding, and
  `sde_indices` for every row. Geometry and schedules are request-shared;
  mismatched latent shapes or text lengths are rejected before the forward.
  The existing transformer batch axis keeps attention documents independent;
  it does not concatenate samples into one sequence.
- **Flash-GRPO engine-mode training batches support multiple prompts per rank.**
  Set trainer `batch_size` to a positive multiple of `rollout dp_size`.
  With `rollout.forward_batch_size > 1`, trainside splits per-sample SDE requests
  at each prompt boundary, pins that prompt's `sde_indices`, then chunks its rows.
  Keep `stack.micro_batch_size=1` for training. Eval may have multiple
  prompts per rank; `samples_per_prompt % forward_batch_size == 0` keeps chunks
  within one prompt. See [`../../trainer/README.md`](../../trainer/README.md).
- **Batched forwards need not be bit-identical to serial forwards.**
  GEMM batch geometry can change rounding. B=1 keeps its original operations;
  each row's denoising noise remains keyed by sample ID, step and base seed,
  with a separate audio salt. Supply one seed key per row for order-independent
  noise. The engine retains each row's `[before, after, final]` video/audio
  slots and joint SDE log-probability.
- **Both VAEs decode one row at a time and return CPU media.**
  This keeps decode activations at B=1 even when denoising is batched. The audio
  VAE's batch axis is the two stereo channels of a single sample, not rollout
  rows. Decoded media retain input row order; CPU media storage grows with B.
