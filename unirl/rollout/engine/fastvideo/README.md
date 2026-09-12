# FastVideo rollout engine

> **Where it fits:** the `fastvideo` backend of the *rollout* step — an
> in-process `VideoGenerator` driven one prompt at a time through
> `executor.execute_forward`. Full map: [`../README.md`](../README.md).

*Runs WAN 2.1 and WAN 2.2 A14B T2V rollouts on the pinned FastVideo RL fork and
re-routes its hard-coded deterministic Euler steps through UniRL's canonical
UniPC solver (`unirl/sde/unipc.py`), so rollouts follow the checkpoint's native
inference solver without integer-timestep loss.*

## The pin

The integration targets the **unmodified upstream**
**`hao-ai-lab/FastVideo@2095477eac7e289c7a7ab13acb367ca60687c304`** (PR #1222's
source snapshot), injected via `cfg.fastvideo_path` / `$FASTVIDEO_PATH` — there
is no pip pin. Upstream ships none of the RL surface this engine needs, so
UniRL owns it under `_patches/`: `contracts.py` adds the
`RLData.sde_step_indices` / `sde_type` fields, `denoising.py` adds the `eta` /
`sde_type` parameters on `sde_step_with_logprob` and the transition itself
(whose stock version uses a different diffusion law than UniRL's Dance/Flow
kernels), `conditions.py` restores the `rl_data` / trajectory / prompt
embeddings that upstream's `MultiprocExecutor.execute_forward` drops when it
rebuilds its response batch, and `weights.py` adds the full-weight update API
upstream does not expose at all. `hijack.py` installs them in that order and
`multiproc.py` re-installs them in every spawned worker.

`compat.py` fingerprints the surface at patch-install time (engine init and
every spawned worker), taking the **stock** signatures before any UniRL patch
rewrites them: `sde_step_with_logprob`'s stock parameter list plus the
`DenoisingStage.forward` source markers, `FlowUniPCMultistepScheduler.set_timesteps`,
`MultiprocExecutor.collective_rpc` / `execute_forward`, `Worker.execute_forward`,
`ModuleHookManager.get_from` / `get_forward_hook`,
`LayerwiseOffloadHook.mutate_params_scope`,
`fsdp_load.load_model_from_full_model_state_dict`, and
`PipelineComponentLoader.load_module`. Drift fails closed at init instead of
mid-rollout. Before editing a patch, read that exact commit (CLAUDE.md:
monkey-patch doctrine), then update the fingerprints together with the patch.

## How the canonical UniPC path works

- **σ SSOT.** The engine sends `FlowMatchSchedulePolicy`'s already-shifted
  canonical σ verbatim (no shift pre-image). The patched `set_timesteps`
  rejects any schedule transform on external sigmas, requires strictly
  decreasing σ (duplicate adjacent float32 values alias into one
  `index_for_timestep` slot, skipping one transition and double-stepping
  another with an `h=0` NaN), appends the terminal zero, and keeps **float32**
  model timesteps (`σ · WAN21DiffusionStep.TIMESTEP_SCALE`) — stock stores
  `int64`, which truncates conditioning (`833` vs `833.333…`, the drift #248
  tracked).
- **Solver SSOT.** `WAN21PipelineConfig.unipc_*` declares the deterministic
  solver (`WAN22PipelineConfig` inherits the same fields). At init the engine
  verifies it against the checkpoint's
  `scheduler/scheduler_config.json`, resolving either a local diffusers-layout
  directory or a Hugging Face model repo — fail closed on a mismatch, a
  missing/unreadable file, a non-UniPC `_class_name`, or a non-empty
  `disable_corrector` (no config knob) — and requires an injected SDE strategy
  whose `canonical_name` the plan supports. The spec travels as a `UniPCSpec`
  inside the per-request `FastVideoUniPCPlan`, carried through the fork's
  str-typed `RLData.sde_type` with `sde_step_indices=None` so every index
  reaches the patched helper. The worker validates the live scheduler contract
  (`flow_prediction`, `predict_x0`, no thresholding / nested `solver_p`),
  builds `UniPCStrategy` from the plan spec, and resets multistep history
  across SDE jumps. Checkpoint verification cannot live in the worker: the
  fork's `WanPipeline.initialize_pipeline` discards the checkpoint-loaded
  scheduler and rebuilds `FlowUniPCMultistepScheduler` from constructor
  defaults, so the live scheduler never reflects `scheduler_config.json`.
- **Dispatch.** Plan indices → the fork's Dance/Flow SDE helper (real
  log-probs); every other index → `strategy.denoise(...)`, whose placeholder
  zero log-prob column `_build_segment` slices off before building
  `LatentSegment`.
- **Verification.** Every sample's worker-echoed `RLData.trajectory_timesteps`
  goes through `verify_engine_used_sigmas` (scale-normalized). The patched
  `set_timesteps` pins the live scheduler's own `num_train_timesteps` onto the
  scheduler, and dispatch fails closed unless it equals the
  `timestep_scale` the request plan carries — the engine resolves that from the
  selected family's step kernel (`WAN21DiffusionStep.TIMESTEP_SCALE` /
  `WAN22DiffusionStep.TIMESTEP_SCALE`).

## WAN 2.2 A14B dual expert

`model_family: wan2.2` turns on boundary-routed dual-expert rollout.

- **Boundary agreement.** FastVideo selects the expert on
  `t >= boundary_ratio * num_train_timesteps` and UniRL's trainside routes on
  `sigma >= boundary_ratio`. Those two agree only when the model config's
  `num_train_timesteps` equals the step kernel's `TIMESTEP_SCALE`, so the engine
  `require`s exactly that at init, then pins `boundary_ratio` onto both
  `pipeline_config` and `dit_config` before boot and onto every request batch.
  FastVideo's `transformer` is the high-noise expert and `transformer_2` the
  low-noise one; the weight sync routes `high_noise.*` / `low_noise.*`
  accordingly. The engine also verifies the checkpoint's `model_index.json`
  declares both.
- **Weight sync ships a path, not a state dict.** An A14B state dict through the
  worker pipes exhausts `/dev/shm`, so `update_transformer_weights_from_path`
  sends only the `CheckpointWeightSync` file path and each worker `torch.load`s
  it (`weights_only=True`) and splits the experts itself. Every worker reads the
  whole file, so host RAM — not `/dev/shm` — bounds `sp_size`.
- **Mutating layerwise-offloaded blocks must use
  `LayerwiseOffloadHook.mutate_params_scope()`.** The hooks are circularly
  linked: the last block's `pre_forward` prefetches block 0's parameters onto
  GPU, and `wait_and_replace_params` reuses a cached `gpu_named_parameters`
  entry when one exists. Writing `state.cpu_named_parameters` directly therefore
  leaves that prefetched copy stale, so block 0 of each expert would run the
  previous policy's weights for one forward. The colocate loop currently hides
  this by tearing the generator down every step; the scope keeps it correct if
  that ever changes.
- **Loading two 14B experts needs the offload patch.** Upstream stages a whole
  DiT on GPU before attaching the layerwise hooks, which transiently exceeds the
  device. `_patches/offload.py` materializes those modules on CPU, then moves only the
  stem/head parameters the hooks left real. It reads `cpu_offload` out of
  `**kwargs`, which is why that signature is fingerprinted.

## Gotchas

- **`mp` executor only.** The engine `require`s
  `distributed_executor_backend == "mp"`: Ray actors are fresh processes that
  never receive the worker patches, so the plan would only die loudly inside
  the unpatched helper at denoising time.
- **`FASTVIDEO_WAN_SCHEDULER` must stay `unipc`** (the fork default). Other
  values make the WAN pipeline build a scheduler the patches do not target, so
  the engine rejects them at init.
- **`sde_indices: null` means opposite things across engines.** This engine
  resolves `None` to *all-steps SDE*; trainside WAN21
  (`unirl/models/wan21/diffusion.py`) reads the same `None` as *no SDE*
  (`eta=0` everywhere, no log-probs). Set `sde_indices` explicitly in the
  recipe for engine-portable behavior.
- **`RLData.collect_kl` with `kl_reward > 0` is unsupported** on this path:
  the fork's KL block needs `prev_latents_mean`/`std_dev_t` from every step,
  and UniPC columns return `None` for both → `ValueError`. UniRL never enables
  it; keep it off in `engine_kwargs` too.
- **Debug replay dumps** (`DIFFUSIONRL_FASTVIDEO_DEBUG_OUTPUT_DIR`) record the
  placeholder zero log-probs on UniPC columns — align offline ratio checks
  against SDE columns only.
- **Deterministic-index trajectories differ from other engines** until
  trainside/SGLang/vLLM-Omni adopt the model config's declared solver; the
  ratio stays honest regardless (`unirl/sde/README.md`).
