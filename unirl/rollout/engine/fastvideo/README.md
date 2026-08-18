# FastVideo rollout engine

> **Where it fits:** the `fastvideo` backend of the *rollout* step — an
> in-process `VideoGenerator` driven one prompt at a time through
> `executor.execute_forward`. Full map: [`../README.md`](../README.md).

*Runs WAN 2.1 rollouts on the pinned FastVideo RL fork and re-routes its
hard-coded deterministic Euler steps through UniRL's canonical UniPC solver
(`unirl/sde/unipc.py`), so rollouts follow the checkpoint's native inference
solver without integer-timestep loss.*

## The pin

The integration targets
**`Zcchill/FastVideo@7fe1d7db9a0b8aebb46679e7924f597431f23665`** (a snapshot of
hao-ai-lab/FastVideo PR #1222, the Wan2.1 RL pipeline), injected via
`cfg.fastvideo_path` / `$FASTVIDEO_PATH` — there is no pip pin. `_unipc.py`
therefore fingerprints the patched surface at patch-install time (engine init
and every spawned worker): the parameter lists of
`FlowUniPCMultistepScheduler.set_timesteps` and `sde_step_with_logprob`, the
`WorkerMultiprocProc.worker_main` entrypoint, and (engine side) the
`ForwardBatch.RLData` fields. Drift fails closed at init instead of mid-rollout.
Before editing a patch, read that exact commit (CLAUDE.md: monkey-patch
doctrine), then update the fingerprints together with the patch.

## How the canonical UniPC path works

- **σ SSOT.** The engine sends `FlowMatchSchedulePolicy`'s already-shifted
  canonical σ verbatim (no shift pre-image). The patched `set_timesteps`
  rejects any schedule transform on external sigmas, appends the terminal
  zero, and keeps **float32** model timesteps
  (`σ · WAN21DiffusionStep.TIMESTEP_SCALE`) — stock stores `int64`, which
  truncates conditioning (`833` vs `833.333…`, the drift #248 tracked).
- **Solver SSOT.** `WAN21PipelineConfig.unipc_*` declares the deterministic
  solver. The engine embeds it as a `UniPCSpec` inside the per-request
  `FastVideoUniPCPlan`, carried through the fork's str-typed `RLData.sde_type`
  with `sde_step_indices=None` so every index reaches the patched helper. The
  worker builds `UniPCStrategy` from the plan only after verifying the
  checkpoint scheduler config matches the spec, and resets multistep history
  across SDE jumps.
- **Dispatch.** Plan indices → the fork's Dance/Flow SDE helper (real
  log-probs); every other index → `strategy.denoise(...)`, whose placeholder
  zero log-prob column `_build_segment` slices off before building
  `LatentSegment`.
- **Verification.** Every sample's worker-echoed `RLData.trajectory_timesteps`
  goes through `verify_engine_used_sigmas` (scale-normalized), and
  `_wan_timestep_scale` pins `num_train_timesteps == 1000` against the WAN21
  model contract.

## Gotchas

- **`mp` executor only.** The engine `require`s
  `distributed_executor_backend == "mp"`: Ray actors are fresh processes that
  never receive the worker patches, so the plan would only die loudly inside
  the unpatched helper at denoising time.
- **`FASTVIDEO_WAN_SCHEDULER` must stay `unipc`** (the fork default). Other
  values make the WAN pipeline build a scheduler the patches do not target, so
  the engine rejects them at init.
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
