# sglang diffusion patches

> **Where it fits:** inside the *rollout* step — the `sglang_diffusion` engine's
> boot path installs these before importing `DiffGenerator`.
> Full map: [`../../../README.md`](../../../README.md).

*Re-hosts the `sglang-drl` fork's RL additions as in-process monkey-patches on
**stock upstream** sglang, so UniRL tracks upstream instead of carrying a hard fork.*

## What it is

13 patches are installed by one idempotent `SglangDiffusionHijack.hijack()`
(`hijack.py`), backed by 2 support modules (`io_struct.py` and
`memory_saver.py`). Every patch is `setattr`, dataclass-field injection, or an
AROUND-wrap — **no sglang source is edited**, so a version bump is a re-pin, not
a re-merge.

## Why it exists

The fork (`sglang-drl`) added rollout behavior that stock sglang still does not
fully provide: driver-pinned σ, driver-authoritative `x_T`, per-sample SDE noise,
tagged CUDA-VM sleep/wake, distributed weight push, and conditions emission.
Depending on the fork means re-merging it on every upstream release. Re-homing
only the missing semantics as patches keeps each upstream upgrade reviewable.

## How it works

`hijack.py` installs by direct `setattr`, not sglang's `HookRegistry`: the
diffusion scheduler/worker runs under forced spawn
(`diffusion_generator.py: mp.set_start_method("spawn", force=True)`) and the
diffusion path never calls srt `load_plugins()`, so `HookRegistry` is not wired in.
A parent-only patch would silently no-op in the worker —
`wrap_mp_process_for_children` propagates the install into every spawn child.

Install **before** importing `DiffGenerator` (which forces spawn at import) and
before `from_pretrained` spawns the scheduler. Idempotent; safe from parent and
child. Patch installation is fail-closed: an incompatible pinned SGLang layout
aborts engine boot instead of silently running without the rollout contract.

**Extending it:** a new patch is a module here with an idempotent `install()`
called from `hijack.py`, plus a row below. Prefer AROUND-wrap over REPLACE — a
REPLACE re-vendors upstream source and must be hand-re-synced on every bump.

## The patches

Single definition sites — both the UniRL adapter and the scheduler import request
classes from here, so `request_handlers` (keyed by `type(req)`) matches on class
identity:

| Module | What it defines |
| --- | --- |
| `io_struct.py` | The 5 UniRL-only distributed-sync and tagged sleep/wake request structs |
| `memory_saver.py` | Tagged CUDA-VM sleep/wake helper layered over `torch_memory_saver` |

Re-homed fork surface:

| Module | What stock upstream does wrong | DELETE-WHEN |
| --- | --- | --- |
| `patch_gpu_worker.py` | Upstream owns tensor/LoRA updates and ordinary CPU sleep/wake; UniRL adds external-process-group broadcasts plus tagged CUDA-VM sleep/wake | upstream accepts external distributed broadcasts and tagged memory-saver regions |
| `patch_scheduler.py` | Registers those UniRL-only verbs and blocks generation while resumed CUDA-VM regions contain disposable weights | same |
| `patch_safe_unpickler.py` | sglang's `SafeUnpickler` (CVE-2025-10164 mitigation) allowlists `builtins`/`torch`/… but **not** `unirl.`, so the first full-weight push dies. Must be installed **in every process that deserializes** | upstream allowlists are configurable |

Driver-authoritative rollout contract — these are what keep the GRPO ratio honest:

| Module | What stock upstream does wrong | DELETE-WHEN |
| --- | --- | --- |
| `patch_rollout_trajectory.py` | **The gradient-killer.** `_merge_expanded_singletons` keeps only output 0's trajectory, so a grouped request returns K identical per-sample trajectories → advantages cancel → `grad_norm≈0` with flat reward. SDE math, conditions, advantages and the `x_T` recipe were all ruled out first | upstream merges per-output trajectories |
| `patch_set_timesteps.py` | `FlowMatchEulerDiscreteScheduler.set_timesteps` **always** shifts provided sigmas, double-shifting the driver's already-final schedule (`sigma_verify` then fails). Three mutation paths must all be neutralized — the old `mu = 0.0` trick is identity only for `exponential` and **zeroes the schedule** under `linear` | upstream honours provided σ verbatim |
| `patch_sampling_io.py` | `SamplingParams` rejects the four driver fields. They must be injected as **real dataclass fields**, because upstream `generate` runs `dataclasses.replace` per prompt and would silently drop plain attributes | upstream carries the fields |
| `patch_latent_prep.py` | The provided-latents branch is only `latents.to(device)`: it neither expands `[1, …]` to per-sample `[batch_size, …]` nor runs packing, so packed models (FLUX.2-Klein / M3) leave `batch.latent_ids = None` → `AttributeError` in `get_freqs_cis` | upstream mirrors the randn branch |
| `patch_denoising.py` | One generator per request, reused across steps — every GRPO-group sample is a separate B=1 request reseeded to the same `batch.seed`, so all samples draw **byte-identical per-step `z_t`**; exploration freezes and reward regresses after ~100 rollouts. The blake2b derivation must match `make_step_generators`. Same root cause as the vLLM-Omni BAGEL fix (PR #89) | upstream keys noise per sample |
| `patch_conditions.py` | `GenerationResult` / `OutputBatch` do not carry text-encoder embeds and `SamplingParams` rejects `return_prompt_embeds`, so `populate_conditions=true` recipes crash. Upstream's `TextEncodingStage` already populates the positive and (under CFG) negative batch fields, so this only **copies**, never re-encodes. Both `DecodingStage.forward` and `_req_to_output_batch` must be wrapped — the monolithic path bypasses the latter | upstream emits conditions |
| `patch_dance.py` | Upstream supports only `sde`/`cps`/`ode`; FLUX.2-Klein's primary objective is `dance` (constant `std_dev_t = eta`). **The ONLY REPLACE patch** — it re-vendors upstream `flow_sde_sampling` with one extra `elif`, so it must be hand-re-synced on any bump. Parity with `unirl/sde/kernels.py:DanceSDEStrategy` is verified by hand only | upstream adds a `dance` branch |
| `patch_wan_scheduler.py` | `WanPipeline` hardcodes `FlowUniPCMultistepScheduler`, which has no `SchedulerRLMixin` — no SDE log-prob path, and `set_timesteps` rejects the pinned σ list. Hard dependency on `patch_set_timesteps` | upstream's WAN pipeline is RL-capable |
| `patch_ltx2_rollout_sde.py` | LTX-2's stage overrides drop 6 pieces of the rollout contract the base stages provide | upstream's LTX-2 stages are rollout-complete |

Version-window and environment bridges:

| Module | What stock upstream does wrong | DELETE-WHEN |
| --- | --- | --- |
| `patch_vae_decode_safe.py` | **Opt-in, default no-op** — set `UNIRL_DISABLE_CUDNN=1`. Diagnostic for a `munmap_chunk(): invalid pointer` inside `Conv2d._conv_forward` on cuda-compat-13 + driver 535 | the compat-layer bug is gone |

## Gotchas

- **`patch_dance` is the only REPLACE** — every other patch is additive. On an
  sglang bump, re-sync its re-vendored `flow_sde_sampling` body by hand first.
- **Install order matters.** `hijack.py` must run before `DiffGenerator` is
  imported; `patch_wan_scheduler` requires `patch_set_timesteps`;
  `patch_scheduler` requires `patch_gpu_worker`.
- **Patched bodies copied verbatim from the fork are nested functions**, so their
  free globals resolve in *this* package's scope, not sglang's — re-bind every
  name locally or the patch fails at call time, not at install time.
- **Two patches are silent when they regress**, so check them first when the GRPO
  ratio drifts: `patch_rollout_trajectory` (identical trajectories → zero
  gradient) and `patch_denoising` (identical per-step noise → frozen exploration).
