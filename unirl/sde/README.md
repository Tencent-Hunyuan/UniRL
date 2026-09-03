# SDE runtime

> **Where it fits:** cross-cutting — shared by *rollout* and train-side *replay*
> for the σ schedule and stochastic transition density. Deterministic solver
> choice can remain engine-specific.
> Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/sde-kernel-modes-new.png" alt="UniRL SDE: one multi-step denoise loop (x_T to x_0) walked twice — rollout (diffuse) walks all T steps, drawing fresh noise and keeping a Gaussian log-prob on the eta>0 SDE steps while the rest stay deterministic, and stores the trajectory; train (replay) re-walks only those SDE steps, feeding the stored transitions back through the same kernel to get new log-probs for the GRPO/FlowDPPO ratio" width="100%">
</div>

*One `strategy.denoise()` per step. **Sampling** vs **replay** is the
`prev_sample` argument; the rollout's selected indices use an SDE kernel, while
engine adapters may route the remaining deterministic indices through a
stateful ODE solver such as UniPC.*

## What it is

`unirl.sde` owns shared per-step diffusion math: the stochastic **kernels**
replayed during training (Flow / Dance / CPS), deterministic solvers (DPM2 /
UniPC), the FlowMatch **σ schedule** policy (with a per-model μ override), and
the deterministic **initial-noise (`x_T`) recipe**.

## Why it exists

The policy-gradient ratio is only meaningful when train-side replay scores each
stored stochastic transition with the same σ, model conditioning, and SDE
density used during rollout. Cross-engine output parity additionally requires
the same start noise and deterministic solver; adapters may deliberately choose
different deterministic solvers, so that stronger parity is not implied here.

## How it works

- **One SDE kernel, two modes.** Each model's diffusion stage calls
  `strategy.denoise(...)` once per step. For stochastic kernels,
  `prev_sample=None` means *sampling* (draw fresh noise); passing a `prev_sample`
  means *replay* (score the given transition, no noise drawn). Those two modes
  share the exact transition code. The math runs in fp32 (σ forced to fp32 to
  match SGLang).
- **SDE indices own the policy-gradient density.** Selected `sde_indices` get a
  stochastic transition with a real per-step Gaussian log-prob (→
  `LatentSegment.sde_logp`). Trainside loops can collapse the same SDE kernel to
  Euler with `eta=0`; adapters that need inference-scheduler parity can instead
  route non-SDE indices through `UniPCStrategy`. UniPC keeps model-output
  history across consecutive ODE steps and clears it after an intervening SDE
  jump before warming up from first order again.
- **The σ schedule** comes from `FlowMatchSchedulePolicy` (`runtime.py`), loaded
  once per actor from the checkpoint JSON (no weights). Static schedules apply the
  SD3 time-shift locally (to dodge diffusers' double-shift bug #13243); dynamic
  schedules derive μ from `(H, W)` — `compute_mu` is the single per-model override
  point (FLUX.2-klein subclasses it). `ensure_req_sigmas(req, policy)` pins the
  result onto `req.sigmas` at the top of every *diffusion* engine's `generate`, so
  every backend samples on the exact schedule the trainer will replay (AR-only
  paths skip it).
- **The `x_T` recipe** (`noise.py`). The driver doesn't ship the noise tensor — it
  ships a recipe (per-sample group ids + latent shape). Each engine calls
  `regen_initial_noise(...)`, which draws on **CPU in fp32** with a per-group seeded
  generator, then casts to device. CPU randn is bit-stable across machines, so every
  engine starts each rollout from byte-identical noise.

**Extending it:** a new kernel subclasses `SDEStrategy` (or `StepStrategy` for a
deterministic ODE solver), wired under `pipeline.strategy` or an engine adapter. A
per-model σ override subclasses `FlowMatchSchedulePolicy` and overrides only
`compute_mu`. A new SDE-index schedule is *not* here — it's a `TimestepScheduler`
in `utils/scheduler_utils.py`, wired under `sampling.scheduler`. DanceGRPO/MixGRPO
add no kernel: DanceGRPO swaps in `DanceSDEStrategy` under `pipeline.strategy`,
MixGRPO keeps `FlowSDEStrategy` and adds a `WindowScheduler` under
`sampling.scheduler`.

## Gotchas

- **`DPM2Strategy` can't train** — it returns `log_prob=None` everywhere, so
  GRPO/FlowDPPO have no ratio. Eval-only, and stateful (needs `init_schedule`/`reset`).
- **`UniPCStrategy` is deterministic and stateful** — it also returns
  `log_prob=None`, requires `init_schedule`, and only reuses history across
  consecutive ODE indices. An SDE jump must reset its history.
- **σ silently arrives as float64** — `torch.linspace` (the static-σ branch) defaults
  to float64, so without `denoise`'s `.float()` cast the transition computes in float64
  while SGLang uses float32; the `1/(2σ²)` term amplifies the gap into the replayed
  log-prob and skews the ratio.
- **Only `x_T` is reproducible, not the per-step SDE noise** — `denoise` hard-codes
  `generator=None` into every step ("DONOT PASS GENERATOR HERE — it hurts diversity"),
  so the intermediate stochastic transitions are unseeded. That is *why* replay
  re-scores the stored `prev_sample` instead of re-drawing it.
- **`initial_latents` wins over the recipe** — a shipped latent (img2img / i2v
  first frame) is used verbatim; the recipe only fills the t2i `x_T`.
- **Don't "simplify" the static-σ branch to call diffusers** — it exists to avoid
  the double-shift bug (#13243) and is tagged `DELETE-WHEN` for when upstream is
  fixed.
