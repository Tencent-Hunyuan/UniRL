# Algorithms

> **Where it fits:** the loss half of the *train* step —
> rollout → reward → advantage → **train** → sync. In: a track with advantages
> (supervised / teacher-anchored algorithms opt out via `requires_advantages = False`).
> Out: gradients on the model (the optimizer step in `../train` consumes them).
> Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/algorithm-contract-new.png" alt="UniRL algorithm contract: a StageAlgorithm combines new_logp from replay, the frozen pi_old anchor, and advantages into a loss (four interchangeable families: GRPO, FlowDPPO, DRPO, and DiffusionNFT as the ratio-free exception), and declares knobs — requires_ema_rollout, supports_multi_update, anchor_fields/recomputes_anchor — that reconfigure the sampler and train stack around it" width="100%">
</div>

*A `StageAlgorithm` is two things: a **loss combine** (`stage.replay → new_logp`, mixed with the frozen **π_old** anchor and advantages — four interchangeable families) and a few **declared knobs** (`requires_ema_rollout`, `supports_multi_update`, `anchor_fields`/`recomputes_anchor()`) that reconfigure the sampler and the train loop around it.*

## What it is

`unirl.algorithms` is the train-side loss half of the framework. Each algorithm is
a `StageAlgorithm` that takes a rollout track with advantages already attached,
replays the stage at the current weights, computes a policy-gradient loss, and
calls `backward()`. It owns the loss math and nothing else — no optimizer, no
model, no data.

## Why it exists

The four objectives need *different things from the rest of the train step*, and
`StageAlgorithm` is where that divergence is declared without forking the trainer.
Two class attributes drive the surrounding machinery: `requires_ema_rollout` tells
the sampler whether to roll out under EMA weights (DiffusionNFT sets it `True`; GRPO keeps it
`False` so rollout and replay share weights and the step-1 ratio is exactly 1), and
`supports_multi_update` tells `TrainStack` whether one rollout may be split into N
optimizer steps (it *raises* if a `False` algorithm meets `num_updates_per_batch > 1`).
The π_old anchor geometry is *not* centralized here — the algorithm only declares
`anchor_fields` / `recomputes_anchor()`; `TrainStack` does the per-slice recompute.
So this module keeps four rollout/update **contracts** selectable at the loss node,
not just three-tensor arithmetic.

## How it works

- **The loop.** The trainer builds one algorithm per track and hands it to a
  `TrainStack`. Per rollout the stack runs `prepare_segment` once (freeze the π_old
  anchor), then `num_updates_per_batch` optimizer steps over disjoint mini-batches,
  each a micro-batch loop of `compute_loss_and_backward`.
- **`compute_loss_and_backward` is pure ratio/quadratic math.** It delegates the
  whole forward — CFG batching, noise prediction, SDE stepping — to
  `stage.replay(...)` and gets back `new_logp` (DiffusionNFT is the exception: it runs its own
  dual-adapter loop via `predict_noise_at_step`). The families: GRPO is a
  PPO-clipped ratio (`flowgrpo.py` / `grpo.py`); FlowDPPO masks `-A·r` by a
  Gaussian-KL-vs-advantage criterion (`flowdppo.py`); DiffusionNFT is a dual-adapter
  reconstruction MSE (`diffusionnft.py`); DRPO is a token-adaptive SPO quadratic (`drpo.py`);
  DiffusionOPD is the teacher-anchored family — a per-step Gaussian KL against frozen
  teacher LoRA adapters (backend-owned `frozen_adapters`), distillation rather than RL:
  it ignores advantages and picks its teacher from the batch's `metadata["domain"]`
  (`diffusionopd.py`).
- **The anchor contract — the subtle part.** bf16 forwards are batch-shape
  sensitive, so a π_old anchor computed at a different geometry than `new_logp`
  drifts the on-policy ratio off 1 (and FlowDPPO's KL off 0). Algorithms just declare
  `anchor_fields` (which segment fields to freeze) and `recomputes_anchor()`
  (whether `prepare_segment` replays); `TrainStack` then recomputes the anchor over
  the *exact same* mini/micro slices it will train on. No hardcoded field names.
- **Variants are recipes, not classes.** DanceGRPO and MixGRPO are `FlowGRPO`
  with a different SDE strategy or a windowed index scheduler. Add a class only when
  the loss math itself changes.

**Extending it:** a new diffusion loss subclasses `StageAlgorithm`, calls
`stage.replay(...)`, computes a per-element loss, and `(loss * loss_scale).backward()`;
if it needs multi-update, set `anchor_fields` and `supports_multi_update = True` and
mirror `FlowGRPO`. A new AR loss mirrors `GRPO` (early-return on an empty
segment, expand advantages per token), keeping `supports_multi_update = False`.

## Gotchas

- **`old_logp_source: rollout` with a replay-only engine** — a separate-worker
  SGLang rollout emits no per-step `sde_logp`, so the `rollout` source raises in
  `prepare_segment`. Use `replay` (the cost is one extra `torch.no_grad` replay).
- **`num_updates_per_batch > 1` on DiffusionNFT** raises in `TrainStack.__init__` — DiffusionNFT keeps
  the default `supports_multi_update = False`. The four that allow it all freeze a
  stable anchor: `FlowGRPO`/`FlowDPPO` freeze `sde_logp` once in
  `prepare_segment`, while `GRPO`/`DRPO` reuse the rollout log-prob as the anchor
  for all N steps (verl `bypass_mode` parity — so the AR ratio also carries the
  rollout-vs-train engine gap, by design).
- **FlowDPPO isn't fully on-policy under `rollout`** — it always replays `sde_means`
  (KL = 0) but keeps the engine's `sde_logp`, so its ratio isn't pinned to 1. Use
  `replay` to also pin the ratio.
- **`params` must reuse the rollout `guidance_scale`/`eta`/`shift`** — single-track
  recipes bind `params: ${sampling}`; composed recipes bind the sub-block (e.g.
  `${sampling.diffusion}`). A mismatch silently skews log-probs.
- **DiffusionOPD's ODE recipes keep `sampling.eta` vanishing-but-nonzero** (e.g. `1e-6`,
  so `stage.replay` still emits log-probs) **with `add_kl_coefficient=false`**. Never pair a
  near-zero `eta` with `add_kl_coefficient=true` — the KL divides by a transition std that
  scales with `eta` (the algorithm raises at init on `eta == 0`, but cannot judge "too small").
- **AR `sampling_temperature` must equal the rollout `sampling.temperature`** —
  `ARStage.replay` rescales logits by it (`log_softmax(logits / T)`) to match SGLang's
  distribution; when unset it silently falls back to the `ARSamplingParams` default,
  *not* the engine's actual temperature, biasing every ratio with no raise. Watch
  `rollout_replay_logp_absdiff_mean` — it should be ~0 on an on-policy step.
- **DiffusionNFT's `ref_deviation_coef > 0` anchors to the LoRA-disabled base, not the EMA shadow** — the
  shadow tracks the policy by construction, so anchoring to it would bound no drift. The reference
  is a third `predict_noise_at_step` per trained timestep, on top of the trainable and shadow ones;
  it runs under `no_grad` and needs no backward (~+24% train phase rather than +50%), and under
  `train_timestep_mode: all` it is paid once per timestep in the K-loop. That wall-clock number is
  not the whole cost — the penalty competes with the reward gradient, so at equal step count a
  `ref_deviation_coef > 0` run settles at a lower proxy reward than a `ref_deviation_coef = 0` one; budget steps for that
  rather than reading the trade off the timing alone. `ref_deviation_coef=0` returns a `None` reference before
  any of that, so it stays bit-identical to a build without the term.
- **`ref_prediction_deviation` is the raw mean-difference², not the σ-normalized KL** — the metric
  is named for what it measures rather than for `ref_deviation_coef`, which weighs it: the penalty is
  `((new_pred - ref_pred)**2).mean()`, the same formula as the neighbouring `prediction_deviation`
  with the anchor swapped from the EMA shadow to the LoRA-disabled base, so the two share a scale
  and can be read side by side as drift-from-shadow against drift-from-base. DiffusionNFT trains on
  a freshly noised `xt` rather than the rollout trajectory, so it has no `stage.replay` step indices
  to hand `_transition_sigma`. `segment.sigmas` is present (`train_timestep_mode: all` requires it)
  and so is `t_batch`, so a time weight is available if one is ever wanted; what is absent is the SDE
  transition std itself, because every NFT recipe runs `eta: 0.0` and that std vanishes with `eta` —
  the same trap the DiffusionOPD bullet above raises on. That leaves the `add_kl_coefficient=false`
  variant minus the `/2`, so the number is **not** comparable to FlowGRPO/FlowDPPO's `kl_ref_mean`, which carries
  `_gaussian_kl_div`'s `/(2σ²)`. Measuring on the prediction rather than the reconstructed `x0`
  drops the `t²` Jacobian of `xt - t*pred`, spreading pressure uniformly over trained timesteps
  instead of `t²`-weighting it — the magnitude still moves with `t` (training lower timesteps raises
  it several-fold).
- **`adv_std_saturate` is how many advantage σ map to `r = 0` or `1`** — write `C` for the value and
  `q = clamp(adv, ±C)/C ∈ [-1, 1]` so `r = 0.5 + q/2`. Then `total` splits exactly into
  `(C/2)·mean(pos_loss + neg_loss)/β` plus `(C/2)·mean(q · (pos_loss − neg_loss))/β`. Both halves
  carry the same `C/2`, so `total = policy_loss * adv_std_saturate` is a **pure gain**: it cancels the
  `1/C` inside `q` and leaves the objective's shape untouched, and the learning rate absorbs it. The
  `/β` is a gain as well — the raw NFT gradient scales linearly in `β` (grad-norm/β is constant over
  `β = 0.05 … 1.0`), so dividing by it makes the step β-independent. Raising `C` shrinks `E|q|`
  (0.63 at `C=1` versus 0.16 at `C=5`) and the signal-to-symmetric ratio falls to 0.29x across that
  range. It is a first-class RL knob, not a safety clip.
- **`adv_std_saturate: 5.0` de-contrasts ~3.4x against the paper's parameterization** — DiffusionNFT
  (arXiv:2509.16117, Alg. 1) uses `r = 0.5 + 0.5·clip(r_norm / Z_c, -1, 1)` with `Z_c` "some
  normalizing factor, which could take the form of a global reward std", and its loss carries **no**
  outer scale. UniRL's advantages already arrive std-normalized (`Part.compute_advantages(normalize=
  True)` ⇒ `(reward − group_mean)/(group_std + eps)`), so `adv_std_saturate` divides a z-score by another
  5 and `r` spans only `[0.066, 0.966]` rather than saturating at 0 and 1. Consequence when porting
  coefficients: the policy term carries an overall `adv_std_saturate/β` gain that the upstream objective
  does not — 50x at the H3 recipe's `β=0.1`, `adv_std_saturate=5` — so `ref_deviation_coef` is **not** on
  the same scale as verl-omni's `ref_kl_coef`. Check that factor before copying a value across.
  (verl-omni documents its own knob as a "prediction-space reference MSE regularizer", the same
  reading of the quantity this file takes above.)
- **The reference penalty is uninformative while the adapter delta is sub-ULP** — both operands come
  straight out of a bf16 forward, and standard LoRA init (`B=0`) starts the delta at exactly zero.
  Below roughly 2 bf16 ULP (RMS delta ≲ 0.01 against O(1) predictions) the difference is mostly
  quantization noise: measured gradient-direction cosine against the fp32 answer is 0.52 at RMS
  1e-3 and 0.97 at 1e-2. At the deviations these runs actually reach (0.008 → 0.057, i.e. 23 → 61
  ULP) bf16 costs nothing measurable — 1.00x error, cosine 0.9999 — so the term is sound once the
  adapter has moved, and merely inert before that. Upcasting inside the penalty does not change
  this: the operands are already rounded when the forward returns them. DiffusionOPD's
  `fp32 before squaring` is not the same situation — it upcasts scheduler-computed
  `prev_sample_means`, not raw network output.
