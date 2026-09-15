# Algorithms

> **Where it fits:** the loss half of the *train* step —
> rollout → reward → advantage → **train** → sync. In: a track with advantages
> (supervised / teacher-anchored algorithms opt out via `requires_advantages = False`).
> Out: gradients on the model (the optimizer step in `../train` consumes them).
> Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/algorithm-contract-new.png" alt="UniRL algorithm contract: a StageAlgorithm combines new_logp from replay, the frozen pi_old anchor, and advantages into a loss (four interchangeable families: GRPO, FlowDPPO, DRPO, and DiffusionNFT as the ratio-free exception), and declares knobs — requires_ema_rollout, supports_multi_update, anchor_fields/recomputes_anchor — that reconfigure the sampler and train stack around it" width="100%">
</div>

*A `StageAlgorithm` is two things: a **loss combine** (`stage.replay → new_logp`, mixed with the frozen **π_old** anchor and advantages — four interchangeable families) and a few **declared knobs** (`requires_ema_rollout`, `supports_multi_update`, `anchor_fields`/`recomputes_anchor`) that reconfigure the sampler and the train loop around it.*

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
`anchor_fields` / `recomputes_anchor`; `TrainStack` does the per-slice recompute.
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
- **DPO is the offline preference family** (`dpo.py`) — the only algorithm here
  that needs neither rollout nor advantages. It replays the same segment twice,
  once with the LoRA adapter live and once under `adapters_disabled` (the
  reference policy), reduces both to per-sequence log-probs, and applies the
  Bradley-Terry objective to adjacent chosen/rejected rows. Despite the name it
  is unrelated to `DPPO`/`FlowDPPO`, which are Divergence-PPO trust regions.
- **The anchor contract — the subtle part.** bf16 forwards are batch-shape
  sensitive, so a π_old anchor computed at a different geometry than `new_logp`
  drifts the on-policy ratio off 1 (and FlowDPPO's KL off 0). Algorithms just declare
  `anchor_fields` (which segment fields to freeze) and `recomputes_anchor`
  (whether `prepare_segment` replays); `TrainStack` then recomputes the anchor over
  the *exact same* mini/micro slices it will train on. No hardcoded field names.
- **Variants are recipes, not classes.** DanceGRPO and MixGRPO are `FlowGRPO`
  with a different SDE strategy or a windowed index scheduler. Add a class only when
  the loss math itself changes.

**Extending it:** a new diffusion loss subclasses `StageAlgorithm`, calls
`stage.replay(...)`, computes a per-element loss, and `(loss * loss_scale).backward()`;
if it needs multi-update, set `anchor_fields` and `supports_multi_update = True` and
declare `recomputes_anchor` when `prepare_segment` must follow the planned micro
geometry (see `FlowGRPO`). A new AR loss mirrors `GRPO` (early-return on an empty
segment, expand advantages per token), keeping `supports_multi_update = False`.

## Gotchas

- **`old_logp_source: rollout` with a replay-only engine** — a separate-worker
  SGLang rollout emits no per-step `sde_logp`, so the `rollout` source raises in
  `prepare_segment`. Use `replay` (the cost is one extra `torch.no_grad` replay).
- **`num_updates_per_batch > 1` on DiffusionNFT** raises in `TrainStack.__init__` — DiffusionNFT keeps
  the default `supports_multi_update = False`. Multi-update algorithms freeze their
  declared anchors: `FlowGRPO`/`FlowDPPO` prepare `sde_logp`; `GRPO` always reuses
  the rollout log-prob, while DPPO/CPPO/DRPO do so under `old_logp_source: rollout`
  and recompute it over the planned training micros under `replay`.
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
- **DPO's pair layout is positional, so batch geometry is load-bearing.** A
  preference `Part` is `2P` rows laid out `[chosen0, rejected0, chosen1, ...]`,
  and the loss recovers the pair with `[0::2]`/`[1::2]`. Nothing downstream
  knows a pair is a unit: `pytree_chunk` shards contiguously for `DP_SCATTER`
  and `CountPlanner` slices contiguously into micros, so an odd rows-per-rank or
  an odd `micro_batch_size` silently severs pairs. Hence `micro_batch_size: 2`
  and a recipe `batch_size` counting **pairs** — with `batch_size % dp_size == 0`
  already enforced by the trainer, pair-counted batches make rows-per-rank even
  automatically. `DPO._split_adjacent` raises on an odd count rather than
  training on mismatched rows.
- **DPO is LoRA-only, and the adapter must not reach the frozen towers.** The
  reference policy is the *same* weights with adapters disabled
  (`_resolve_reference_model` raises without a LoRA adapter), so an adapter
  injected inside `visual`/`audio_tower` would move the reference too and shrink
  the objective toward zero. Keep `exclude_modules: ".*visual.*|.*audio_tower.*"`.
- **DPO's first step should report `dpo_loss ≈ log 2 = 0.693`.** PEFT zero-inits
  LoRA `B`, so before any optimizer step the policy and the adapter-disabled
  reference are the same function and the logit margin is exactly 0. A first step
  far from 0.693 means the reference is not actually frozen (or the pairing is
  misaligned). Expect `reward_accuracy == 0` there too — exact ties fail the
  strict `>`, so it is not a bug. Note what this check *cannot* see: it holds for
  **any** value of `sampling_temperature`, because a zero margin stays zero under
  any scaling. It validates the reference and the pairing, not the scaling.
- **The supervised span decides how much of the margin is the EOS token, and with
  `average_log_prob=false` that term does not cancel.** The margin is a *sum* over
  supervised response tokens, so every extra supervised token adds
  `(π_c − ref_c) − (π_r − ref_r)` for that position. `ARPreferenceTrackBuilder`
  appends EOS and supervises it (`append_eos=true`); verl-omni marks only the
  assistant text span its chat template produces, which is exactly **one token fewer
  per branch on every row** (measured on 5 rows × 2 branches: 7/8, 1/2, 5/6, 12/13,
  9/10, …). That single position carries a large share of the objective. Dropping it
  at scoring time costs this stack **−12.9pp accuracy and 54% of the margin**
  (1.1150 → 0.5144) but costs verl only −2.5pp and 9%, and on the common span verl
  scores *higher* (0.7583 vs 0.7083). `P(EOS)` also depends on what precedes it — a
  one-word answer ends differently from a sentence — so the term correlates with
  answer length, which is the mechanism behind the length-sensitivity difference
  (+0.231, CI [+0.120, +0.335]) reported in `datasets/omni_preference/README.md`.
  Set `track_builder.append_eos: false` to match verl's span. Neither choice is
  wrong, but they are different objectives, and comparing across them is not a
  comparison of implementations.
- **Dense-padded pairing shifts the margin slightly, and the shift tracks the length
  difference.** `Qwen3OmniARStage.replay` forwards a pair as a dense `[B, T]` batch
  padded to the longer branch, so the shorter branch carries the padding; verl-omni
  instead runs `pad_mode=no_padding` over nested tensors, where each branch sees only
  its own tokens. Forwarding each branch alone and re-deriving the margin changes it
  by **mean +0.003, median +0.005, max 0.13**, and that shift correlates with
  `len(chosen) − len(rejected)` at **+0.21** — so padding is not perfectly inert and
  the leak is length-dependent, which is the same direction as the length-sensitivity
  gap against verl. It is small: 55/60 ranking decisions are unchanged, and the
  accuracy difference (0.70 paired vs 0.65 alone) is well inside the ±8.4pp binomial
  SE at n=60 and not significant (McNemar p=0.37).

  Repeating it powered, over the full 600-row split with every row scored both ways,
  separates the two halves of the claim cleanly. The **leak is real and
  length-dependent**: the shift correlates with `len(chosen) − len(rejected)` at
  **+0.221, 95% CI [+0.144, +0.296]**, excluding zero, and it splits by direction —
  padding *helps* when chosen is longer (n=478, accuracy +1.67pp, mean shift +0.027)
  and *hurts* when chosen is shorter (n=94, −5.32pp, mean shift −0.006). So dense
  padding systematically favours the longer branch, which is the mechanism a
  length-sensitivity difference would need. But its effect on **overall** accuracy is
  not resolvable: +0.83pp, 95% CI [−0.83, +2.50], McNemar p=0.42 (15 vs 10 discordant
  of 600), because this split is 478/94 skewed toward chosen-longer and the two
  directions cancel. The CI's upper bound is below the ~3.9pp residual it was meant to
  explain, so padding is not that explanation. On a length-balanced set the
  cancellation would not hold, which is the case where switching to a no-padding
  forward would actually matter.
- **The segment-sum must not use `index_add`/`scatter_add_`.** Both accumulate
  with CUDA atomics, so the addition order varies between otherwise identical
  calls and the per-sequence sum is not reproducible. Measured spread on one
  fixed input over 8 calls: `index_add` 0 at 64 tokens, 1.2e-03 at 622, 3.9e-02
  at 4096, 1.6e-01 at 16384; `scatter_add_` is no better (1.9e-01 at 16384).
  This is invisible in the loss and in the forward — the `replay` output is
  bitwise identical across calls (30/30 rows measured) — and surfaces only after
  the reduction, where it made the policy and the *adapter-disabled* reference
  differ at zero adapter delta. The visible symptom was the log-2 control
  reporting `reward_accuracy = 0.15` instead of 0: with a true margin of exactly
  0, tie-breaking noise of 1e-07 is resolved by the strict `>`, and because the
  error scales with length it hit long image rows (6/20) and never short audio
  rows (0/20), which reads exactly like a modality-dependent modelling effect.
  `_reduce_to_sequences` therefore scatters to `[S, Lmax]` and sums along a fixed
  axis, which is a shape-determined tree reduction; the control then returns
  exactly 0.0 with every margin identically zero.
- **What a bit-identical loss does and does not prove.** Reproducing a reference
  implementation's loss to `|d| = 0` shows the formula is right *given the same
  log-probs*. It says nothing about the rest of the chain — manifest conversion,
  prompt rendering, media injection, tokenisation, masking scope, metric
  aggregation — and a wrong input there yields a faithfully-computed wrong number.
  Every defect found while building this algorithm was invisible at the loss layer:
  pad rows polluting an eval mean, `target_parameters` missing from the checkpoint
  meta (silently rebuilding an attention-only adapter on resume), a rank-dependent
  all-reduce width hanging NCCL for 1800 s, `image_max_pixels` silently inert for
  image-only rows, a generation-default sampling temperature rescaling the
  objective, and a modality marker surviving in the prompt as literal tokens.
  Pair the loss check with checks that can see those: give each knob two values and
  require the output to change, assert declared metric keys match emitted ones, and
  compare what the docs claim against what the code does. Two configurations that
  should differ but produce bit-identical numbers are a bug signature, not a
  reassurance.
- **`sampling_temperature` defaults to 1.0 here, not to `ARSamplingParams`.**
  `replay` divides the `lm_head` logits by it, and that does not cancel out of
  `(π_c − π_r) − (ref_c − ref_r)`: `log_softmax` is non-linear in the temperature,
  so the margin scales by roughly `1/T` and the effective `beta` moves with it.
  Offline DPO never samples, so inheriting a *generation* default (0.7) would
  silently rescale the objective against reference implementations, which compute
  preference log-probs at 1.0.
- **DPO allows `num_updates_per_batch > 1` for a reason the other families do
  not.** The multi-update gate exists to stop a *moving π_old anchor*, but DPO
  has no π_old: its reference is the adapter-disabled base — frozen weights,
  recomputed inline in the same micro geometry as the policy forward. So N
  optimizer steps per batch is sound, and it is how the objective reaches a
  useful margin in a modest number of data batches. Two knock-ons: the LR
  schedule counts *optimizer* steps, so `total_steps` must be
  `num_steps × num_updates_per_batch`; and each update's slice must stay a
  multiple of `rows_per_record` so no pair is split across updates (the trainer
  checks both).
- **AR `sampling_temperature` must equal the rollout `sampling.temperature`** —
  `ARStage.replay` rescales logits by it (`log_softmax(logits / T)`) to match SGLang's
  distribution; when unset it silently falls back to the `ARSamplingParams` default,
  *not* the request Sample's actual temperature, biasing every ratio with no raise. Watch
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
