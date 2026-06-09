# DRPO — Divergence-Regularized Policy Optimization

Implementation of the paper "Rethinking the Divergence Regularization in LLM RL"

`DRPO` is the repo's LLM RL algorithm. It replays the sampled tokens, compares
train-side log-probs against the rollout log-probs, expands sample-level advantages to
tokens, and applies the paper's smooth quadratic DRPO loss (Eq. 8).

- **Loss:** [`unirl/algorithms/drpo.py`](../unirl/algorithms/drpo.py) (`DRPO`, `_drpo_loss`)
- **Recipe (SGLang):** [`examples/ar/qwen3_drpo_4b_base_dpao_sglang.yaml`](../examples/ar/qwen3_drpo_4b_base_dpao_sglang.yaml) — Qwen3-4B-Base on DAPO-Math
- **Config extract:** [`config.yaml`](config.yaml)

Lineage: **PPO → GRPO → SPO → DPPO → DRPO**.

## Why divergence instead of PPO ratio clipping

LLM RL is almost always off-policy: the rollout engine (SGLang) and the training engine
differ numerically, and one batch of rollouts is split into gradient steps, so the updated
policy `π` is not the behavior policy `µ` that sampled the tokens. PPO/GRPO build the trust
region from the ratio `r_t = π/µ`, but for a long-tailed vocabulary that is a poor proxy
for distributional shift: a rare token gives a huge ratio after a tiny probability change,
while a common token moves a lot of mass at a modest ratio (paper §1–2.4).

DPPO addresses this by replacing ratio-based clipping with a **Binary-TV mask** — the trust
region is `|π(y_t|s_t) − µ(y_t|s_t)| ≤ δ`, constraining the absolute probability shift
on the sampled token. However, DPPO still uses a hard 0/1 mask: once a token crosses the
boundary, its gradient is zeroed entirely with no corrective signal.

DRPO replaces this hard mask with a **smooth, advantage-weighted quadratic regularizer**
that preserves the same Binary-TV trust-region geometry while providing continuous gradient
reweighting — attenuating diverging updates and providing corrective signals beyond the
boundary.

The reward here is **rule-based**: `MathBoxedRewardScorer`
([`unirl/reward/local/math_boxed.py`](../unirl/reward/local/math_boxed.py)) checks the
`\boxed{}` answer against the DAPO-Math ground truth — exactly the verifiable reward the
paper trains on, not a learned reward model.

![drpo overview: the prompt to SGLang rollout (behavior policy mu) to group advantage to replay for new logp pi to ratio r and TV shift |pi - mu| to reweighted REINFORCE pipeline (single update), with the centerpiece contrast between DPPO's hard-mask step (full gradient, then a cliff to 0 at the threshold delta) and the paper's smooth, bounded DRPO weight that ramps down and crosses zero into a corrective region past delta.](../assets/drpo_overview.png)

## The DRPO objective (paper §3, Eq. 8)

For each sampled token, the ratio and Binary-TV shift are:

$$ r_t = \frac{\pi_\theta(y_t|s_t)}{\mu(y_t|s_t)} = \exp(\log\pi_\theta - \log\mu), \qquad D^{\text{Bin-TV}}_t = |\pi_\theta(y_t|s_t) - \mu(y_t|s_t)| $$

The key insight (paper §3): the Binary-TV trust region `|π − µ| ≤ δ` is equivalent to a
**token-adaptive ratio bound** `|r_t − 1| ≤ δ / µ(y_t|s_t)`. Applying SPO's quadratic
construction (§2.3) with this token-adaptive bound yields the DRPO objective:

$$ L_\text{DRPO}(x,\pi) = \mathbb{E}_{y\sim\mu}\!\left[\sum_t r_t \hat A_t - \frac{|\hat A_t|}{2\delta}\,\mu(y_t|s_t)\,(r_t - 1)^2\right] $$

The first term is the importance-weighted policy gradient (TRPO surrogate, §2.1 Eq. 2).
The second is an advantage-weighted quadratic regularizer whose curvature is scaled by the
behavior probability `µ(y_t|s_t)` — this single factor changes the equilibrium from a fixed
ratio shift (SPO) to a fixed absolute probability shift (Binary-TV).

Taking the gradient (Appendix B) gives each token a continuous weight (Table 1):

$$ w_t = 1 - \mathrm{sign}\big(\hat A_t (r_t - 1)\big)\,\frac{|\pi_\theta(y_t|s_t) - \mu(y_t|s_t)|}{\delta} \;\in\; \Big[1 - \tfrac1\delta,\; 1 + \tfrac1\delta\Big] $$

**Diverging updates** (`sign(Â_t(r_t−1)) > 0`): the weight decays to 0 at the trust-region
boundary and becomes *corrective* (negative) beyond it, pulling the policy back.

**Converging updates** (`sign(Â_t(r_t−1)) < 0`): the weight is amplified above 1,
encouraging the policy to move back toward the behavior distribution.

Because the weight depends on an **absolute probability shift** (bounded in [0,1]) rather
than an importance ratio, it remains bounded even for low-probability tokens — unlike SPO,
whose weight grows without bound as `µ → 0` (paper §3.2, Figure 1).

## The code: `_drpo_loss`

The loss helper in [`drpo.py`](../unirl/algorithms/drpo.py) implements Eq. 8 directly.
The ratio `r_t` is kept differentiable (no `.detach()`, no TIS truncation), so the smooth
Table-1 gradient weight arises naturally via autograd:

```python
log_diff = torch.clamp(new_logp - old_logp, min=-20.0, max=20.0)
ratio = torch.exp(log_diff)                          # r_t = π/μ (differentiable)
old_prob = torch.exp(old_logp).detach()              # μ = rollout-policy token probability

# SPO quadratic (§2.3 Eq 5) with Binary-TV token-adaptive ε_t = ε / μ (§3)
ratio_delta = ratio - 1.0
quadratic_penalty = adv.abs() * old_prob * ratio_delta.square() / (2.0 * epsilon)
pg_losses = -adv * ratio + quadratic_penalty         # L_t = −Â·r + |Â|·μ·(r−1)²/(2ε)
```

The recipe sets `loss_agg_mode: seq-mean-token-sum-norm`: per-token losses are summed per
sequence, divided by `horizon`, then averaged over sequences.

## SPO vs. DRPO: why `µ` matters (paper §3.2)

SPO (§2.3) uses the same quadratic form **without** the `µ(y_t|s_t)` factor:

$$ L_\text{SPO} = \sum_t r_t \hat A_t - \frac{|\hat A_t|}{2\epsilon}\,(r_t - 1)^2 $$

This implicitly penalizes an advantage-weighted **χ² divergence** (`Σ (π−µ)²/µ`), which is
hypersensitive to low-probability tokens. DRPO's extra `µ` factor makes the penalty an
advantage-weighted **ℓ₂² distance** (`Σ (π−µ)²`), treating the same absolute probability
shift equally regardless of the token's behavior probability.

The practical consequence:
- SPO's gradient weight `1 − sign(Â(r−1))·|r−1|/ε` grows unbounded as `µ → 0`
  (the ratio `|r−1|` can be huge for rare tokens with small absolute probability shift).
- DRPO's gradient weight uses `|π−µ|/δ` instead of `|r−1|/ε`, which is bounded in [0,1]
  for all tokens. The result: bounded weights in `[1−1/δ, 1+1/δ]`.

## Math → code map

| Math object | Repo object |
|---|---|
| State `s_t = (x, y_{<t})` | `track.conditions` + the packed prefix in `TextSegment` |
| Sampled token `y_t` | `segment.tokens` |
| Behavior log-prob `log µ(y_t\|s_t)` | `segment.log_probs` (emitted by SGLang) → `old_logp` |
| New log-prob `log π_θ(y_t\|s_t)` | `stage.replay(segment, temperature=sampling_temperature)` → `new_logp` |
| Ratio `r_t = π/µ` | `torch.exp(new_logp − old_logp)` |
| Behavior prob `µ_t` | `torch.exp(old_logp).detach()` |
| Regularization threshold `δ` (paper) / `ε` (code) | `drpo_epsilon` (default 12.5; paper §4) |
| Sample-level advantage `Â` | `track.advantages` |
| Token-level advantage | `GRPO._expand_advantages_to_tokens(advantages, segment.lengths, ...)` |
| Quadratic penalty `\|Â\|·µ·(r−1)²/(2ε)` | `adv.abs() * old_prob * ratio_delta.square() / (2.0 * epsilon)` |
| Padding/eos mask | `segment.loss_mask` |

## From rollout to update

1. `unirl.train_ar` builds `ARTrainer` for the text-only Qwen3 recipe.
2. `SGLangLLMRolloutEngine` samples completions and returns an `"ar"` track with packed
   `TextSegment.tokens`, `log_probs`, `lengths`, and masks.
3. `MathBoxedRewardScorer` scores each completion correct/incorrect.
4. `RolloutTrack.compute_advantages(normalize=False, scope="group")` mean-centers rewards
   within each prompt group; the recipe sets `normalize_adv_by_std: false`, so there is **no
   std division**.
5. `TrainStack.train_track` calls `DRPO.compute_loss_and_backward`, which replays the
   sampled tokens at `temperature=sampling_temperature`, reads `old_logp = segment.log_probs`,
   expands advantages to tokens, calls `_drpo_loss`, applies `segment.loss_mask`,
   reduces, and `backward()`s.

Unlike FlowGRPO/FlowDPPO, `DRPO` does **not** freeze a train-side `old_logp` in
`prepare_segment` — it reuses the rollout log-prob, so the recipe must keep
`stack.num_updates_per_batch: 1` (`TrainStack` raises otherwise:
`supports_multi_update = False`).

## Key knobs ([`config.yaml`](config.yaml))

| Knob | Meaning |
|---|---|
| `drpo_epsilon` | Regularization threshold `ε` (code) / `δ` (paper). Default `12.5` (paper §4). Larger ⇒ weaker regularization; per-token trust region is `ε_t = ε / µ`. |
| `sampling_temperature` | **MUST equal `sampling.temperature`** (and the rollout engine's). Replay tempers logits so `π` and `µ` share a distribution (`ratio_mean ≈ 1`). |
| `loss_agg_mode` | `token-mean`, or the recipe's `seq-mean-token-sum-norm`. |
| `horizon` | Fixed normalizer for `seq-mean-token-sum-norm`; recipe `8192`. |
| `normalize_adv_by_std` | Recipe `false` → mean-center only (no std division). |

## Debug checklist

| Symptom | First files / variables to check |
|---|---|
| `ratio_mean` far from 1 at first update | `sampling_temperature` vs. `sampling.temperature` vs. rollout `temperature`; SGLang logprob config |
| Large `rollout_replay_logp_absdiff_mean` | train/rollout mismatch, weight sync, tokenization/chat-template mismatch |
| No gradient on many tokens | `segment.loss_mask`; zero advantages from all-correct/all-wrong groups |
| `drpo_penalty_mean` very large | `drpo_epsilon` too small for the model's off-policy gap |
| Completion misses the boxed answer | chat template `/no_think` + `enable_thinking: false` (Qwen3 overruns on `<think>`) |
| SGLang serves stale weights | `LocalLoraWeightSync`, adapter name `default`, rollout wake/sleep logs |

Metric source: `ratio_mean`, `ratio_max`, `approx_kl`, `drpo_penalty_mean`,
`clipfrac_upper`, `clipfrac_lower`, and the AR-only `rollout_replay_logp_absdiff_mean` are
emitted by `_drpo_loss`.

## Run it

```bash
# one-time: build the local jsonl from the raw DAPO-Math + AIME datasets
python -m unirl.utils.prepare_dapo_math --out-dir data/dapo_math

DATA_PATH=data/dapo_math/train.jsonl EVAL_DATA_PATH=data/dapo_math/aime_eval.jsonl \
python -m unirl.train_ar --config-name=ar/qwen3_drpo_4b_base_dpao_sglang num_devices=64
```

The model defaults to `Qwen/Qwen3-4B-Base`; set `QWEN3_PATH` to a local checkpoint dir to
avoid downloading at runtime. (Note the `train_ar` entrypoint — the AR loss is
modality-agnostic and shares the AR trainer.)

## Related tutorial

- **[FlowDPPO](../FlowDPPO/)** is the closest conceptual sibling: both replace ratio
  clipping with divergence-aware control. FlowDPPO has *exact* Gaussian KL over latent
  transitions; `DRPO` approximates token-distribution shift from chosen-token log-probs.

## References

- DRPO: Yao et al., *"Rethinking the Divergence Regularization in LLM RL"*: objective Eq. 8,
  gradient Eq. 9, weight Table 1, experiments §4.
- DPPO: Qi et al., *"Rethinking the Trust Region in LLM Reinforcement Learning"*
  [arXiv:2602.04879](https://arxiv.org/abs/2602.04879).
- SPO: Xie et al., *"Simple Policy Optimization"*
  [arXiv:2401.16025](https://arxiv.org/abs/2401.16025).
