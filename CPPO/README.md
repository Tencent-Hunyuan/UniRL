# CPPO — Cumulative Prefix-divergence Policy Optimization

Implementation of the paper "Beyond Uniform Token-Level Trust Region in LLM
Reinforcement Learning" ([arXiv:2606.10968](https://arxiv.org/abs/2606.10968)).

`CPPO` is an LLM RL algorithm. It replays the sampled tokens, compares train-side
log-probs against the rollout log-probs, expands sample-level advantages to tokens, and
applies a **position-weighted, cumulative-prefix-budget Binary-TV mask** on top of the
DPPO ratio-advantage surrogate (this repo ships the **Binary-TV** variant).

- **Loss:** [`unirl/algorithms/cppo.py`](../unirl/algorithms/cppo.py) (`CPPO`, `_cppo_loss`, `_cppo_mask`)
- **Recipe (SGLang):** [`examples/ar/qwen3_cppo_30b_a3b_base_dapo_sglang.yaml`](../examples/ar/qwen3_cppo_30b_a3b_base_dapo_sglang.yaml) — Qwen3-30B-A3B-Base on DAPO-Math
- **Config extract:** [`config.yaml`](config.yaml)

Lineage: **PPO → GRPO → DPPO → CPPO**. CPPO is the **hard-mask** sibling of DPPO Binary-TV;
[`DRPO`](../DRPO/) is the **smooth-mask** sibling of the same trust region.

![CPPO overview: the cumulative prefix constraint on a schematic token window. The flat blue line is the uniform token-level Binary-TV threshold δ; the orange curve is the prefix-adjusted threshold δ + δ_b·W_{t-1} − S_{t-1}, and the green curve is the effective threshold (their minimum). Both start at δ at t=0. While the accumulated weighted deviation stays below the per-token budget δ_b the token-level threshold is active (blue region) and a token is masked only if its own weighted deviation w_t·D_t exceeds δ (the blue bar); once the prefix over-spends the budget the effective threshold drops below δ (orange region), so a later token can satisfy the token-level threshold yet still be masked by the prefix constraint (the orange bar). Grey bars are the per-token weighted deviations.](../assets/cppo_overview.png)

## Why a position-weighted, cumulative trust region

LLM RL is off-policy: the rollout engine (SGLang) and the training engine differ, and one
batch of rollouts is split into several gradient steps, so the updated policy `π` drifts
from the behavior policy `µ` that sampled the tokens. DPPO replaces PPO's ratio clipping
with a **Binary-TV mask** — the trust region is `|π(y_t|s_t) − µ(y_t|s_t)| ≤ δ`, constraining
the absolute probability shift on the sampled token, which is better behaved than the ratio
for a long-tailed vocabulary.

But DPPO applies the **same threshold `δ` to every token**. CPPO's observation (paper §3-4)
is that not all positions deserve the same budget: errors compound along a sequence, and a
fixed per-token threshold lets the cumulative drift over a long response grow unchecked.
CPPO therefore makes the trust region **position-aware and cumulative**:

1. **Position weight** `w_t` decreasing from 1 (first token) to `w_min` (last token), so the
   *effective* per-token allowance shrinks as the response goes on.
2. **Cumulative prefix budget**: the threshold at token `t` tightens by how much the prefix
   `y_{<t}` has already spent, so a response that drifts early is held to a tighter bound later.

Only the **mask** changes — the loss term is DPPO's `−A_t · r_t` on kept tokens.

## The CPPO objective (paper Eq. 8-11)

For a response of length `T` (`t` is the 1-based token position):

$$ D_t = |\pi_\theta(y_t|s_t) - \mu(y_t|s_t)|, \qquad r_t = \frac{\pi_\theta(y_t|s_t)}{\mu(y_t|s_t)} $$

$$ w_t = w_{\min} + (1 - w_{\min})\frac{T - t}{T - 1} \;\in [w_{\min}, 1], \qquad Z_t = w_t\, D_t $$

With prefix sums `S_{t-1} = Σ_{j<t} Z_j` and `W_{t-1} = Σ_{j<t} w_j` (`S_0 = W_0 = 0`), the
**effective threshold** and **keep rule** are:

$$ c_t = \min\big(\delta,\; \delta + \delta_b^{\text{seq}} W_{t-1} - S_{t-1}\big) $$

$$ \text{keep token } t \iff A_t (r_t - 1) \le 0 \;\;\text{OR}\;\; Z_t \le c_t $$

The first clause **always keeps** updates that move `π` back toward `µ` (i.e. `A_t(r_t−1) ≤ 0`);
the budget only restricts updates that push `π` *farther* from `µ`. The loss is then:

$$ L = \mathbb{E}\!\left[\sum_t \text{keep}_t \cdot (-A_t\, r_t)\right] $$

**Dynamic prefix budget** (paper Eq. 22, Base-model warm-up calibration): each sequence sets
its own budget floor from its divergence statistics,

$$ \delta_b^{\text{seq}} = \mathrm{clamp}\big(\mathrm{P90}(D_{1:T}),\; \delta_b^{\min},\; 2\,\delta_b^{\min}\big) $$

where `δ_b^min` is the config `cppo_delta_b`.

The reward here is **rule-based**: `MathVerifyRewardScorer`
([`unirl/reward/local/mathverify.py`](../unirl/reward/local/mathverify.py)) checks the parsed
answer against the DAPO-Math ground truth — the verifiable reward the paper trains on.

## The code: `_cppo_mask` / `_cppo_loss`

`_cppo_mask` builds the keep-mask under `torch.no_grad` (it is a trust-region *gate*, not part
of the differentiable loss). UniRL packs an AR batch as a single varlen `[total_tokens]` tensor,
so the mask is computed **per sequence** via `torch.split(..., segment.lengths)` — the prefix
sums must not bleed across packed boundaries, and the position weight keys on each sequence's
own length `T` (there is no padding, unlike a 2D right-padded layout):

```python
pos = torch.arange(1, T + 1)
w_t = w_min + (1.0 - w_min) * (T - pos) / max(T - 1, 1)   # decreasing position weight
Z_t = w_t * D_t
S_prev = torch.cat([Z_t.new_zeros(1), torch.cumsum(Z_t, 0)[:-1]])   # right-shifted prefix sums
W_prev = torch.cat([w_t.new_zeros(1), torch.cumsum(w_t, 0)[:-1]])
delta_b_seq = torch.quantile(D_t, 0.9).clamp(min=delta_b, max=2.0 * delta_b)   # Eq. 22
c_t = torch.minimum(torch.full_like(Z_t, delta), delta + delta_b_seq * W_prev - S_prev)   # Eq. 8
keep = (adv * (ratio - 1.0) <= 0.0) | (Z_t <= c_t)   # Eq. 10
```

`_cppo_loss` then forms `−A_t · r_t · keep` with `r_t` kept differentiable (no `.detach()`),
matching `GRPO` / `DRPO`.

## Math → code map

| Math object | Repo object |
|---|---|
| State `s_t = (x, y_{<t})` | `track.conditions` + the packed prefix in `TextSegment` |
| Sampled token `y_t` | `segment.tokens` |
| Behavior log-prob `log µ(y_t\|s_t)` | `segment.log_probs` (emitted by SGLang) → `old_logp` |
| New log-prob `log π_θ(y_t\|s_t)` | `stage.replay(segment, temperature=sampling_temperature)` → `new_logp` |
| Binary-TV divergence `D_t` | `(exp(new_logp) − exp(old_logp)).abs()` |
| Position weight `w_t` | `w_min + (1-w_min)*(T-pos)/max(T-1,1)`, per sequence |
| Effective threshold `c_t` (Eq. 8) | `torch.minimum(delta, delta + delta_b_seq*W_prev - S_prev)` |
| Token-level threshold `δ` | `cppo_delta` (0.20 for 30B-A3B; paper Table 3) |
| Position-weight floor `w_min` | `cppo_w_min` (0.8) |
| Prefix-budget floor `δ_b^min` | `cppo_delta_b` (0.02) |
| Sample-level advantage `Â` | `track.advantages` |
| Token-level advantage | `GRPO._expand_advantages_to_tokens(advantages, segment.lengths, ...)` |
| Padding/eos mask | `segment.loss_mask` |

## From rollout to update

1. `unirl.train_ar` builds `ARTrainer` for the text-only Qwen3 recipe.
2. `SGLangRolloutEngine` samples completions and returns an `"ar"` track with packed
   `TextSegment.tokens`, `log_probs`, `lengths`, and masks.
3. `MathVerifyRewardScorer` scores each completion correct/incorrect.
4. `RolloutTrack.compute_advantages(normalize=False, scope="group")` mean-centers rewards
   within each prompt group; the recipe sets `normalize_adv_by_std: false`, so there is **no
   std division**.
5. `TrainStack.train_track` calls `CPPO.compute_loss_and_backward`, which replays the sampled
   tokens at `temperature=sampling_temperature`, reads `old_logp = segment.log_probs`, expands
   advantages to tokens, builds the CPPO mask, applies `segment.loss_mask`, reduces, and
   `backward()`s.

Like `GRPO` / `DRPO`, `CPPO` reuses the rollout log-prob as `old_logp` (the behavior policy
`µ`) — it does **not** freeze a train-side anchor by default, so `old_logp_source: rollout` is
the canonical mode. (`old_logp_source: replay` is available for ablations.)

## Key knobs ([`config.yaml`](config.yaml))

| Knob | Meaning |
|---|---|
| `cppo_delta` | Token-level Binary-TV threshold `δ`. Paper Table 3: 0.15 dense, 0.20 for 30B-A3B. |
| `cppo_w_min` | Position-weight floor `w_min` (0.8). Earlier tokens get weight 1, late tokens `w_min`. |
| `cppo_delta_b` | Dynamic prefix-budget floor `δ_b^min` (0.02); `δ_b^seq = clamp(P90(D), δ_b, 2·δ_b)`. |
| `sampling_temperature` | **MUST equal `sampling.temperature`** (and the rollout engine's). |
| `loss_agg_mode` | `token-mean`, or the recipe's `seq-mean-token-sum-norm`. |
| `horizon` | Fixed normalizer for `seq-mean-token-sum-norm`; recipe `16384`. |
| `old_logp_source` | `rollout` (canonical: `µ` = the SGLang sampler's logp) or `replay` (ablation). |

Metric source: `ratio_mean`, `ratio_max`, `approx_kl` (k3), `masked_fraction` (the budget-mask
share — paper Fig. 7), and the AR-only `rollout_replay_logp_absdiff_mean` are emitted by
`_cppo_loss` / `compute_loss_and_backward`.

## Run it

```bash
# one-time: build the local jsonl from the raw DAPO-Math + AIME datasets
python -m unirl.utils.prepare_dapo_math --out-dir data/dapo_math

DATA_PATH=data/dapo_math/train.jsonl EVAL_DATA_PATH=data/dapo_math/aime_eval.jsonl \
python -m unirl.train_ar --config-name=ar/qwen3_cppo_30b_a3b_base_dapo_sglang num_devices=128
```

The model defaults to `Qwen/Qwen3-30B-A3B-Base`; set `QWEN3_PATH` to a local checkpoint dir to
avoid downloading at runtime. The MoE + cluster knobs in the recipe (`num_devices`, `tp_size`,
`mem_fraction_static`) are starting points to tune for your hardware.

## Related tutorial

- **[DRPO](../DRPO/)** is the closest sibling: same DPPO Binary-TV trust region, but DRPO
  *smooths* the hard mask into an advantage-weighted quadratic regularizer, whereas CPPO keeps
  a hard keep/reject mask and instead makes the threshold position-weighted and cumulative.

## References

- CPPO: *"Beyond Uniform Token-Level Trust Region in LLM Reinforcement Learning"*
  [arXiv:2606.10968](https://arxiv.org/abs/2606.10968): mask Eq. 8-11, Algorithm 1, dynamic
  `δ_b` Eq. 22, experiments §4.
- DPPO: Qi et al., *"Rethinking the Trust Region in LLM Reinforcement Learning"*
  [arXiv:2602.04879](https://arxiv.org/abs/2602.04879).
