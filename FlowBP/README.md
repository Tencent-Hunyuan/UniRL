# FlowBP - Reward Backpropagation for Flow Matching

`FlowBP` is the reward-gradient alignment framework from the paper
*"Exploring the Design Space of Reward Backpropagation for Flow Matching."*
It keeps the sampling rollout under `no_grad`, caches the trajectory, and then
builds a lightweight surrogate backward graph from cached and selectively
re-forwarded velocities.

The paper names are used throughout this document.

- **Implementation:** TODO
- **Paper:** *"Exploring the Design Space of Reward Backpropagation for Flow Matching."*

## What problem it solves

Direct reward backpropagation is sample-efficient: instead of estimating a
policy-gradient objective, it differentiates a scalar reward through the
flow-matching sampler. The naive version is not practical at modern scale:

- storing activations for every sampling step is too expensive;
- chained Jacobian products over many Euler steps can inflate gradients;
- connector methods such as LeapAlign avoid full backpropagation, but long
  single-velocity jumps can create large connector residuals, so the reward
  gradient is routed through an inaccurate surrogate endpoint.

FlowBP treats the backward trajectory itself as the design object. The forward
sample is produced by the normal rollout and cached. The reward still sees the
sampled endpoint `x_0`, but the gradient flows only through a sparse surrogate
graph whose active velocity calls are chosen by the method.

## The FlowBP design axes

The paper separates reward backpropagation into four choices:

| Axis | Meaning in FlowBP |
|---|---|
| Reward-model input | Whether the reward sees the actual rollout image `x_0` or a posterior-mean estimate |
| Active set | Which cached steps are re-forwarded with gradients |
| Integration weights | Euler weights `h_i` or Lagrange quadrature weights `w_i^L` |
| Bridge coupling | Whether a split latent `x_j` lets post-segment gradients affect pre-segment active calls |

This recovers prior methods as settings of the same framework: ReFL,
DRaFT-LV, and DRTune use detached short/sparse posterior-mean reward targets;
LeapAlign uses two connector-pinned single-velocity hops; FlowBP explores
endpoint-faithful sparse reconstruction, controlled bridge coupling, and
higher-order leap quadrature.

## Core surrogate

The cached rollout follows the flow Euler update:

$$
x_{i+1} = x_i + (\sigma_{i+1} - \sigma_i) v_\theta(x_i, \sigma_i, c).
$$

FlowBP caches this rollout without storing the full backward graph. The
surrogate then replaces only selected velocities with gradient-bearing
evaluations while treating the rest of the cached trajectory as fixed. This
keeps the forward sample tied to the actual rollout and bounds memory by the
active set size rather than by the full sampling length.

## FlowBP-Sparse

`FlowBP-Sparse` reconstructs the full endpoint with Euler composition while
only exposing `K` selected velocities to autograd.

Mathematically, it uses no bridge (`j = 0`) and Euler weights:

$$
x_0 = x_N - \sum_{i=1}^{N} h_i \tilde v_i,
\qquad
\frac{\partial x_0}{\partial \theta}
= -\sum_{i \in A} h_i
\frac{\partial v_\theta(x_i, \sigma_i, c)}{\partial \theta}.
$$

## FlowBP-Bridge

`FlowBP-Bridge` keeps the Euler endpoint reconstruction of `FlowBP-Sparse`,
but splits the trajectory at `j` and lets the post-segment anchor consume an
`alpha`-mixed `x_j`.

The gradient has the sparse direct terms plus one controlled nested term:

$$
\frac{\partial x_0}{\partial \theta}
= -\sum_{i \in A} h_i
\frac{\partial v_\theta(x_i, \sigma_i, c)}{\partial \theta}
+ \alpha h_j
\frac{\partial v_\theta(x_j, \sigma_j, c)}{\partial x_j}
\sum_{i \in A_\text{pre}} h_i
\frac{\partial v_\theta(x_i, \sigma_i, c)}{\partial \theta}.
$$

## FlowBP-Lagrange

`FlowBP-Lagrange` stays closest to LeapAlign's two connector topology, but
replaces each single-velocity leap with integrated Lagrange quadrature over a
small support set.

For a segment from `s` to `t`, it selects support velocities `S` and computes
integral weights:

$$
w_i^L = \int_{\sigma_s}^{\sigma_t}
\prod_{q \in S, q \ne i}
\frac{\sigma - \sigma_q}{\sigma_i - \sigma_q}\,d\sigma.
$$

The connector prediction is:

$$
\hat x_t = x_s + \sum_{i \in S} w_i^L \tilde v_i,
\qquad
x_t = \hat x_t + \mathrm{sg}(x_t - \hat x_t).
$$

Compared with LeapAlign, this reduces connector residuals `d_j` and `d_0`
over long intervals.

## References

- FlowBP: *"Exploring the Design Space of Reward Backpropagation for Flow Matching."*
- LeapAlign: connector-based two-step reward backpropagation for flow matching.
- ReFL, DRaFT-LV, and DRTune: prior direct reward-gradient baselines recovered
  as settings of the FlowBP surrogate design space.
