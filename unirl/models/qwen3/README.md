# Qwen3 AR replay

> **Where it fits:** model replay for AR training. In: prompt conditions and a
> packed response segment. Out: response log-probabilities and optional statistics.
> Full map: [`../README.md`](../README.md).

## Gotchas

- **`return_entropy=True` returns `ReplayResult` with detached fp32 entropy.**
  Each response position uses the full vocabulary conditional distribution
  `softmax(logits / T)` and natural logarithms (nats), before top-k/top-p filtering.
  `T` is replay's `temperature` when positive, otherwise 1, matching its log-probs;
  greedy rollout therefore reports model entropy at T=1, not zero sampling entropy.
  Entropy is computed within the existing checkpointed LM-head chunks; only per-token
  entropy leaves replay, and `ReplayResult.logits` remains unset.
- **Packed and padded replay return entropy in `segment.tokens` order.**
  Prompt and padding positions are excluded; `loss_mask` is applied by the algorithm.
  `return_values=True` can be combined with entropy. The default returns the existing
  log-prob tensor (or critic result) without softmax/entropy work.
