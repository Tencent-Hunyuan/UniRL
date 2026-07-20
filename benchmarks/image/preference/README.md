# image/preference

Human-preference scores over [PartiPrompts](https://github.com/google-research/parti)
(P2: 1632 prompts across 12 categories, Apache-2.0, vendored in `data/`).

Protocol: 1 image/prompt; scored by the reward service `hpsv3`, `pickscore` and
`imagereward` scorers in one pass. These are relative metrics — meaningful for
before/after-RL comparisons of the same base, not across papers.

```bash
python -m benchmarks.run -b image/preference --ckpt <base> [--lora <ckpt>] --reward-url http://<host>:8080
```
