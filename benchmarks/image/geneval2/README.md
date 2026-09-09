# image/geneval2

GenEval2 — a compositional text-to-image benchmark. Each of the 800 prompts
(`datasets/geneval2/synthetic/test.jsonl`) ships a `vqa_list` of atomic
`(question, expected_answer)` checks (object presence, count, color/attribute, position, verb).
Quality = **Soft-TIFA VQAScore**: a VLM (Qwen3-VL) answers each atom, and the per-atom answer
probabilities are aggregated (geometric mean) into a per-image score in `[0, 1]` (report ×100).

## Run

```bash
python -m benchmarks.run -b image/geneval2 --ckpt <base> [--lora <adapter>] --reward-url http://<host>:8080
```

The runner generates images, then scores them through the reward service's `geneval2` scorer,
sending each prompt's `vqa_list` as request metadata (a canary request guards against a
non-Soft-TIFA service). Use geometric-mean aggregation for the headline number.

## Scoring backends

- **Reward service (vLLM)** — the default (`--reward-url`); fast, for general benchmarking /
  monitoring. Reads top-k logprobs, so it approximates the full-vocab softmax.
- **Local transformers Qwen3-VL-8B** (`--local-geneval2`) — full-vocab softmax, geometric mean, no
  reward service. **Required to reproduce the DPPO paper numbers** (the vLLM top-k service gives
  close but non-identical scores).

## Note: DPPO GenEval2 reproduction

The `image/geneval2` spec pins the DPPO eval regime so `benchmarks.run` matches the recipes:
512×512, 40 steps, cfg 1.0, `max_sequence_length=256`, `linspace(1, 1/steps, steps)` flow-match
sigma grid, per-prompt-content seed, 1 sample/prompt. Score with the local transformers scorer
(`--local-geneval2`). (`max_sequence_length=256` and the linspace grid are load-bearing — the
diffusers pipeline defaults roughly halve the score.)

```bash
# base model, exact 800 unique prompts
python -m benchmarks.run -b image/geneval2 \
    --ckpt stabilityai/stable-diffusion-3.5-medium --local-geneval2
# DPPO single-reward LoRA, + the simulated 32-GPU x bs-32 config
python -m benchmarks.run -b image/geneval2 \
    --ckpt stabilityai/stable-diffusion-3.5-medium \
    --lora Tencent-Hunyuan-Multimodal-RL/SD3.5-GenEval2-Single-Reward \
    --local-geneval2 --sim-even-batches 32x32
```

Two eval configs:
- **exact 800** (default): mean over the 800 unique prompts.
- **simulated 32×32** (`--sim-even-batches 32x32`): the original distributed eval repeated the last
  partial wave of a 32-GPU × batch-32 loader, double-counting a fixed prefix of prompts; this flag
  additionally reports the `*_sim32x32` metric.
