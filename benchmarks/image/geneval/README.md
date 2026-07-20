# image/geneval

Official [GenEval](https://github.com/djghosh13/geneval) (553 prompts, MIT, vendored in
`data/`): single/two objects, counting, colors, position, color attribution. Headline
score = unweighted mean of the six per-tag accuracies.

Protocol: 4 images/prompt. Scoring is the official pipeline — Mask2Former (Swin-S)
detection + CLIP color check — available as the reward service `geneval` scorer, which
is **disabled by default** (needs a Python 3.10 env; see
`unirl-reward-service/README.md`). Enable it there, then:

```bash
python -m benchmarks.run -b image/geneval --ckpt <base> [--lora <ckpt>] --reward-url http://<host>:8080
```

Alternatively generate with `--stage generate` and score with the upstream
`evaluation/evaluate_images.py` (it expects its own folder layout; our flat
`p{idx:05d}_s{k}.png` naming maps 1:1 onto prompt index / sample index).
