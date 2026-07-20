# Benchmarks

Standalone checkpoint evaluation, decoupled from the training stack. A checkpoint is an
artifact — an HF repo id / local dir, optionally plus a LoRA adapter or a raw UniRL
checkpoint — and each benchmark scores it under a fixed, documented protocol, so one
checkpoint can be run across many benchmarks and the numbers compare across runs and
to published tables. (The trainer's in-loop `eval/*` is a training-time signal and does
**not** follow these protocols — see issue #182.)

## Quickstart

```bash
python -m benchmarks.run --list                # what exists

# t2i: pretrained, or base + LoRA (a raw UniRL checkpoint dir is auto-exported to PEFT)
python -m benchmarks.run -b image/geneval2 -b image/preference \
  --ckpt stabilityai/stable-diffusion-3.5-medium --lora <ckpt-or-adapter-dir> \
  --reward-url http://<reward-host>:8080

# text: serve the checkpoint first (sglang serve / vllm serve), then
python -m benchmarks.run -b text/aime24,text/math500 --endpoint http://127.0.0.1:30000

python -m benchmarks.run --report              # all summaries -> markdown tables
```

Smoke-test any benchmark with `--num-prompts 8` (summaries get flagged as subsets);
`--dry-run` prints the plan. Multi-GPU generation: run one process per GPU with
`--shard i/n`. Stages can run on different machines: `--stage generate` on the GPU box,
`--stage score` wherever the reward service / CPU is.

## Benchmarks

| name | prompts (license) | protocol | scored by |
|---|---|---|---|
| `image/geneval2` | 800, in-repo `datasets/geneval2` | 4 img/prompt | reward service `geneval2` (VQAScore soft-TIFA) |
| `image/geneval` | 553, vendored (MIT) | 4 img/prompt, official [GenEval](https://github.com/djghosh13/geneval) | reward service `geneval` (Mask2Former+CLIP, off by default) |
| `image/dpg_bench` | 1065, `fetch.sh` (Apache-2.0) | 4 img/prompt | external: official [ELLA](https://github.com/TencentQQGYLab/ELLA) mPLUG script |
| `image/preference` | 1632 PartiPrompts, vendored (Apache-2.0) | 1 img/prompt | reward service `hpsv3`, `pickscore`, `imagereward` |
| `text/aime24` / `text/aime25` | 30+30 (MIT / fetch script) | avg@16, temp 0.6 | local `math-verify` |
| `text/math500` | 500, vendored (MIT) | avg@4, temp 0.6 | local `math-verify` |
| `text/gpqa` | 198 Diamond, gated — `fetch.py` | avg@4, MC letter | local letter match |

Video (VBench) is generated and scored with the official toolkit — see
[`video/vbench/`](video/vbench/). Framework speed comparisons live in
[`speed_benchmarks/`](speed_benchmarks/).

## Conventions

- Results: `benchmarks_results/<ckpt-tag>/<benchmark>/{images/ | completions.jsonl, scores.jsonl, summary.json}`.
- Images are `p{prompt:05d}_s{k}.png` with seed `--seed + 1000*prompt + k` — the naming
  is the only state shared between stages.
- Generation uses each pipeline's own defaults (steps/guidance/resolution) unless
  overridden — record any override next to reported numbers. Distilled/turbo
  checkpoints (e.g. Z-Image-Turbo) ship base-schedule pipeline defaults: pass the
  model card's `--steps`/`--guidance` or you benchmark the wrong sampler setting.
- Data files are vendored when small and redistributable; otherwise the benchmark folder
  ships a fetch script. Never commit gated data (`text/gpqa`).
- Code lives in `core/` + `run.py` only; benchmark folders hold README + data. Adding a
  benchmark = data (or fetch script) + one `BenchmarkSpec` in `core/registry.py`.

Roadmap (not in this drop): VL understanding (Geo3K test / MathVista), unified-model
und-side, UniGenBench++, image-editing benchmarks.

Extending: the staging/shard/resume/report machinery is modality-agnostic — a new
modality (audio, editing, interleaved) is one driver in `core/generate.py` plus
scorers. When the first benchmark lands whose loading/eval cannot be expressed as
data + a spec, specs grow optional `loader`/`runner`/`scorer` dotpath hooks so that
code lives in the benchmark's own folder (the lm-eval-harness / VLMEvalKit escape
hatch), while core and the results contract (`summary.json`, tags, `--report`) stay
fixed and global. Environment-style benchmarks (agent suites) are wrapped, never
absorbed: the official harness runs the benchmark; we only normalize its results
into the report.
