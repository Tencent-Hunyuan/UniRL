# speed_benchmarks

How to measure UniRL training throughput and compare it fairly against another RL
framework. One subfolder per rival framework with pinned same-setting pairs
(currently [`verl_omni/`](verl_omni/) — a submodule-pinned upstream, a runnable aligned
SD3.5+FlowGRPO pair, and first measured rows).

## Measuring UniRL

Every trainer logs per step to wandb: `perf/step_time_s` (end-to-end wall clock;
`perf/rollout_time_s` on older runs), `perf/<phase>_time_s` (generate / reward / train /
weight-sync attribution), `perf/max_memory_*`. Without wandb access, the console log is
enough — one `rollout N/M reward=…` line per step:

```bash
python benchmarks/speed_benchmarks/parse_perf.py train.log --samples-per-step <batch×group> --gpus 8
```

prints steps, median/mean/p90 s/step, samples/s and samples/GPU-hour (first `--skip 2`
steps dropped as warmup).

## Smoke schema check

```bash
python -m benchmarks.speed_benchmarks.benchmark_smoke
```

writes one schema-v1 JSON result to `outputs/benchmark/` from a CPU-only synthetic loop;
it checks the result contract and is not a UniRL performance measurement (`--help`
for options; `UNIRL_RUN_ID` overrides the generated run id).
Rollout, reward, and train producers build a `BenchmarkResult` from
`unirl.utils.benchmark_result` with the total samples and wall-clock of their measured
window, and call `write_result`; throughput is derived, and unavailable memory, phases,
or torch version are omitted. Runtime adapters belong beside their owning `unirl`
components, offline parsers in `benchmarks/speed_benchmarks/`.
See [#414](https://github.com/Tencent-Hunyuan/UniRL/issues/414) and [#25](https://github.com/Tencent-Hunyuan/UniRL/issues/25).

## Fair-comparison protocol

Pin on both sides, and publish the full configs next to any number you report:

1. **Model + algorithm** — identical checkpoint and RL algorithm.
2. **Workload** — same prompts/step, group size, resolution, denoise steps (t2i) or
   max tokens (LLM), same number of optimizer updates per step.
3. **Reward** — same scorer behind the same HTTP service, same placement (reward GPUs
   counted on both sides, or excluded on both sides).
4. **Hardware** — same GPU count/type/interconnect, one framework at a time.
5. **Steady state** — ≥20 steps, drop warmup, report median s/step and samples/GPU-hour.
6. Prefer two rows per pair: *aligned backends* (e.g. vLLM-Omni rollout on both sides)
   and *each framework's best config* — they answer different questions.

A time-to-reward-threshold curve (same eval, wall-clock x-axis) is the strongest
end-to-end evidence; report it when training long enough.
