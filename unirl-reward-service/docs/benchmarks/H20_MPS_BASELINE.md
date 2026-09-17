# H20 reward-service MPS baseline

This is the benchmark gate for the PR 1 slice of
[#463](https://github.com/Tencent-Hunyuan/UniRL/issues/463). It qualifies MPS
for this workload at 100% active-thread capacity; it is not a general MPS
performance claim.

## Setup

- NVIDIA H20 96 GB, driver 535.247.01
- Python 3.12.11, Ray 2.46.0, Torch 2.7.1
- Transformers 4.56.1.dev0
- float16 CLIP and PickScore actors using local PickScore_v1 weights
- batch sizes 1/4/8 and concurrency 1/4/16
- 60 requests per point, three repetitions
- 1,620 requests and 7,020 requested items per mode
- score-parity tolerance: `1e-3`

The four modes were dedicated GPUs, fractional placement without MPS, the
same fractional placement with MPS, and a consolidated one-process control.
The checked-in
[`h20-dedicated.yaml`](h20-dedicated.yaml) and
[`h20-shared.yaml`](h20-shared.yaml) contain the exact actor placement.

## Result

- Dedicated (two GPUs): 1,620/1,620 requests valid.
- Fractional no-MPS (one GPU): 1,620/1,620 valid and 27/27 parity checks
  passed; mean rewards/GPU-hour improved 34.2% over dedicated.
- MPS at 100% (one GPU): 1,620/1,620 valid and 27/27 parity checks passed;
  mean raw throughput improved 16.5% over fractional no-MPS and mean
  rewards/GPU-hour improved 54.7% over dedicated.
- Consolidated (one GPU): 1,620/1,620 valid and 27/27 parity checks passed;
  mean rewards/GPU-hour improved 127.6% over dedicated.
- All 108 run-level parity outcomes passed. Job-scoped MPS logs contained no
  explicit FAULT, OOM, or restart event.

## Active-thread limit

`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` corrupted float16 CLIP output on this
stack. A direct-process probe produced:

- CLIP float16 at 50%: 0/20 finite scores
- PickScore float16 at 50%: 20/20 finite scores
- concurrent CLIP/PickScore float16 at 100%: 200/200 finite scores
- concurrent CLIP/PickScore float32 at 50%: 200/200 finite scores

The failure therefore does not require Ray or multiple clients. Use 100% for
this qualified recipe; lower limits require scorer/dtype-specific validation.

## Reproduce

Replace `/path/to/PickScore_v1` in the two YAML files, launch each deployment,
and run `scripts/bench_concurrent.py` with:

```bash
--sweep 1 4 16 --batch-sweep 1 4 8 --total 60 --repetitions 3 \
--rewards clip,pickscore --gpu-sample-interval 0.5
```

Use identical workload arguments and the appropriate `--deployment-mode` for
all four topologies. Generated run data is intentionally not committed.
