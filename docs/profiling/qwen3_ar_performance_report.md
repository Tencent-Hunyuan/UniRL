# Qwen3-4B-Base AR performance analysis

## Scope and status

- Model: `Qwen/Qwen3-4B-Base`, revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Pipeline: `Qwen3Pipeline` / `Qwen3Bundle` / `Qwen3DecoderLayer`.
- Training: DPPO, SGLang native rollout, DAPO-Math, thinking enabled, 8192-token response limit.
- Precision: fp32 masters and log-probabilities; bf16 compute and FSDP communication.
- Worktree: `/apdcephfs/private_aimicahchen/Project/UniRL-qwen3-ar-nsys-opt`.
- Branch: `perf/qwen3-ar-optimization`, created from `upstream/main` at `ca384514`; zero commits behind.
- Hardware validated here: one 8× NVIDIA H20 96GB node. No 32-GPU allocation was available, so no 32-GPU result is claimed or extrapolated.
- The GRPO, DRPO, and async-GRPO companion recipes were inspected for shared Qwen3/SGLang paths only. They were not changed or benchmarked; all performance numbers below are for the main DPPO recipe.
- Retention result: no speed candidate passed every precision, confidence-interval, and end-to-end-impact gate. The remaining production changes are correctness fixes and opt-in profiling/reproducibility support.
- No commit, push, or pull request was created.

## Reproducibility

The run used an isolated Python 3.12 environment with torch `2.11.0+cu130`, transformers `5.6.0`, SGLang `0.5.12.post1`, Ray `2.56.0`, and Nsight Systems `2025.1.3`.

The H20 node has a 535 driver. Its `/usr/local/cuda/compat` package still reports CUDA 12.9 and cannot run torch cu130. All measured jobs therefore prepend the verified forward-compat package:

`/apdcephfs/private_aimicahchen/.local/cuda-compat-13.0/usr/local/cuda-13.0/compat`

Model files total 8,056,519,973 bytes and are hashed individually in `outputs/profiling/qwen3_ar/assets/model_manifest.json`.

Pinned data:

- DAPO-Math revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`: 1,791,700 rows, SHA256 `5e7ea0b588598eaf39a6cf989d7197214dc907e5c95c465255017fabdd91ae2d`.
- AIME 2024 revision `8d88b2876a82a080e2f172cc9b25d0d9d2cb4792`.
- AIME 2025 revision `6f71d77b0b89b9dabe07ab466c51df33f514df7f`.
- Combined AIME file: 60 rows, SHA256 `d259db11133a4ab7fac53c4b1ccf9660bfa911bc432c327ca4181e34c3271742`.

The official DAPO source is not 1.79M unique problems. The audit found 17,405 unique prompt/answer pairs, 17,398 unique prompts, and seven prompts with conflicting answers. Pair multiplicities are 100× for 16,917 pairs, 200× for 468, 300× for 16, and 400× for four. The workload preserves the official pinned ordering and duplicates; it does not silently substitute a cleaned dataset.

## Determinism and precision

SGLang `sampling_seed` is ignored by SGLang 0.5.12 unless `enable_deterministic_inference=true`. Before enabling it, two equal-seed runs diverged at token zero for every sample. The corrected recipe enables the server switch and assigns a stable distinct seed to each expanded sibling.

Post-review hardening replaced worker-local ordinal seed derivation with a stable `(base_seed, sample_id)` mapping, added full-request identity validation before DP sharding, and shared the injection path with the VLM adapter. The retained fixture fingerprints below predate that mapping change: they remain evidence that SGLang deterministic mode is bit-identical across fresh engines, but they are not golden outputs for the current seed mapping and must be regenerated before an exact artifact-reproduction claim.

After the correction, two fresh SGLang engines produced bit-identical fixtures:

- `input_ids`: `torch.equal`, max absolute difference 0.
- `attention_mask`: `torch.equal`, max absolute difference 0.
- `position_ids`: `torch.equal`, max absolute difference 0.
- Generated token IDs: `torch.equal`, max absolute difference 0.
- Response lengths: `torch.equal`, max absolute difference 0.
- Rollout log-probabilities: `torch.equal`, max absolute difference 0.
- Eight responses contained 13,331 generated tokens with lengths 3436, 1037, 2313, 2271, 2123, 739, 890, and 522.

Teacher-forced replay of that fixed fixture was repeated 12 times in one model instance. The ten retained rounds were bit-identical (`torch.equal`, max absolute difference 0), with 3.9996 s mean, 3.9993 s median, 0.00120 s standard deviation, 0.0301% CV, and 3,333.1 tokens/s.

The SGLang-versus-HF replay gap was measured rather than hidden: mean absolute log-probability difference 0.00898, maximum 0.22710, and mean ratio 0.99930. The fixed fixture happened to score all-zero reward groups, so its mean-centered advantages and DPPO loss were zero. The real 8-GPU smoke still exercised backward and optimizer with reward 0.0625 and grad norm 0.0256.

Flash Attention backward reports that its selected kernel is nondeterministic. Consequently, independently restarted full-RL runs diverge after the first optimizer update even when rollout seeds are fixed. Full-RL baseline and optimized validation are therefore reported independently, not used as a paired speedup claim. All retained performance claims below use fixed-token, alternating, same-process/same-instance A/B.

A full 47 GiB checkpoint smoke also audited `trainer_state.json`. It exposed a separate correctness bug: with W&B disabled, the null logger returned before advancing its optimizer-step counter, so the checkpoint recorded 0 after four real updates. The logger now advances the same train-axis count whether reporting is enabled or disabled; focused tests verify a four-update DPPO result records step 4 and multitrack counting matches the live logging path. The pre-fix checkpoint and `trainer_state_audit.json` are retained as evidence and must not be used to test resume semantics.

## Single-GPU microbaseline

The strict design used 12 rounds, discarded the first two, and retained ten steady samples.

- Prefill: 0.04393 s mean, 4,462.6 tokens/s, 22.78 GiB peak allocated memory.
- Decode: eight fixed tokens in 0.32508 s mean, 24.64 tokens/s, 22.78 GiB peak.
- KV cache after eight decode tokens: 60,162,048 bytes (0.0560 GiB).
- Padded log-probability replay: 0.19353 s mean, 2,645.5 tokens/s, bit-identical repeats.
- DPPO loss math: 0.000248 s mean.
- Backward over 32 fixed response tokens: 0.22628 s mean, 0.20620 s median, 0.06193 s standard deviation, 27.37% CV, 33.19 GiB peak. One retained round was a large outlier and is preserved.
- AdamW optimizer step: 0.125546 s mean, 0.125541 s median, 0.000100 s standard deviation, 0.0795% CV, 75.57 GiB peak.

## Strict optimization decisions

### Rejected: duplicate prompt tokenization elimination

`ARTrainer` expands each prompt into adjacent GRPO siblings. The candidate rendered and tokenized each unique prompt once per request, then copied the exact token list into each sibling payload.

- Same process, tokenizer, adapter, and loaded model instance.
- Alternating A/B and B/A order over 12 rounds; first two discarded.
- Baseline mean 0.0020349 s; candidate mean 0.0005318 s.
- Paired reduction mean 73.8635%.
- Paired 95% Student-t CI `[73.6746%, 74.0525%]`; lower bound is strictly positive.
- All payloads and prompt token IDs are exactly equal.

The canonical source is `microbaseline_full/analysis.json`. The absolute saving is 0.00150 s, only about 0.0023% of the measured 64.14 s full step. The separate 12-rollout candidate validation is not a paired comparison because post-update token streams diverged, so it provides no valid end-to-end confidence interval. The candidate therefore fails the explicit retention gate requiring a meaningful tokens/s or complete-step improvement. It was rolled back from the production adapter and target recipe; the isolated benchmark evidence remains for auditability.

### Rejected: flex-attention packed replay

Packed replay was tested against padded replay in the same flex-attention model instance with the same response IDs.

- Padded mean 0.18644 s; packed mean 0.18114 s.
- Paired reduction 2.8419%, 95% CI `[2.4357%, 3.2481%]`.
- Maximum log-probability difference 0.12263; `torch.equal` is false.

The candidate is statistically faster but fails the required `max_abs_diff=0` precision gate, so it is rejected. The main DPPO recipe remains on its original SDPA padded-replay path.

### Required correctness mode: deterministic SGLang sampling

Enabling deterministic inference is not counted as a speed optimization. SGLang changes its sampling backend from FlashInfer to PyTorch and disables piecewise CUDA graphs. In observed long-tail single-request decode logs, throughput fell from roughly 209 tokens/s in the unseeded path to roughly 90 tokens/s in deterministic mode. The production recipe keeps this switch off and leaves `sampling.seed` unset; only the profiling launcher enables deterministic inference while overriding both data and sampling seeds.

### Candidate screening boundary

- SGLang already uses native continuous batching and CUDA Graphs. The reduced 8-GPU workload has four requests per rank, so max-running-request, graph-batch, KV-block, and chunked-prefill changes cannot support a 32-GPU claim. Sampling controls and sync frequency are fixed semantics, not tuning knobs.
- Variable-length `balance_shards` and `TokenBudgetPlanner(token_budget=10240)` are part of the baseline. No additional reorder was accepted because sample order and loss weighting must remain unchanged.
- SDPA and activation checkpointing are the baseline. Flex packed replay is the measured transformer/replay candidate and failed exactness; compile and fused-kernel substitutions are not reported as tested or bit-exact.
- Rollout old-logp is already cached on the segment, so the main path does not perform a duplicate old-policy forward.
- FSDP resharding, prefetch, deferred grad sync, fp32 masters, and bf16 compute/communication were held fixed. No unmeasured collective-order change is presented as an optimization.
- DPPO Binary-TV, delta, mean-centering, horizon, and reduction are unchanged. The scalar loss math is only 0.000248 s and is not a meaningful optimization target.
- Dense 64 MiB-bucket weight sync costs 0.598 s (0.93% of the step). Lowering its frequency would change on-policy semantics and was not attempted.

## Eight-GPU full DPPO baseline

The profiling workload uses eight prompts × four samples = 32 trajectories, preserving four optimizer updates and every requested algorithm/precision setting. It is a reduced 8-GPU workload, not the 32-GPU reference recipe.

Ten steady rollouts after discarding two:

- Step time: 64.1422 s mean, 68.3028 s median, 33.5418 s standard deviation, 52.29% CV.
- Generated tokens: 29,487.9 mean per rollout, 38.63% CV.
- Throughput: 562.50 tokens/s mean, 496.34 median; 0.7113 samples/s mean.
- Time per token: 2.179 ms mean.
- External peak memory: 52.40 GiB.
- Rollout generation: 58.8356 s mean (91.73% of step mean).
- Training: 4.5949 s mean (7.16%).
- Dense weight sync: 0.5978 s mean (0.93%).
- Reward: 0.0842 s mean.

Variable response lengths are the dominant source of variance; generation CV is 54.72%. The reduced batch also leaves only four trajectories per rank. Even after `balance_shards`, the smoke run's token spread remained high, so these numbers cannot be projected to 32 GPUs.

The candidate 12-rollout validation completed successfully with 62.7783 s mean and 593.53 tokens/s mean, but token counts and rewards diverged after the first update because of nondeterministic Flash Attention backward. It is validation-only, is not a paired improvement claim, and does not reverse the rejection above.

After fixing the disabled-W&B trainer-state counter, a fresh 12-rollout baseline was run with the rejected tokenizer candidate disabled. Its ten retained rounds measured 53.8719 s mean step time, 53.5711 s median, 13.2214 s standard deviation, 24.54% CV, 535.01 tokens/s, 0.6283 samples/s, and 1.985 ms/token. Mean generation, train, and weight-sync times were 48.9249 s, 4.0475 s, and 0.6891 s. Most importantly, `trainer_state.optimizer_step` matched the cumulative four updates per rollout exactly at every boundary: 4, 8, ..., 48. This independent run validates the state fix; variable output lengths mean it does not replace the original baseline or create a speedup claim.

## Nsight Systems findings

Eight per-rank `.nsys-rep` files, eight SQLite exports, official stats JSON, `stats.json`, and `analysis.json` were produced for rollout 2 (11,301 generated tokens). The captured run's wall step was 17.8109 s: generation 14.2938 s, training 2.6668 s, weight sync 0.6074 s, reward 0.1880 s. Nsight values are used for attribution rather than an absolute latency claim.

Across all eight reports:

- 684,494 GPU kernels, or 60.57 kernels per generated token.
- 582,935 kernels below 10 µs: 85.16% of instances, 2.4324 GPU-s.
- 684,494 kernel-launch API calls, 4.1727 API-s.
- CUDA kernel time summed across ranks: 41.4009 GPU-s.
- GEMM: 36,170 kernels, 6.5168 GPU-s.
- NCCL all-gather: 5,528 kernels, 6.2665 GPU-s.
- NCCL reduce-scatter: 1,184 kernels, 6.7361 GPU-s.
- Attention forward: 2,304 kernels, 0.2502 GPU-s.
- Attention backward: 3,456 kernels, 0.2510 GPU-s.
- Softmax: 6,270 kernels, 0.2474 GPU-s.
- KV-cache copy: 576 kernels, 0.00107 GPU-s.
- Explicit top-k and top-p categories: zero kernels, as required by `top_k=0` and `top_p=1.0`.
- Recognized fused cross-entropy/logsumexp category: zero kernels. The dominant unclassified fused reduction was `triton_red_fused__to_copy_add_argmax_clamp_div_log_neg_2` at 16.7615 GPU-s, so it is reported by name rather than mislabeled.

Host synchronization across ranks:

- `cudaEventSynchronize`: 6,254 calls, 58.2177 API-s.
- `cudaStreamSynchronize`: 2,599 calls, 1.7801 API-s.
- `cudaStreamWaitEvent`: 40,678 calls, 0.0595 API-s.
- `cudaDeviceSynchronize`: eight calls, 0.00018 API-s.

Projected NVTX GPU time summed across ranks:

- Full captured step: 142.1032 GPU-s.
- `unirl.rl.train_track`: 21.2024 GPU-s.
- `unirl.ar.optimizer`: 5.7879 GPU-s.
- `unirl.ar.weight_sync`: 4.7589 GPU-s.
- `unirl.ar.logprob_replay`: 4.6188 GPU-s.

The GPU-kernel straggler spread is material: slowest rank / median 1.289, slowest / fastest 1.854, and spread 59.37% of median. The main remaining performance problem is therefore long-output/rank imbalance plus fragmented replay/FSDP work, not DPPO's scalar loss math.

Full-run GPU-monitor means ranged from 6.33% to 19.88% and peak memory reached 52.46 GiB, but these averages include model/Ray startup and shutdown and are not capture-window utilization. During the SGLang NVTX request range, the Ray worker shows 100% host-side wait and 2.23–14.29 s rank wall time because GPU kernels execute in SGLang scheduler child processes; that value must not be interpreted as the SGLang GPU being idle.

SGLang 0.5.12 does not emit scheduler-internal prefill/decode NVTX ranges into this capture. Inventing those boundaries in UniRL would be incorrect. The local one-GPU fixed-token loop provides the only controlled split: prefill is 0.04393 s (11.9% of prefill+eight-token decode time) and decode is 0.32508 s (88.1%). For the real SGLang full step, only the truthful aggregate generation share, 91.73%, is claimed.

Nsight Systems 2025.1 on this node does not expose a standalone `nccl` trace domain. Official `nccl_sum`/`nccl_gpu_sum` reports are unavailable and recorded as nonfatal report gaps. NCCL attribution above comes from CUDA kernel names and NVTX annotations.

## Evaluation

The combined AIME24+AIME25 file was evaluated over all 60 prompts with 16 samples per prompt:

- 960 scored responses.
- Mean per-sample correctness: 0.021875 (2.1875%).
- Generation and reward elapsed time: 720.775 s, excluding model/Ray startup.

This metric is the mean binary reward over 960 responses. It must not be mislabeled as pass@16.

## Artifacts

- Final summary: `outputs/profiling/qwen3_ar/optimization_summary.json`
- Aggregate full-RL analysis: `outputs/profiling/qwen3_ar/full_rl_analysis.json`
- Environment/model manifests: `outputs/profiling/qwen3_ar/assets/`
- Deterministic fixtures: `outputs/profiling/qwen3_ar/fixture/`
- Single-GPU baseline: `outputs/profiling/qwen3_ar/microbaseline_full/analysis.json`
- Packed replay rejection: `outputs/profiling/qwen3_ar/micro_flex/analysis.json`
- Clean micro Nsight capture: `outputs/profiling/qwen3_ar/micro_nsys_clean/`
- 8-GPU baseline: `outputs/profiling/qwen3_ar/full_rl_baseline/`
- Post-fix 12-rollout trainer-state validation: `outputs/profiling/qwen3_ar/full_rl_postfix_baseline/`
- 8-GPU Nsight capture: `outputs/profiling/qwen3_ar/full_rl_nsys/`
- Optimized validation: `outputs/profiling/qwen3_ar/full_rl_validation/`
- AIME evaluation: `outputs/profiling/qwen3_ar/aime_eval/`
- Trainer-state audit: `outputs/profiling/qwen3_ar/trainer_state_smoke/trainer_state_audit.json`

## Verification

- Pre-hardening `pytest -q tests`: 62 passed; one third-party `pynvml` deprecation warning. Post-review focused tests were added for stable-ID seed derivation, DP slicing, VLM propagation, seed-mode guards, disabled-W&B step parity, and scalar checksum dtypes; their local run did not reach collection because the Ceph-hosted runtime environment stalled while importing dependencies.
- Ruff over `scripts/profiling/*.py`, `unirl`, and `tests`: clean.
- Shell syntax for both profiling launchers and `git diff --check`: clean.
- Canvas TypeScript check: no errors.
- Final branch distance from freshly fetched `upstream/main`: 0 behind, 0 ahead.

## Reproduction commands

```bash
cd /apdcephfs/private_aimicahchen/Project/UniRL-qwen3-ar-nsys-opt
export LD_LIBRARY_PATH=/apdcephfs/private_aimicahchen/.local/cuda-compat-13.0/usr/local/cuda-13.0/compat:/usr/local/cuda-12.9/lib64
export QWEN3_PATH=Qwen/Qwen3-4B-Base
export DATA_PATH=$PWD/data/dapo_math/train.jsonl
export EVAL_DATA_PATH=$PWD/data/dapo_math/aime_eval.jsonl
export UNIRL_SEED=20260712

# 8-GPU baseline / validation harness
RUN_KIND=baseline bash scripts/profiling/run_qwen3_ar_full_rl.sh
RUN_KIND=validation bash scripts/profiling/run_qwen3_ar_full_rl.sh

# Deferred per-rank Nsight capture
RUN_KIND=nsys UNIRL_NSYS_CAPTURE_ROLLOUT=2 \
  bash scripts/profiling/run_qwen3_ar_full_rl.sh

# Full AIME24+AIME25 evaluation
RUN_KIND=eval bash scripts/profiling/run_qwen3_ar_full_rl.sh
```
