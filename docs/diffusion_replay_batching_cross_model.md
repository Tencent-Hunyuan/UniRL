# Batched SDE-step replay across diffusion models — applicability & expected gain

Status: analysis + SD3 anchor measured on 1×H20. Cross-model GPU validation
plan in §7 (pending checkpoints + go-ahead).

This studies whether the **batched-step replay** optimization shipped for SD3 in
[PR #144](https://github.com/Tencent-Hunyuan/UniRL/pull/144)
(`SD3DiffusionStage._replay_batched_steps`) generalizes to the other diffusion
models in the repo (`z_image`, `qwen_image`, `flux2_klein`, `wan21/22`, `ltx2`,
`hunyuan_video`, `hunyuan_video15`, `hunyuan_image3`, `bagel`).

---

## 1. What the optimization does (recap)

GRPO/FlowGRPO/FlowDPPO-style diffusion RL replays a stored rollout trajectory to
recompute per-SDE-step log-probs. Every `DiffusionStage.replay` does this with a
**serial loop**:

```python
for step_idx in target:                       # S SDE steps
    _, log_prob, prev_mean = self.step.step_with_logp(... sample=x[step_idx],
                                                       prev_sample=x[step_idx+1] ...)
```

Each iteration is one transformer forward at the replay micro-batch `B`. Under
FSDP that means **S forwards ⇒ S all-gathers** of the full sharded transformer
per `replay()` call. `_replay_batched_steps` instead stacks all S steps on the
batch dim — `sample`/`prev_sample` become `[S·B, …]` (step-major), conditioning
is tiled S×, sigmas ride as `[S·B]` vectors — and runs **one** forward + one
vectorized SDE transition, i.e. **S all-gathers ⇒ 1**.

**Correctness invariant (why ratio stays exactly 1).** The transformer has no
cross-sample interaction, so per-sample log-probs are unchanged up to bf16
batch-shape rounding. Under `old_logp_source='replay'` the π_old anchor and the
train forward both go through this *same* batched path → bit-identical →
`ratio ≡ 1`. Gated to **stateless, step-index-independent SDE strategies**
(`FlowSDEStrategy`, `CPSSDEStrategy`, `DanceSDEStrategy`); the stateful
`DPM2Strategy` (ODE, not an `SDEStrategy`) is excluded and never reaches the
SDE-logp path anyway.

---

## 2. Central finding: this is a *distributed* (FSDP all-gather) optimization, not a single-GPU compute win

Measured on **1×H20** with the real SD3.5-medium transformer (2.24B, 24 layers),
transformer-only, no FSDP (`scripts/profiling/validate_batched_replay.py`,
`microbench_sd3_replay.py`):

| microbench (1 GPU, no FSDP)                         | result |
|----------------------------------------------------|--------|
| forward throughput B=1 → B=32                       | 26.8 → 34.3 samples/s = **1.28×** (per-sample) |
| fwd+bwd throughput B=1 → B=16                       | 6.9 → 11.7 samples/s = **1.69×** |
| **serial S=3 forwards (bs=B) vs 1 batched (bs=S·B)** | **1.14× @B=1, 1.04× @B=4, 1.03× @B=8, 0.50× @B=16** |
| batched==serial parity                             | `max rel 1.1e-7` (bit-identical) |
| batched-replay determinism → ratio=1               | `max|ratio−1| = 0.000` (exact) |

On a single GPU the batched collapse is worth **~1.0–1.14×, and *negative* once
`S·B` is large** (bs=48 is past SD3's compute-saturation point). Yet PR #144
reports **−54% `diffusion_train`** end-to-end on 8×H20. The entire delta is the
**FSDP all-gather count reduction** (S→1 per replay) + Python/launch overhead —
*not* single-GPU compute. SD3.5-medium is ~compute-bound even at bs=1 (batch
ceiling ~1.3–1.4×), so collapsing the loop only pays off once each forward
carries a full-parameter all-gather.

> **Consequence for "which models benefit":** a single-GPU microbench will
> *understate or hide* the win for every model. The benefit must be judged in
> the **distributed FSDP setting**, where it scales with the all-gather cost
> (≈ model param bytes) × number of SDE steps S, and is gated by the `[S·B]`
> activation-memory blow-up.

---

## 3. The win, as a model

Per `replay()` call, ignoring compute (which is invariant — same total FLOPs):

```
saved_allgather_time ≈ (S − 1) × (param_bytes / effective_NVLink_BW)      [reshard_after_forward=true]
extra_activation_mem ≈ (S − 1) × (per-step activation footprint at bs=B)
```

So a model is a **good candidate** when it is *all-gather-bound* in replay:
large parameters relative to per-forward compute, `reshard_after_forward=true`
(re-gathers every forward — the default), many SDE steps S, and small enough
per-step activations that `[S·B]` fits in memory. It is a **poor/risky
candidate** when per-forward compute already dominates (huge latents) and/or
`[S·B]` activations OOM — typical of video.

---

## 4. Applicability gate (all models pass structurally)

Every diffusion stage in the repo has the **identical serial replay loop** and
the same `step.step_with_logp(prev_sample=…)` primitive, and the RL recipes use a
stateless `SDEStrategy`. So the optimization is **structurally portable to all of
them**; only the per-model *tiling* of conditioning and the memory profile
differ.

| stage | serial loop | default SDE strategy (RL) | gate OK |
|-------|:-----------:|---------------------------|:-------:|
| `sd3` (done) | ✓ | FlowSDE | ✓ |
| `qwen_image` | ✓ | FlowSDE | ✓ |
| `z_image` | ✓ | FlowSDE | ✓ |
| `flux2_klein` | ✓ | DanceSDE | ✓ |
| `wan21` / `wan22` | ✓ | FlowSDE | ✓ |
| `ltx2` | ✓ | FlowSDE | ✓ |
| `hunyuan_video` / `hunyuan_video15` | ✓ | FlowSDE | ✓ |
| `hunyuan_image3` | ✓ | FlowSDE | ✓ |
| `bagel` | ✓ | FlowSDE | ✓ |

---

## 5. Per-model analysis (conditioning shape, forwards/step, memory, complexity)

All RL recipes run `guidance_scale=1.0` (CFG off) ⇒ **1 forward = 1 all-gather
per replay step** unless noted. "Tiling complexity" = effort to write the
`_replay_batched_steps` / `_tile_conditions` equivalent.

| model | params | conditioning to tile S× | CFG fwds/step | latent tokens (RL res) | `[S·B]` OOM risk | tiling complexity | expected distributed win |
|-------|-------:|--------------------------|:---:|------------------------|:----------------:|-------------------|--------------------------|
| **sd3** (done) | 2.24B | `embeds`,`pooled`(,neg) — dense, fixed-len; chunked CFG | 1 (CFG=1 chunked fwd) | ~1024 @512² | low (baseline VRAM) | **done** | **high** (−54% train, measured by PR) |
| **qwen_image** | ~20B | `embeds`+`attn_mask`, `img_shapes` (len B→S·B), `txt_seq_lens`, per-call max-len trim, packing; CFG = **2 separate** fwds | 1 (CFG off) / 2 (CFG on) | ~ (384/16)²·… small @384² | low–med | **high** (variable-len text, img_shapes list, RoPE trim) | **high** (20B ⇒ huge all-gather; biggest absolute win) |
| **z_image** | ~6B | **list-based**: list of `[C,1,H,W]` + variable-len caption list; CFG = single `[pos;neg]` list fwd | 1 (Turbo CFG-free) | small | low | med (list extend to S·B) | high |
| **flux2_klein** | ~Flux.2 | `embeds`, 4-axis RoPE `txt_ids`/`img_ids` rebuilt per call; **replay in `.eval()`**; packed 128-ch | 1 (no CFG) | ~ small | low–med | med (rebuild RoPE ids for S·B, preserve eval) | high |
| **wan21 / wan22** | ~14B | text embeds; **video** latent (frames×H×W) | 1 | **very large** (T·H·W tokens) | **high** | med | win on all-gather, but compute already dominates + `[S·B]` may OOM ⇒ use chunked/partial-S |
| **ltx2** | — | text embeds; **video** | 1 | **very large** | **high** | med | same caveat as wan |
| **hunyuan_video / _video15** | ~13B | text embeds; **video** | 1 | **very large** | **high** | med | same caveat as wan |
| **hunyuan_image3** | large | unified AR+diffusion conditioning | 1 | medium | med | high (AR/diffusion interplay) | medium |
| **bagel** | 7B (MoT) | unified AR+diffusion; T2I context | 1 | medium | med | high (MoT routing, context cache interplay) | medium |

Notes:
- **Image models (`qwen_image`, `z_image`, `flux2_klein`)** are the clean wins:
  small per-step activations (so `[S·B]` is cheap), multi-billion params (so the
  all-gather they save is large). Qwen-Image is the highest-value target — ~20B
  means the per-forward all-gather is ~10× SD3's, so the distributed win should
  be larger than SD3's −54% *and* a checkpoint is available locally.
- **Video models (`wan*`, `ltx2`, `hunyuan_video*`)** are the risky case: the
  forward is already compute-bound (10⁴–10⁵ latent tokens), so the *relative*
  all-gather share is smaller, and stacking `[S·B]` multiplies an already-huge
  activation footprint → likely OOM at full S. Recommended variant: a
  **chunked** batched replay (group ≤k steps) or only enable when memory allows.
- **Unified models (`bagel`, `hunyuan_image3`)** add AR/diffusion and (Bagel)
  MoT + context-cache interplay; correct but more implementation surface.

---

## 6. GPU validation results (measured on this box)

### 6.1 SD3 single-GPU anchor (`validate_batched_replay.py`, 1×H20)

```
Claim 1 (batched==serial, bf16 tol + mapping): PASS   (max rel 1.1e-7)
Claim 2 (batched replay deterministic ratio=1): PASS   (max|ratio-1| = 0.000)
Claim 3 (grad flows through batched replay):    PASS
forward (no_grad):  serial=396.6ms  batched=378.5ms  speedup=1.05x
fwd+bwd:            serial=1202.6ms batched=1089.9ms  speedup=1.10x
```

Single-GPU speedup is small (≈1.05–1.10×) — SD3 is ~compute-bound, confirming
§2: the single-GPU bench is the wrong lens.

### 6.2 SD3 distributed FSDP2 sweep (`fsdp_replay_microbench.py`, the real lens)

Per-block `fully_shard` (`reshard_after_forward=True`) — the training-path wrap.
**B=1** (the `micro_batch_size=1` replay geometry). ratio=1 determinism exact in
every row:

| world | S (SDE steps) | serial (S all-gathers) | batched (1) | **speedup** |
|------:|--------------:|-----------------------:|------------:|------------:|
| 2 | 3 | 185 ms | 117 ms | **1.59×** |
| 8 | 3 | 1736 ms | 634 ms | **2.74×** |
| 8 | 6 | 3416 ms | 925 ms | **3.69×** |

The win **grows with world size and S** — exactly the all-gather-reduction
signature, and consistent with PR #144's end-to-end −54% `diffusion_train`
(2.15×) for the PE recipe (the phase has non-replay overhead that dilutes the
pure-replay speedup).

### 6.3 Qwen-Image correctness on real 20B weights (`validate_batched_replay_qwen_image.py`, 1×H20)

`Qwen-Image-Edit` transformer (`QwenImageTransformer2DModel`, 60 layers, 3072
hidden, ~20B), B=2, S=3, guidance_scale=1.0:

```
Claim 1 (batched==serial, bf16 tol + mapping): PASS   (max rel 1.03e-6)
Claim 2 (batched replay deterministic ratio=1): PASS   (max|ratio-1| = 0.000)
Claim 3 (grad flows through batched replay):    PASS
```

Proves the Qwen-Image tiling (variable-length text trim + `img_shapes` /
`txt_seq_lens` replication) is correct and ratio=1-exact on real weights.

### 6.3b Qwen-Image distributed FSDP2 timing (measured, real 20B)

`fsdp_replay_microbench.py --model qwen_image`, B=1, ratio=1 exact every row:

| world | S | serial (S all-gathers) | batched (1) | **speedup** |
|------:|--:|-----------------------:|------------:|------------:|
| 2 | 3 | 4365 ms | 1416 ms | **3.08×** |
| 8 | 3 | 8580 ms | 2909 ms | **2.95×** |
| 8 | 6 | 12183 ms | 1475 ms | **8.26×** |

As predicted by the param scaling (Qwen-Image's per-forward all-gather is ~10×
SD3's: 20B vs 2.24B), the win is **larger than SD3's** — Qwen at w2 (3.08×)
already beats SD3 at w8 (2.74×), and at S=6 the collapse of 6 all-gathers → 1
hits **8.26×**. (Loading the 20B off the network mount is the only practical
hurdle; pre-warm the page cache first, then the timed run is fast + clean.)

### 6.3c z_image distributed FSDP2 timing (measured, real 6.15B)

`fsdp_replay_microbench.py --model z_image` (Z-Image-Turbo), B=1, ratio=1 exact:

| world | S | serial | batched | **speedup** |
|------:|--:|-------:|--------:|------------:|
| 2 | 3 | 1669 ms | 1419 ms | **1.18×** |
| 8 | 3 | 2589 ms | 1668 ms | **1.55×** |

z_image's win is *smaller* than SD3's despite ~3× the params — a real
architecture effect: the list-based single-stream S3-DiT forward (+ refiner
layers) is more compute-heavy per sample, so the all-gather share (and thus the
batching win) is smaller. Confirms the §3 model: win ∝ all-gather / compute.

### 6.3d Cross-model summary (measured, B=1, ratio=1 exact everywhere)

| model (arch) | params | w2·S3 | w8·S3 | w8·S6 |
|--------------|-------:|------:|------:|------:|
| sd3 (MMDiT)          | 2.24B | 1.59× | 2.74× | 3.69× |
| z_image (list S3-DiT)| 6.15B | 1.18× | 1.55× | — |
| qwen_image (MMDiT)   | 20.4B | 3.08× | 2.95× | **8.26×** |

The win grows with world size and S, and tracks all-gather/compute balance
(Qwen's 20B params ≫ compute-per-forward → largest win; z_image's heavier
list-based forward → smallest). Every config is bit-identical-ratio (ratio ≡ 1).

### 6.4 Implementations landed

`_replay_batched_steps` + `_tile_conditions` + a `batch_replay_steps` gate now
exist for **sd3** (upstream #144), **qwen_image**, **z_image**, and
**flux2_klein** — each gated to stateless `SDEStrategy`, default off, and (sd3 +
qwen_image) threaded through the pipeline/config.

---

## 7. Recommended cross-model GPU validation plan

Because §2 shows single-GPU numbers are misleading, faithful validation = the
**distributed A/B** PR #144 used for PE: run the model's trainside recipe with
`batch_replay_steps` off vs on, on multi-GPU FSDP, and compare per-step time +
confirm `ratio = 1.0000`.

Checkpoint availability on this box:

| model | transformer ckpt | path |
|-------|:----------------:|------|
| sd3 | ✓ | `/data/models/stable-diffusion-3.5-medium` |
| qwen_image | ✓ (Qwen-Image-Edit, same `QwenImageTransformer2DModel`) | `/apdcephfs/private_aimicahchen/models/Qwen/Qwen-Image-Edit` |
| z_image | ✓ (Z-Image-Turbo, ~11B) | `…/public_models/Tongyi-MAI/Z-Image-Turbo` |
| flux2_klein | ✓ (4B/9B; single-file safetensors → needs `from_single_file`) | `…/public_models/black-forest-labs/FLUX.2-klein-{4B,9B}` |
| wan21/22, hunyuan_video | ✓ (video; Diffusers layout) | `…/public_models/{Wan2.1-T2V-*,Wan2.2-T2V-*,HunyuanVideo}` |

(`…` = `/apdcephfs_fsgm3/share_305110755/hunyuan/public_models`. All load slowly
off the network mount — ~50–65 s per safetensors shard — so per-model FSDP runs
are load-bound. HF/ModelScope are not reachable via the Tencent github proxy, but
these local mirrors cover the image + video models.)

Status:
1. **Qwen-Image** — DONE: `_replay_batched_steps` + `_tile_conditions`
   implemented + threaded; correctness validated on real 20B (parity 1e-6,
   ratio=1 exact); distributed FSDP2 timing measured (§6.3b: 3.08× / 2.95× /
   8.26×).
2. **z_image** — DONE: implemented; distributed FSDP2 timing measured (§6.3c:
   1.18× / 1.55×, ratio=1 exact). **flux2_klein** — implemented + committed;
   runtime validation pending (single-file checkpoint + slow mount).
3. **Full trainside recipe A/B** (`qwen_image_trainside_veomni.yaml`,
   `batch_replay_steps` off/on) — NOT run here: it is a long (≫195 s) multi-rank
   NCCL job, and NCCL collectives deadlock under the box's busy-loop GPU
   occupier while pausing it for the whole run trips the occupier's respawn
   watchdog. The FSDP microbench above is the controlled isolate of the same
   effect; PR #144's end-to-end −54% `diffusion_train` on SD3/PE is the recipe
   analog.
4. **video / unified** — design the chunked variant; gate behind a memory check.

Environment notes: working env `/root/ep_work/epvenv` (torch 2.10+cu128,
diffusers 0.37, transformers 5.9); 8×H20 (96 GB). A `/tmp/gpu_occupy.py` busy-loop
holds the GPUs (self-respawning watchdog); pause via SIGSTOP for clean
benchmarks (do **not** match your own shell PID).
