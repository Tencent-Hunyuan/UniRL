# How to reproduce the bf16 vs fp32 gradient comparison

There are two ways to reproduce the numbers in the PR description. Only the first one can be run outside our setup.

| | A. Standalone script | B. Fixed-batch replay |
|---|---|---|
| Depends on UniRL | no (plain PyTorch, diffusers, peft) | yes (this branch) |
| Needs our training run | no | yes: the run directory, the reward service, the prompt list |
| Inputs | H3 weights, one captured micro-batch (145 MB), the step-20 LoRA | the saved 192-row batch and the step-20 LoRA, both produced by step B1 |
| Produces | per-block forward error and back-propagated signal, gradients of blocks 0–2 | the full 600-tensor LoRA gradient of the 192-row batch |
| Figure in `results/` | figure 2 | figure 1 |
| Time on our host | 71 min | 13 min (8 rows per forward) to 39 min (1 row per forward) per variant |

The captured micro-batch and the step-20 LoRA are not in this branch. We can share both on request.

## Environment

- 8 GPUs with about 64 GB each. fp32 H3 weights take 124 GiB, spread over the 8 cards; the standalone script peaked at 46.3 GiB allocated on one card.
- About 50 GiB of free host memory at start. The standalone script stops by itself when available memory drops below 12 GiB.
- PyTorch 2.13.0+cu130, diffusers 0.42.1, peft 0.21.2, accelerate, safetensors, psutil.
- Our GPUs report as `NVIDIA CMP 170HX`. Nothing below has been run on stock A100s.

## A. Standalone script

Files: `standalone/`. The model code is the vendored `unirl/models/minimax_h3/vendor/transformer_minimax_h3.py` of this branch, loaded by path; nothing else from UniRL is imported in the default mode.

### A1. CPU self-check (no weights needed, a few seconds)

Runs the whole flow on a small real H3 (4 blocks, hidden size 16) with a real PEFT LoRA.

```bash
V=unirl/models/minimax_h3/vendor/transformer_minimax_h3.py
python diag/h3_bf16_fp32/standalone/cpu_validate.py \
  --vendor "$V" --out /tmp/h3-cpu-check --with-lora --loss-options
```

Expected: exit code 0, and fp32 batch-8 vs batch-1 gradient cosine of 1 up to rounding. `--out` must be a new directory.

### A2. GPU run with a fixed cotangent (this produced figure 2)

Four cases on the same 8 captured rows: fp32 and bf16, each as one batch of 8 and as 8 batches of 1. Every case starts the backward pass from the same captured gradient at the model output.

```bash
python -u diag/h3_bf16_fp32/standalone/standalone_micro.py \
  --vendor "$V" \
  --pretrained <H3 weights directory> \
  --dump <captured batch>/fwd_rank0.pt \
  --live-checkpoint <step-20 checkpoint>/checkpoint.pt \
  --lora-alpha 64 \
  --out <new output directory>
```

The LoRA is rank 32, alpha 64, on `attn.to_q/k/v`, `attn.to_out.0`, `ff.net.0.proj`, `ff.net.2` of all 50 blocks. It is not merged into the base weights.

What to read in the output directory:

| File | Content |
|---|---|
| `captured_outputs.json` | each case's output against the output the training process produced; bf16 batch 8 must be identical (cosine 1, relative L2 0) |
| `summary.json` → `blocks_0_2_gradient_vector` | gradient of all parameters of blocks 0–2, the four comparisons below |
| `block-XX.json` | per block: forward hidden-state error and cosine of the back-propagated signal |
| `location.json`, `probe-bf16.json` | first place where batch 8 and batch 1 differ, per operation inside block 0 |
| `*-settings.json`, `*-block0-profile.json` | precision flags and GPU kernel names actually used |

Numbers we got:

| Comparison (gradient of blocks 0–2) | Cosine |
|---|---|
| fp32 batch 8 vs fp32 batch 1 | 0.99997 |
| bf16 batch 8 vs bf16 batch 1 | 0.289 |
| bf16 batch 8 vs fp32 batch 8 | 0.115 |
| bf16 batch 1 vs fp32 batch 1 | 0.091 |

Back-propagated signal, bf16 against fp32: 0.997 at block 49, about 0.92 at block 35, 0.60 at block 30, 0.15–0.26 from block 25 down to block 0.

### A2b. LoRA-rounding control

Adds `--rounding-control`. Runs three batch-8 cases on the same inputs and cotangent: fp32, fp32 with the LoRA masters rounded to bf16 and upcast (`fp32r_b8`), and bf16. The batch-1 cases and the block-0 probe are skipped.

| Comparison (gradient of blocks 0–2) | Cosine | Relative L2 |
|---|---|---|
| fp32 rounded LoRA vs fp32 (adapter rounding only) | 0.9988 | 0.055 |
| bf16 vs fp32 rounded LoRA (compute precision only) | 0.113 | 1.150 |
| bf16 vs fp32 | 0.111 | 1.142 |

Adapter rounding gives a relative error of at most 0.001 in blocks 36–49 and about 0.05 in blocks 0–20; compute precision gives 1.1 in the shallow blocks. Per block: `results/data/rounding_control_grad_per_block.csv`.

### A3. Optional: each case computes its own cotangent from the loss

Adds `--cotangent loss --reference-output recompute --groups all --replay-inputs`. Each case rebuilds the Flash-GRPO loss (policy term plus KL, beta 0.004) from its own output. This needs a capture that also stores the loss inputs (see A5) and imports the SDE and loss functions from this branch.

We completed this for one prompt group only (sigma = 1.0): bf16 batch 8 vs fp32 0.074, bf16 batch 1 vs fp32 0.377. The run then stops at the second group, because the script checks that its bf16 batch-8 output is identical to the captured one and it is not; A4 explains why. The script on this branch does not restore the input memory layout, so groups 2 and 3 cannot be completed with it as is.

### A4. Optional: input memory layout (forward only, about 15 min)

```bash
python -u diag/h3_bf16_fp32/standalone/diag_g1.py \
  --vendor "$V" --pretrained <H3 weights directory> \
  --dump <three-group capture>/fwd_rank0.pt \
  --live-checkpoint <step-20 checkpoint>/checkpoint.pt \
  --lora-alpha 64 --out <new output directory>
```

It runs the bf16 batch-8 forward 46 times, changing one thing at a time, and compares with the captured output. In `results.json`, only the `input_strides` variant (inputs placed as slices of a larger tensor, as in training) makes the second group identical to the capture. With contiguous inputs of the same values the output differs by 3.2 % (video) and 1.3 % (audio) relative L2.

### A5. How the micro-batch was captured

`capture_supervisor.py` starts the replay of part B (bf16 with the speed-ups off, 8 rows per forward), waits until rank 0 has saved the inputs, outputs and output gradients of its first micro-batches, then stops the replay. The hook is installed by adding these lines to `unirl/utils/h3_checkup_replay.py`, right after `assert len(parameter_names) == 600`; they are not committed:

```python
        from os import environ as capture_environ
        capture_dir = capture_environ.get("H3_DIAG_DUMP_FWD")
        if capture_dir and rank == 0:
            from capture_hook import install_capture, make_loss_context
            install_capture(backend.model, capture_dir, {
                "variant": variant, "micro": micro, "rank": rank,
                "rollout_id": 20, "same_prompt_group": True,
                "population_rows": 192, "local_rows": 24,
            }, loss_context=make_loss_context(self.algorithm, part, 20 / 199),
               groups=int(capture_environ.get("H3_MICRO_CAPTURE_GROUPS", "1")))
```

```bash
python -u diag/h3_bf16_fp32/standalone/capture_supervisor.py \
  --directory <new capture directory> --groups 3 --timeout 3600 -- \
  python -u replay.py --variant V2 --micro 8
```

Known problem: at the end the supervisor refuses to stop the process group because two Ray helper processes do not carry its owner tag. The capture file is complete at that point; we stopped the remaining processes by hand.

## B. Fixed-batch replay

Files: `unirl/utils/h3_checkup_*.py` (the replay itself), `replay-tools/` (launchers), `configs/` (the recipe and overrides as run, and `VARIANTS.md`).

This part is tied to our run. It needs:

- the training run directory with its Hydra config and `ckpt/checkpoint-20` (`replay-tools/runtime.py` reads `runs/long192-klema/...` under `TRAIN_ROOT`);
- an HPSv3 reward service and the adapter `configs/unirl_hps_adapter.py`, which lives outside the repository;
- our prompt list.

The code also assumes 8 ranks, 24 rows per rank, 600 LoRA tensors and rollout id 20.

Layout expected by `replay-tools/env.sh`: `REPLAY_ROOT/source` is a checkout of this branch, `REPLAY_ROOT/tools` is a copy of `replay-tools/`, `TRAIN_ROOT` is the training task root (its `env.sh`, virtualenv, `adapter/` and `runs/`).

```bash
export TRAIN_ROOT=<training task root> REPLAY_ROOT=<replay task root>
. "$REPLAY_ROOT/tools/env.sh"

# B1. Sample 192 rows once with the step-20 weights and save them per rank (no optimizer step).
python -u tools/capture.py

# B2. Recompute the gradient of the saved batch, one process per variant.
python -u tools/replay.py --variant V1 --micro 8      # bf16 as trained
python -u tools/replay.py --variant V3 --micro 4      # fp32 reference
python -u tools/replay.py --variant V2 --micro 1      # bf16, speed-ups off

# B3. The two extra settings, written to their own folders.
CHECKUP_OUT_NAME=V1m1 CHECKUP_SPLITS=full python -u tools/replay.py --variant V1 --micro 1
CHECKUP_OUT_NAME=V2m8 CHECKUP_SPLITS=full python -u tools/replay.py --variant V2 --micro 8

# B4. Compare: overall and per-block cosine, norm ratio, relative L2.
python tools/cmp_micro.py V2m8/full V2/full V3/full V1/full V1m1/full
```

`chain.py` is the wrapper we used to run B1 and B2 in sequence with a time limit. Every replay loads the same saved batch and the same LoRA, uses `old_logp = new_logp.detach()` and checks afterwards that no parameter changed. Gradients are written to `gradients/<name>/<split>/rank{0..7}.pt`; B4 writes `results/micro-compare.json`. The files we obtained are in `results/data/grad_compare_raw/`.

Numbers we got (cosine of the 600 LoRA gradients against V3, 192 rows):

| Setting | Folder | Cosine vs fp32 | Blocks 0 / 1 / 2 |
|---|---|---|---|
| bf16, speed-ups off, 1 row per forward | `V2` | 0.480 | 0.91 / 0.45 / 0.26 |
| bf16, speed-ups off, 8 rows per forward | `V2m8` | 0.037 | −0.43 / 0.04 / 0.02 |
| bf16, as trained, 1 row per forward | `V1m1` | 0.131 | see `results/data/grad_cos_per_block.csv` |
| bf16, as trained, 8 rows per forward | `V1` | 0.028 | see `results/data/grad_cos_per_block.csv` |

## What these steps do not cover

- No training run in fp32 exists, so there is no reward curve to set against the bf16 curves in `results/`.
- Only the step-20 checkpoint and the first update were measured.
- The second framework mentioned in the PR description is not part of this branch.
