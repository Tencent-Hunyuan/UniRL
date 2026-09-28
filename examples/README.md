# Examples

Self-contained Hydra recipes — one YAML per experiment. A recipe is the single
source of truth for a run: model, algorithm, rollout engine, placement, reward,
weight sync, and batch geometry, each instantiated directly by `_target_` (no
Hydra config-group overrides). Recipes are grouped by trainer domain or agentic
workflow. Select one with `--config-name` set to its path under `examples/`,
without `.yaml`. Keep every directory — `diffusion/sd3/sd3_trainside`, not
`sd3_trainside`.

> This directory replaces the old top-level `recipes/` tree.

## Domains & entrypoints

The **default recipe** is used when `--config-name` is omitted
(`@hydra.main(config_name=...)` on the entrypoint). Async AR and diffusion have
their own entrypoints; those recipes still live next to the sync ones.

| Training path | Entrypoint | Built-in default recipe |
|---|---|---|
| Diffusion RL | [`python -m unirl.train_diffusion`](../unirl/train_diffusion.py) | [`diffusion/sd3/sd3_trainside`](diffusion/sd3/sd3_trainside.yaml) |
| AR RL | [`python -m unirl.train_ar`](../unirl/train_ar.py) | [`ar/qwen_vl_grpo_geo3k_mc_4x8`](ar/qwen_vl_grpo_geo3k_mc_4x8.yaml) |
| SFT | [`python -m unirl.train_sft`](../unirl/train_sft.py) | [`sft/qwen3_sft`](sft/qwen3_sft.yaml) |
| Prompt enhancement | [`python -m unirl.train_pe`](../unirl/train_pe.py) | [`pe/pe_trainside_pickscore`](pe/pe_trainside_pickscore.yaml) |
| Unified RL | [`python -m unirl.train_unified_model`](../unirl/train_unified_model.py) | [`unified_model/hi3_vllmomni`](unified_model/hi3_vllmomni.yaml) |
| Agentic RL | [`python -m unirl.train_agentic`](../unirl/train_agentic.py) | [`deep_research/deep_research_search_judge`](deep_research/deep_research_search_judge.yaml) |
| Async AR RL | [`python -m unirl.train_async_ar`](../unirl/train_async_ar.py) | [`ar/qwen3_grpo_4b_base_dapo_sglang_async`](ar/qwen3_grpo_4b_base_dapo_sglang_async.yaml) |
| Async diffusion RL | [`python -m unirl.train_async_diffusion`](../unirl/train_async_diffusion.py) | [`diffusion/bagel/bagel_vllmomni_async`](diffusion/bagel/bagel_vllmomni_async.yaml) |

The AR default is written for 4×8 (32 GPUs) and requires `DATA_PATH`. Launchers
override `num_devices` from the node GPU count. Engine extras are in
[INSTALL.md](../INSTALL.md).

## Running a recipe

Launchers live in this directory. The first argument is the recipe path under
`examples/`, without `.yaml`. Extra args are Hydra overrides.
`ENTRY` is the module name from the table (`train_sft`, `train_async_ar`, …)
and defaults to `train_diffusion`; it is not inferred from the recipe path.

Run from the repository root in the engine environment from INSTALL.md.
Launchers run `pip install --no-deps -e .` and will not install missing
dependencies.

```bash
# Compose-check (no training). Pair the entrypoint with the recipe.
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve
```

```bash
# Single node
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
ENTRY=train_ar bash examples/run_experiment_single_node.sh ar/qwen_vl_grpo_geo3k_mc_4x8
# SFT: set SFT_DATA to the training manifest.
ENTRY=train_sft bash examples/run_experiment_single_node.sh sft/qwen3_sft
ENTRY=train_pe  bash examples/run_experiment_single_node.sh pe/pe_trainside_pickscore
ENTRY=train_unified_model bash examples/run_experiment_single_node.sh unified_model/hi3_vllmomni
ENTRY=train_agentic bash examples/run_experiment_single_node.sh deep_research/deep_research_search_judge
# Async AR: SGLang environment; set DATA_PATH to the training data.
ENTRY=train_async_ar bash examples/run_experiment_single_node.sh ar/qwen3_grpo_4b_base_dapo_sglang_async
# Async diffusion: vLLM-Omni environment; set BAGEL_PATH to the model checkpoint.
ENTRY=train_async_diffusion bash examples/run_experiment_single_node.sh diffusion/bagel/bagel_vllmomni_async

# Multi-node
bash examples/run_experiment_multinode.sh diffusion/sd3/sd3_sglang_rollout_colocate

# Direct entrypoint, without the launchers
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside num_devices=8
```

`DRY_RUN=1` prints the launcher command and skips compose and execution.
Launchers set `num_devices` from the GPU count; the recipe's batch and
train/rollout splits still have to fit.

Cluster paths and W&B identity come from `${oc.env:...}` in the selected YAML
(see [INSTALL.md](../INSTALL.md#environment)). Fields without that interpolation
need a Hydra override, for example `logging.report_to_wandb=true` on
`pe/pe_trainside_pickscore`. Start the mooncake metadata server before
`*_tq_mooncake` recipes: `bash examples/mooncake_master.sh start`.

To save: `++save_interval=100 ++save_dir=checkpoints`. To resume:
`++load_dir=<checkpoint-dir>`. Use `++` so Hydra can add or override keys
(SFT recipes already define `save_interval`). HI3 meta-init, including
[`unified_model/hi3_vllmomni`](unified_model/hi3_vllmomni.yaml), needs DCP —
see [Checkpointing](../unirl/trainer/README.md#checkpointing).

## WAN2.1 UCF-101 full-transformer SFT

[`sft/wan21_t2v_ucf101_full`](sft/wan21_t2v_ucf101_full.yaml) performs full
WAN2.1 transformer finetuning from caption/target-video manifests on an
18-class UCF-101 sports/action subset.

See [`datasets/ucf101/README.md`](../datasets/ucf101/README.md) for the exact
download location, expected directory layout, cooking command, manifest
format, and training command.

The UCF-101 run used 2,367 training videos and 128 held-out videos, 17 frames at
256x448, global batch 8, learning rate `2e-6`, and 300 optimizer steps on one
8xH20 node. Deterministic held-out eval loss improved from `0.21981` at
initialization to a best `0.16230` at step 260 (-26.2%), finishing at `0.16264`
at step 300.

## Reading a recipe name

A recipe filename is a fixed-order, `_`-joined chain of segments. Every segment
except `model` is optional and is **omitted when it is the default or does not
apply** — so a name carries only what distinguishes it from its siblings, and
related recipes sort together.

```
<model>[_<task>][_<size>][_<algorithm>][_<engine>][_<adapter>][_<topology>]
```

| Segment | Position | Values (examples) | Omit when |
|---|---|---|---|
| `model` | required, first | `sd3`, `qwen_image`, `flux2_klein`, `sensenova_u1_5`, `wan21`, `wan22`, `hunyuan_video10`, `hunyuan_video15`, `qwen_vl`, `qwen3`, `hi3` | never |
| `task` | after model | `t2v`, `i2v` | text-to-image (the implicit default) |
| `size` | after task | `4b`, `14b` | only one size in the family |
| `algorithm` | middle | `dancegrpo`, `mixgrpo`, `nft`, `flowdppo`, `grpo`, `drpo` | plain FlowGRPO (diffusion default); GRPO (AR default) |
| `engine` | after algorithm | `trainside`, `sglang`, `vllmomni` | — |
| `adapter` | after engine | `full`, `lora` | unambiguous from the rest |
| `topology` | last | placement `colocate`/`separate`; sync `nccl`/`tensor`/`ipc`; engine mode `rollout`/`replay` | single-slab colocate default |

Worked examples:

| Recipe | Reads as |
|---|---|
| `sd3_trainside` | SD3 · trainside engine · (default FlowGRPO) |
| `sd3_nft_sglang` | SD3 · DiffusionNFT · SGLang engine |
| `qwen_image_dancegrpo` | Qwen-Image · DanceGRPO |
| `wan22_t2v_14b_dancegrpo` | WAN 2.2 · text-to-video · 14B · DanceGRPO |
| `hunyuan_video10_t2v_trainside` | HunyuanVideo-1.0 · text-to-video · trainside engine |
| `hunyuan_video15_t2v_dancegrpo_trainside` | HunyuanVideo-1.5 · text-to-video · DanceGRPO · trainside engine |
| `sd3_vllmomni_full_nccl_separate` | SD3 · vLLM-Omni engine · full-weight · NCCL sync · separate slabs |
| `qwen_vl_grpo_geo3k_mc_4x8` | Qwen-VL · GRPO · geo3k multiple-choice · 4 nodes × 8 GPUs |

Domain-specific trailing qualifiers extend the chain:

- **`pe/`** appends the reward: `pe_sglang_full_pickscore`, `pe_sglang_full_wise`.
- **`ar/`** (vision-language) appends dataset + task: `qwen_vl_grpo_geo3k_mc_4x8` (`geo3k` · multiple-choice).
- AR recipes (`ar/`) append the cluster shape `<N>x<G>` (nodes × GPUs): `..._4x8`.

## Adding or editing a recipe

Every recipe **must start with `# @package _global_`** on line 1. Recipes live in
a domain subdirectory, so without it Hydra would nest the whole config under the
domain key (e.g. `diffusion.num_devices`) and the entrypoint's top-level fields
would be missing. Cluster-local paths, model mounts, output dirs, and W&B identity
stay out of the YAML — pass them as env vars / CLI overrides; recipes read them
with `${oc.env:...}`.

1. Copy the closest existing recipe in the right domain directory.
2. Keep line 1 as `# @package _global_`; name the file per the schema above.
3. Keep every choice in YAML, instantiated by `_target_`; use `${oc.env:...}` only
   for deployment-specific paths and logging identity.
4. Before opening a PR, run the checks that match the files you touched:

```bash
# Compose the recipe and print the resolved config
python -m unirl.train_<entry> --config-name=<recipe-path> --cfg job --resolve

# Python syntax check
python -m compileall -q unirl

# Shell launcher syntax check
for f in examples/*.sh; do bash -n "$f"; done

# Lint and repository hooks
pre-commit run --all-files
```
