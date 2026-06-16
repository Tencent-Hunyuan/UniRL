# Examples

Self-contained Hydra recipes — one YAML per experiment. A recipe is the single
source of truth for a run: model, algorithm, rollout engine, placement, reward,
weight sync, and batch geometry, each instantiated directly by `_target_` (no
Hydra config-group overrides). Recipes are grouped into one directory per trainer
domain; select one with `--config-name=<domain>/<recipe>` (drop the `.yaml`).

> This directory replaces the old top-level `recipes/` tree.

## Domains & entrypoints

Each domain maps to one entrypoint. The **default recipe** is that entrypoint's
built-in `config_name` — a safe place to start.

| Domain | Entrypoint | Default recipe (start here) | Models |
|---|---|---|---|
| [`diffusion/`](diffusion/) | `python -m unirl.train_diffusion` | `diffusion/sd3/sd3_trainside` | `sd3`, `qwen_image`, `flux2_klein`, `wan21`, `wan22`, `hunyuan_video`, `hunyuan_video15` |
| [`ar/`](ar/) | `python -m unirl.train_ar` | `ar/qwen_vl_grpo_geo3k_mc_4x8`, `ar/qwen3_drpo_4b_base_dapo_sglang` | `qwen_vl` (vision-language), `qwen3` (text-only) |
| [`pe/`](pe/) | `python -m unirl.train_pe` | `pe/pe_trainside_pickscore` | `pe` (Qwen3 rewriter + SD3, PickScore/WISE reward) |
| [`unified_model/`](unified_model/) | `python -m unirl.train_unified_model` | `unified_model/hi3_vllmomni` | `hi3` (HunyuanImage3, unified AR + diffusion) |

## Running a recipe

The bash launchers live in this directory. The first argument is the
domain-qualified recipe name (passed to Hydra as `--config-name`); any extra args
are forwarded verbatim as Hydra overrides. `ENTRY` selects a non-diffusion
entrypoint (`train_ar` / `train_pe` / `train_unified_model`); the default is
`train_diffusion`.

```bash
# 0. Compose-check first — verifies the config composes and every ${oc.env:...} resolves
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside --cfg job --resolve

# 1. Single node
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside
ENTRY=train_ar bash examples/run_experiment_single_node.sh ar/qwen_vl_grpo_geo3k_mc_4x8
ENTRY=train_pe  bash examples/run_experiment_single_node.sh pe/pe_trainside_pickscore

# 2. Multi-node (taiji)
bash examples/run_experiment_multinode_taiji.sh diffusion/sd3/sd3_sglang_rollout_colocate

# 3. Or invoke an entrypoint directly, without the launchers
python -m unirl.train_diffusion --config-name=diffusion/sd3/sd3_trainside num_devices=8
```

Pass cluster-local paths and W&B identity through env vars (`PRETRAINED_MODEL`,
`DATA_PATH`, `EVAL_DATA_PATH`, `REPORT_TO_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`).
The mooncake-backed recipe (`*_tq_mooncake`) needs its metadata server up first —
start it on the head node with `bash examples/mooncake_master.sh start` before launching.

To save and resume checkpoints and export them to Hugging Face, append the
`+save_interval` / `+save_dir` / `+load_dir` overrides (diffusion/ar/unified
trainers; the hi3 meta-init recipe is not yet supported) — the full
train → resume → export → upload lifecycle is in
[Checkpointing](../unirl/trainer/README.md#checkpointing).

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
| `model` | required, first | `sd3`, `qwen_image`, `flux2_klein`, `wan21`, `wan22`, `hunyuan_video`, `hunyuan_video15`, `qwen_vl`, `qwen3`, `hi3` | never |
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
python -m unirl.train_<entry> --config-name=<domain>/<recipe> --cfg job --resolve

# Python syntax check
python -m compileall -q unirl

# Shell launcher syntax check
for f in examples/*.sh; do bash -n "$f"; done

# Lint and repository hooks
pre-commit run --all-files
```
