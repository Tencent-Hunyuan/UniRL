# experimental/train_inference_parity

Owner: UniRL Qwen3-MoE parity maintainers

This package incubates exact rollout-versus-old-policy and
rollout-versus-gradient-replay log-probability contracts without shipping
monkey patches or experimental kernels in the UniRL wheel.

It contains only the bitwise experiment. A normal baseline uses the original
UniRL Qwen3-MoE AR entry point and does not load this package or its plugin.

## Scope

The first model profile is `Qwen3-30B-A3B`:

```text
experimental model id: qwen3_moe_30b_a3b
actor scoring: full prompt+response no-grad forward
comparison dtype: FP32
required old-policy metrics: max_absdiff=0, K3 mean/max=0, torch.equal=true
required gradient-replay metrics: max_absdiff=0, K3 mean/max=0, torch.equal=true
```

The supported topology is an FSDP actor world of four with one vLLM TP4
rollout engine.

## Install

Install the normal UniRL vLLM environment, then install the experimental
plugin package:

```bash
pip install -e experimental/train_inference_parity/vllm_plugin
```

The installed distribution is `unirl-train-inference-parity-vllm`; its vLLM
general-plugin entry point is `unirl_train_inference_parity`.

## Launch

A Ray cluster must already be running. From the repository root:

```bash
export QWEN3_MOE_PATH=/dev/shm/Qwen3-30B-A3B
export DATA_PATH=/tmp/unirl_gsm8k_4_7.jsonl
export EVAL_DATA_PATH=/tmp/unirl_gsm8k_4_7.jsonl

python -m experimental.train_inference_parity.run \
  --config-name=qwen3_moe_30b_a3b_fsdp_tp4
```

Hydra overrides can enable scheduler features:

```bash
rollout.config.engine_kwargs.enable_chunked_prefill=true
rollout.config.engine_kwargs.enable_prefix_caching=true
```

## Profiles

`public_reference` is the default:

- vLLM batch-invariant linear, norm, reductions and softmax
- vLLM native FA3 and MoE permute
- Torch fixed-order MoE combine and backward recompute
- NCCL tensor-parallel collectives

## Package boundaries

- `unirl/` never imports this package.
- Recipes under this package may target `experimental.train_inference_parity.*`.
- The nested vLLM plugin distribution does not import `unirl` or sibling
  experimental packages.
- Common code needed by a second experimental package must graduate into core;
  it is not imported sideways.

## Verification

| Profile | Topology | Response | Chunked prefill | Prefix cache | Update/reload | Status |
|---|---|---:|---:|---:|---:|---|
| public_reference | FSDP4 / vLLM TP4 | 1024 | on | on | two steps + reload | PASS |

The PASS row was run from this experimental package with
`VLLM_PLUGINS=unirl_train_inference_parity` and no external UniMatch plugin.
Both no-grad old-policy replay and gradient-bearing replay reported:

```text
token_count=4096
torch_equal=True
mismatch_count=0
max_absdiff_fp32=0
k3_mean=0
k3_max=0
```

```text
rollout_replay_logp_absdiff_mean=0
rollout_replay_logp_absdiff_max=0
rollout_replay_k3_mean=0
rollout_replay_k3_max=0
```
