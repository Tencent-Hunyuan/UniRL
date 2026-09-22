# experimental/train_inference_parity

Owner: UniRL Qwen3-MoE parity maintainers

This package incubates an exact rollout-versus-gradient-replay selected-token
log-probability contract without shipping experimental kernels in the UniRL
wheel. A normal baseline does not load this package or its nested vLLM plugin.

## Current status: UNVERIFIED, not PASS

There is no checked-in verification artifact produced by the current recipe,
so this repository makes no PASS claim. The recipe uses the dedicated
`experimental.train_inference_parity.sync.ParityIPCWeightSync` evidence adapter
over UniRL's production vLLM native IPC sync and will fail rather than falling
back to a different transport.

A future result may be called PASS only when the generated JSON artifact has
`status: "PASS"` and contains both `initial_checkpoint` and
`post_update_reload`. Console output, W&B history, and a first-rollout-only run
are not substitutes.

## Frozen target contract

- Model: reviewed local mirror of `Qwen/Qwen3-30B-A3B` revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
- Topology: one FSDP actor world of 4 and one direct vLLM TP4 engine on exactly
  4 visible GPUs
- Software: Python 3.13, vLLM 0.27.0, Torch 2.13.0, Transformers 5.6.0
- Actor/model compute: BF16; selected-token comparison and stored log-probs:
  FP32
- Sampling: temperature 1, top-p 1, top-k 0, processed log-probs
- Lengths: 256 prompt tokens plus 1024 response tokens; actor and rollout must
  both use suffix-preserving (left) truncation
- Data: complete GSM8K JSONL, shuffled with OS entropy (`seed: null`); each
  rollout consumes the next non-repeating batch until the epoch rolls over
- Updates: one optimizer update per rollout; two rollouts total

Bitwise means the selected-token FP32 rollout and differentiable replay
log-probs on this frozen matrix. It does not claim bitwise gradients, optimizer
state, or portability to other GPUs, drivers, kernels, or package versions.

`public_reference` is the only profile constant. There is intentionally no
provider registry or provider environment-variable map for one implementation.

Numerical gotchas:

- The FSDP actor is not actually tensor-parallel. Its exact MoE path reproduces
  vLLM TP4's column/row slicing and concatenation order. `train_tp` has no code
  default, the runtime contract checks divisibility, and actor/vLLM call the
  same nested-plugin MoE combine precision function.
- The exact context remains active through loss and backward so non-reentrant
  activation-checkpoint recomputation selects the same forward providers.
  RMSNorm, MoE, attention, and log-softmax still use documented Torch surrogate
  backward formulas; gradients must be finite but are not bitwise claims.
- Actor monkey patches require `UNIRL_PARITY_ENABLE=1` and a dedicated process.
  Recoverable Python symbols retain their originals. PyTorch does not support
  unregistering the CUDA ATen overrides, so process exit is the unload boundary.

### Verification matrix

| Recipe | Hardware target | Driver/runtime | Python packages | Git/artifact | Status |
|---|---|---|---|---|---|
| `qwen3_moe_30b_a3b_fsdp_tp4.yaml` | 4 × NVIDIA H20 96 GB, one node | driver 535.161.08 + CUDA-13 forward compatibility, CUDA 13.0 runtime; NCCL recorded at run time | Python 3.13, Torch 2.13.0, Transformers 5.6.0, vLLM 0.27.0 | must be the clean HEAD recorded in the generated JSON | **UNVERIFIED** |

This row remains UNVERIFIED until a repository-generated artifact is checked
in or linked. It is a target matrix, not a historical PASS claim.

## Install and preflight

From a clean repository checkout on the target single-node four-GPU host:

```bash
uv sync --extra vllm --extra train
uv pip install -e "experimental/train_inference_parity/vllm_plugin[verified]"

export QWEN3_MOE_PATH=/dev/shm/Qwen3-30B-A3B
export QWEN3_MOE_REVISION=ad44e777bcd18fa416d9da3bd8f70d33ebb85d39
export DATA_PATH=/path/to/gsm8k/train.jsonl
export EVAL_DATA_PATH=/path/to/gsm8k/test.jsonl
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run ray start --head --num-gpus=4
```

Driver 535 requires the host's CUDA-13 forward-compatibility library for the
CUDA-13.0 wheel stack. Configure that library before launch; the artifact
records the driver, runtime, and NCCL versions actually observed.

`run.py` constructs one `ParityRuntimeContract`, installs its complete explicit
environment, then aggregates topology, precision, sampling, length, version,
GPU-count, trust, data/model-path, and Ray-preinitialization errors before
starting a model or connecting to Ray. In particular, Ray must not already be
initialized in the driver process.

The lifecycle sets `UNIRL_PARITY_ENABLE=1`, the vLLM plugin allowlist,
deterministic CUDA/NCCL flags, the model/profile/patch manifest, and propagates
the same values into every Ray role. `trust_remote_code=true` is explicit on
both actor and rollout; use only a reviewed local checkpoint.
The plugin is a complete no-op when the enable flag is absent. When enabled it
requires strict mode, the complete ordered patch set, the three pinned package
versions, and the verified private-symbol signatures; it does not silently
skip a missing provider.

## Quick smoke (no parity claim)

These checks validate imports and the plugin's disabled no-op behavior. They do
not replace the TP4 run:

```bash
uv run python -m compileall -q experimental/train_inference_parity \
  unirl/rollout/engine/vllm unirl/distributed/weight_sync
UNIRL_PARITY_ENABLE=0 uv run python -c \
  'import unirl_train_inference_parity_vllm as p; assert p.register() == ()'
```

## Full two-phase run

On the frozen target host:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m experimental.train_inference_parity.run \
  --config-name=qwen3_moe_30b_a3b_fsdp_tp4
```

Defaults:

- W&B disabled; opt in with `logging.report_to_wandb=true`
- vLLM native packed-IPC sync with a 640 MiB bucket, 2 GiB GPU safety margin,
  and a Gloo control timeout (2400s) that outlives the 1800s weight-update RPC
- atomic JSON artifact:
  `outputs/train_inference_parity/qwen3-moe-tp4.json` (override
  `UNIRL_PARITY_ARTIFACT_PATH`)

`ignore_eos=true` is a verification choice that forces the 1024-token boundary
and avoids EOS-dependent cases. It is not a recommended production RL default.

## Verification artifact

`verification.py` validates phase metrics and publication evidence;
`artifacts.py` owns provenance and atomic JSON persistence. The schema records:

- git commit and tracked-file dirty state
- software, CUDA/driver/NCCL, and per-GPU hardware
- full model-tree and data-file SHA-256 fingerprints
- all numerical flags from the runtime contract
- the vLLM worker's strict before/after symbol-provider manifest
- canonical parity metrics, replay token count, grad norm, optimizer updates,
  and parameter-change evidence for each phase
- the native IPC publication manifest, per-TP-worker receipts, and sampled
  Actor parameter-change fingerprint

The canonical metric names are `token_count`, `torch_equal_fp32`,
`mismatch_count`, `max_absdiff_fp32`, `k3_mean`, and `k3_max`.

## Integration requirements

1. Keep the recipe on `ParityIPCWeightSync`, which subclasses the production
   `IPCWeightSync` data plane and adds evidence only.
2. Preserve `record_initial_actor_fingerprint()`, `mark_actor_updated()`, and
   `verification_receipts()` on the parity sync adapter. Actor-change evidence
   samples large structural shards (embed, Q/O, MoE, LM head), not the smallest
   RMSNorms. Each publication must include the native structural manifest, one
   receipt for every TP rank, matching publication/model identities, unique CUDA
   UUIDs, `parameter_changed: true`, and `prefix_cache_reset: true`.
3. `TrainInferenceParityGRPO` must reach a pre-replay error barrier before any
   rank enters FSDP/vLLM collectives, then the exact log-prob gate after replay.
4. Preserve the plugin's reload lifecycle: clear the derived MoE column cache
   before vLLM sleep, leave the stock expert Parameter loaders unobstructed, and
   load the custom column-parallel `o_proj` directly into its CuMem allocation
   instead of passing it through vLLM's generic meta-staged reload.
5. Keep actor and direct-vLLM prompt handling suffix-preserving at the same
   configured `max_prompt_length`; do not re-enable tokenizer-default
   chat-template truncation before the shared engine limit.
6. Each phase must observe `batch_size × samples_per_prompt × max_new_tokens`
   replay tokens (`4 × 8 × 1024 = 32768` on the frozen recipe).
7. Run from a clean source worktree on the frozen matrix and publish the
   resulting artifact before changing this status to PASS. Generated `outputs/`
   and plugin egg-info do not count as dirty source.

## Package boundaries

- `unirl/` never imports this package.
- Recipes under this package may target
  `experimental.train_inference_parity.*`.
- The nested plugin imports only UniRL's stable vLLM worker-extension seam to
  attach its runtime manifest; core UniRL never imports the plugin.
- Common code needed by a second experiment graduates into core rather than
  being imported sideways.

## Gotchas

- The training-side PR requires the separately installed vLLM plugin for its
  shared numerical providers. Its core UniRL changes remain inert on baseline
  recipes, but the experimental recipe needs both PRs.
- GRPO loss construction is shared with the baseline implementation. The
  experiment accepts only rollout-sourced anchors and keeps exact mode active
  through backward, including activation-checkpoint recomputation.
- A fixed pre-replay barrier synchronizes local setup failures before FSDP
  forwards. It cannot recover a worker that dies inside a model collective.
- Every micro-batch must report all canonical metrics. Missing evidence is an
  error, not a micro-batch that can be omitted from aggregation.
- Source provenance includes individual untracked Python/config files inside
  new directories. Untracked notes are excluded; tracked changes always count.
- The frozen 256/1024 token lengths cannot be shortened while retaining this
  recipe's verification claim. W&B logging may be enabled independently.
- The Qwen3 meta-init path resolves weight shards with `model_revision`, just
  like config and model loading. Local checkpoints still need external review
  against that revision; setting the revision does not authenticate local bytes.
