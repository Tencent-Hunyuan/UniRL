# Rollout

> **Where it fits:** the *rollout* step of the loop —
> **rollout** → reward → advantage → train → sync. In: a request `Sample` from
> the trainer. Out: a filled `Sample`. Full map: [`../README.md`](../README.md).

*Every rollout engine fills one `Sample`. Agentic scheduling and group assembly live
in the driver-side rollout manager.*

## What it is

`unirl.rollout` owns the rollout engines — the box that fills a typed `Sample` by
running a model pipeline and the SDE step kernels. The agentic engine runs the
model/environment turns for one trajectory. It does not compute reward or loss.

## Why it exists

The rollout can come from unrelated codebases — the in-process training
`Pipeline`, or a dedicated backend sampling in its own processes. For on-policy RL they
must walk a *numerically identical* trajectory, because the trainer replays the
rollout to recompute log-probs and any drift silently pushes the GRPO ratio off 1.0.
So this module is a **verification boundary**, not just a backend-hiding shim: it
pins one σ schedule on the generated `Part`'s sampling params and verifies what the
backend used (`engine/sigma_verify.py`). Each engine adapts its backend wire format
into canonical `Part` fields, so a dedicated server swaps in for the training model
without the loop noticing, and a mismatch crashes loudly instead of training on a
wrong objective.

## How it works

- **One generation interface.** `BaseRolloutEngine` (`engine/base.py`)
  is a `Remote` whose concrete engines implement synchronous `generate(sample)`;
  each returns one `Sample`. Batch engines dispatch `generate` with `DP_SCATTER`;
  `RolloutManager` launchers address one replica through `generate_on_slot`.
  Agentic `generate` is undecorated because the driver manager addresses one
  engine slot per trajectory; Ray actor concurrency lets the inner backend batch
  concurrent calls.
- **The typed boundary** (`../types/`). A `Sample` is an ordered chain of `Part`s.
  Each Part carries lineage ids, a raw `primitive`, an encoded `segment`, replay
  conditions, sampling params (including the σ schedule), and optional decoded
  media. Single-stage flows fill one generated Part; composed PE fills its chained
  AR and diffusion Parts.
- **The engines.** `trainside` (in-process — the train actor's pipeline *is* the
  sampler), `sglang_diffusion` (dedicated diffusion), `sglang` (dedicated AR), `vllm`
  (dedicated direct AR), `vllm_omni`
  (dedicated; BAGEL / HI3 / SD3 / HunyuanVideo / Qwen-Image / Qwen3-Omni), `fastvideo`
  (dedicated accelerated video sampling), and `composed` (chains an AR child + a
  diffusion child for prompt enhancement) are the seven single-turn engines.
  `agentic` wraps one of them with an environment to produce multi-turn
  trajectories; see the [engine matrix](#engine-matrix). Each diffusion engine consumes the Part's pinned sigmas verbatim
  and reads the same driver-authored `NoiseRecipe` (`../types/noise_recipe.py`),
  but realizes it differently: `trainside`, `sglang_diffusion` and `vllm_omni`
  resolve the recipe to an `x_T` tensor, so those three start a rollout from
  bit-identical noise; `fastvideo` cannot accept a tensor and instead derives
  per-sample seeds from the recipe's noise-group ids, so its noise matches only
  in grouping, not bit-for-bit. `forward_batch_size` bounds peak memory by
  slicing the Sample and concatenating the results.
- **Deployment modes:** *direct sampling* — the trainside engine, no `sync:`, the
  ratio is 1 on the first update; *separate* — a dedicated engine on its own GPUs
  plus a `sync:` block; *colocate* — a dedicated engine sharing GPUs with train,
  plus offload/onload and `sync:`; *anchored* — a dedicated engine hosted on one
  train-slab Worker (`rollout_anchor_device`, or `ar_rollout` + `dit_rollout`) and
  reached by `RemoteLoraWeightSync`. The mode does not pick how rollout outputs reach
  train: that is the cluster-wide `transport_kind`
  ([data plane](../distributed/README.md#how-it-works)).
- **Driver-side scheduling.** One `manager.RolloutManager` serves batch and
  agentic trainers. Its progress thread dispatches bounded work and observes
  readiness; trainer-thread collection resolves results, assembles agentic
  siblings, and applies the configured filter. Async batch trainers own training
  progress and publication cadence; the manager owns the published rollout version
  and preserves completed generations as FIFO batch chunks. `AgenticTrainer` collects
  one complete barrier batch per step. Async AR refills the same manager immediately
  after collection, so its existing progress thread overlaps the next generation
  with scoring and training. Publication and durable boundaries still quiesce that
  single manager before proceeding.

**Extending it:** a new single-turn engine adds `engine/<name>/config.py` (a
`BaseEngineConfig` whose `make_engine(**deps)` lazily imports and builds it) and
`engine/<name>/engine.py` (subclass `BaseRolloutEngine`, implement
synchronous generation over the whole-`Sample` contract — thread-safe for
concurrent callers if it should serve as an agentic inner, else serialized
internally — and dispatch `generate` with `DP_SCATTER`). A dedicated engine also
implements its weight-receive method and a matching `sync:` handler in
`../distributed/weight_sync`.

## Engine matrix

Layouts and `sync:` handlers below are the ones checked-in recipes use, not the
full set each class could accept.

| Engine package | Runs | Layouts in recipes | `sync:` in recipes | Example recipe |
| --- | --- | --- | --- | --- |
| `trainside` | The train actor's own `Pipeline`, in-process | Direct | None (forbidden) | [`sd3_trainside`](../../examples/diffusion/sd3/sd3_trainside.yaml) |
| `sglang` | AR/VLM through the `text` and `vlm` adapters | Colocate, separate (async only) | `TensorWeightSync`, `LocalLoraWeightSync`, `CkptEngineIPCWeightSync`, `NCCLWeightSync` | [`qwen3_grpo_4b_base_dapo_sglang_async`](../../examples/ar/qwen3_grpo_4b_base_dapo_sglang_async.yaml) |
| `sglang_diffusion` | Diffusion by `model_family`: `flux2_klein`, `hunyuan_video10`, `ltx2`, `qwen_image`, `qwen_image_edit_plus`, `sd3`, `wan21`, `wan22`, `z_image` (`flux` and `mochi` adapters have no recipe) | Colocate, separate | `LocalLoraWeightSync`, `TensorWeightSync`, `NCCLWeightSync`, `RemoteLoraWeightSync` | [`sd3_sglang_full_nccl_separate`](../../examples/diffusion/sd3/sd3_sglang_full_nccl_separate.yaml) |
| `vllm` | Direct text-only AR; startup rejects any model type except `qwen3_moe` | Colocate (requires `enable_fsdp_offload: true`) | `IPCWeightSync` (native engine) | [`qwen3_moe_grpo_30b_a3b_gsm8k_fsdp_vllm_tp4`](../../examples/ar/qwen3_moe_grpo_30b_a3b_gsm8k_fsdp_vllm_tp4.yaml) |
| `vllm_omni` | By `modality`: `bagel_t2i`, `bagel_it2i`, `sd3_t2i`, `qwen_image_t2i`, `hv15_t2v`, `hi3_ar_recaption`, `hi3_dit_recaption`, `qwen3_omni_thinker` (`hi3_t2i`, `hi3_it2i`, `hi3_i2t`, `hi3_t2t` adapters have no recipe) | Colocate, anchored, separate | `LocalLoraWeightSync`, `RemoteLoraWeightSync`, `NCCLWeightSync`, `TensorWeightSync`, `IPCWeightSync` (legacy ZMQ) | [`sd3_vllmomni`](../../examples/diffusion/sd3/sd3_vllmomni.yaml) |
| `fastvideo` | WAN 2.1 and WAN 2.2 A14B T2V ([details](engine/fastvideo/README.md#wan-support)) | Colocate | `CheckpointWeightSync` | [`wan21_t2v_dancegrpo_fastvideo`](../../examples/diffusion/wan21/wan21_t2v_dancegrpo_fastvideo.yaml) |
| `composed` | Coordinator: AR child, then diffusion child, for prompt enhancement. Recipes use `sglang` + `sglang_diffusion` | Colocate | One per track: `LocalLoraWeightSync`, `TensorWeightSync` | [`pe_sglang_full_pickscore`](../../examples/pe/pe_sglang_full_pickscore.yaml) |
| `agentic` | Coordinator: wraps one inner engine with an environment for multi-turn trajectories. The recipe uses `sglang` | Colocate | `TensorWeightSync` (`AgenticTrainer` rejects any other) | [`deep_research_search_judge`](../../examples/deep_research/deep_research_search_judge.yaml) |

## Sampling ownership

The driver decides what to sample; the engine executes it and proves what it used.

| Decision | Owner | Mechanism |
| --- | --- | --- |
| Sampling parameters | Recipe `sampling:` block | Canonical `DiffusionSamplingParams` / `ARSamplingParams` (`../types/sampling.py`), normalized at construction |
| Request stamping | Trainer | `Sample.fork(..., sampling_params=...)` places the parameters on each generated Part |
| SDE step indices | `DiffusionTrainer`, per rollout | `get_sde_indices(rollout_id)` is resolved to a concrete list before dispatch; `null` with no scheduler becomes every step |
| σ schedule | Engine, once | `ensure_sample_sigmas` fills `sigmas` only when unset, from the engine's schedule policy; `sigma_verify` checks the backend echo on `sglang_diffusion`, `vllm_omni`, and `fastvideo` |
| SDE kernel | Recipe `strategy:` | Injected into the engine as well as the pipeline; `fastvideo` refuses to boot without it |
| Per-request AR extras | Trainer `control` | `Part.control["ar"]`, for example `stop` |

SGLang reads AR sampling from the Sample only, and raises if it is missing. Direct
vLLM keeps fallback fields on `VLLMEngineConfig`, used only when a Part has no
sampling parameters; in-tree trainers always stamp them.

## Replay and anchors

The trainer replays each rollout at current weights, so an engine must return
everything replay needs: the `segment`, conditions, and the schedule it used.

- **`old_logp_source` picks the π_old anchor.** `rollout` trusts log-probs the
  engine emitted, so the engine must produce them from the same kernel as replay.
  `replay` recomputes them on the trainer once, before the update loop, and is
  required by `batch_replay_steps: true`. Per-algorithm behavior lives in
  [`algorithms`](../algorithms/README.md#gotchas).
- **EMA rollout depends on the engine.** For `requires_ema_rollout` algorithms,
  trainside swaps EMA weights onto the train module around generation; a dedicated
  engine receives them through weight sync.
- **`rollout_anchor_device` is unrelated.** It places an engine on a device; it has
  nothing to do with the π_old anchor.

## Engine anatomy, and adding a model to an existing engine

Engine dirs use two layouts. The compact engines (`trainside`, `fastvideo`,
`composed`, `agentic`) contain `config.py` + `engine.py`. Server-backed engines
(`sglang`, `sglang_diffusion`, `vllm_omni`) also carry `adapters/`
(per-family/modality wire-format translation), `backends/` (server process
management), `utils/`, `weight_sync.py`, and a
runtime-patch dir for the pinned upstream (`sglang_diffusion/_patches/`,
`vllm_omni/patches/`). `vllm_omni` additionally carries worker-subprocess code
(`pipelines/`, `worker/`) and deployment configs (`deploy_configs/`). `fastvideo`
also carries `_patches/`, and `vllm` carries `runtime.py`, `sampling.py`, and
`worker_extension.py`.

Model onboarding is per-engine, and the adapter file is usually **not** the whole
change surface:

- **`sglang` (AR/VLM):** onboarding is normally config-only: text models use the
  `text` adapter, while `image_token` selects `vlm`. Add and register a new adapter
  only for a genuinely new wire shape, extending `TextLMAdapter` or `VLMAdapter`
  and importing it in `adapters/__init__.py`.
- **`sglang_diffusion`:** add `adapters/<family>.py` extending `ImageAdapter` or
  `VideoAdapter`, register it by `model_family`, and import it in
  `adapters/__init__.py`. New condition fields that cross the wire also need an
  entry in `_COND_FIELDS` and either `_POS_MAP` / `_NEG_MAP` or an explicit copy
  branch in `_copy_conditions`; add `_patches/hijack.py` wiring only when the
  model needs a new upstream patch.
- **`vllm_omni`:** add an `adapters/<family>.py` binder (keyed by modality),
  register it, import it in `adapters/__init__.py`, and add the appropriate boot
  YAML under `deploy_configs/`. DiT families additionally need a worker-side
  `pipelines/<model>/pipeline.py`; if the AR/DiT worker needs new behavior, add a
  `worker/` extension or `patches/compat_<model>.py`.

## SGLang AR knobs

The 4B dense Qwen3 AR recipes (`examples/ar/qwen3_*_sglang*.yaml`) share a
**colocate full-FT** `engine_kwargs` baseline. Async/separate, larger-model,
MoE, and LoRA recipes deliberately deviate as described below and in their
inline comments. Typed `SGLangEngineConfig` fields overlay `engine_kwargs`;
reserved ports always win. Keys that are not live SGLang `ServerArgs` fields
warn at boot (or raise if `UNIRL_SGLANG_STRICT_SERVER_ARGS=1`) — see
[`engine/README.md`](engine/README.md).

**4B dense colocate full-FT baseline** (FSDP train shard time-shares the GPU;
TensorWeightSync):

| Knob | Preset | Why |
|---|---|---|
| `mem_fraction_static` | `0.3` | Lower SRT KV reservation so FSDP can all-gather full dense weights. `server_intent()` defaults to `0.88` if omitted — too high for colocate. |
| `enable_lora` | `false` | TensorWeightSync pushes full dense weights; a LoRA pool would be the wrong receive path. |
| `cuda_graph_max_bs_decode` | `16` | CUDA graph stays on (`disable_cuda_graph: false`); SGLang otherwise auto-tunes the capture cap from GPU memory and TP size, often reserving much larger buffers that fight the weight push. |
| `skip_server_warmup` | `true` | Skip SRT startup warmup on the 4B colocate path; this knob does not control `wake_up()`. |
| `attention_backend` | `triton` | Matches the in-tree 4B full-FT recipes. |

**Larger / MoE colocate:** memory and graph settings are recipe-specific
(`mem_fraction_static` ranges from `0.15` to `0.5`; graph caps include `64` and
`128`, and one VLM MoE recipe disables graphs). Several MoE/CPPO recipes set the
typed `skip_server_warmup: false`, which wins over `engine_kwargs`, so startup
warmup runs. The VLM MoE geo3k recipe also leaves `attention_backend` to
SGLang's platform selection.

**Async / separate** (`*_sglang_async.yaml`): the engine owns the GPU (no colocated
FSDP shard). Raise `mem_fraction_static` to `0.8` and leave the other 4B baseline
keys as-is; keep headroom for NCCL weight-receive buffers. Sync is
`NCCLWeightSync`.

**LoRA colocate** (`*_sglang_lora.yaml`): `enable_lora: true` plus the SGLang LoRA
pool knobs. This is the exception to the full-FT `enable_lora: false` receive path
(`LocalLoraWeightSync` instead of TensorWeightSync). Memory numbers are recipe-specific.

**Reserved ports:** `SGLangPorts.reserve()` selects candidate HTTP `port` and
`nccl_port` values by binding temporary sockets on the engine's node, then
closes those sockets so SGLang can bind (the usual bind-to-zero TOCTOU gap
remains). Do not set `port` / `nccl_port` in `engine_kwargs` — the selected
values overwrite them. The retired 35535 cap is unnecessary on 0.5.19:
SGLang derives `grpc_port = port + 10000` only when gRPC mode is explicitly
enabled, which UniRL's rollout backend does not do.

## Gotchas

- **SGLang diffusion tensor LoRA accepts standard PEFT LoRA only.** A global
  `lora_alpha` is supported; rsLoRA, DoRA, per-layer alpha patterns, and PEFT
  auxiliary features fail closed. `use_lora=true` selects adapter-only sync for
  the engine lifetime. Full-weight sync requires `use_lora=false`; a trainer
  using LoRA must merge the adapter before pushing those full weights.
- **A colocated SGLang engine needs `enable_memory_saver`.** Without it SGLang
  installs a no-op `TorchMemorySaverAdapter`: `release_memory_occupation` reports
  success and frees nothing. Add `enable_weights_cpu_backup: true` to move weights
  to host across the sleep. `AgenticTrainer` rejects a colocated engine without the
  flag; `ARTrainer` warns.
- **Size `mem_fraction_static` from the wake peak, not the rollout peak.** In
  `AgenticTrainer` the engine wakes before `backend.offload()` releases the train
  state, and `torch_memory_saver`'s `resume()` cannot reuse blocks held by torch's
  caching allocator. Take the peak over several rollouts; turn counts vary run to run.
- **KV offload (HiCache, Mooncake) only helps above the radix-cache knee.** With
  `R = concurrency × per-trajectory context / KV pool`, prefix reuse stays flat up
  to `R ≈ 1` and drops beyond it; below that a host tier has nothing to recover.
  Raising `mem_fraction_static` lowers `R`, so re-measure it before adding a tier.
- **Never recompute σ inside an engine** — the generated Part's pinned sigmas are
  the single source of truth; `engine/sigma_verify.py` checks the backend echo (it
  guards the GRPO log-prob ratio).
- **vLLM-Omni AR logprobs fail closed.** `build_ar_segment` requires one
  non-empty dict logprob step per emitted token and raises `RuntimeError` on
  missing, non-dict, or length-mismatched logprobs instead of padding or
  truncating with `0.0`.
- **Batch `generate` must dispatch `DP_SCATTER`.** Agentic is the intentional
  exception: its undecorated method is reached through one `Handle.slot(...)`.
- **Direct sampling forbids a `sync:` block; dedicated requires one.** The trainside
  engine also can't live on a `layout: separate` slab — the trainer `__init__` raises
  before spawning actors.
- **Quiesce before weight sync / eval / checkpoint on async paths** —
  `RolloutManager.quiesce()` pauses dispatch, drains batch work, and cooperatively
  suspends agentic trajectories at turn boundaries. `sync_weights()` rejects queued
  or in-flight work, pushes weights, and publishes the optimizer-update
  `output_version`; immutable buffered groups keep their original provenance and
  remain subject to the configured filter. Eval/checkpoint admission still requires
  an empty manager. Launch and scoring order remains trainer policy. Versions, lag
  budgets, and batch alignment live in the
  [trainer lifecycle](../trainer/README.md#the-shared-lifecycle).
- **A resolve/route failure poisons the `RolloutManager`** — samples may already be
  lost, so every later call (including `empty` / `counts`) re-raises the original
  error rather than reporting clean state; only `close()` stays safe.
- **Reward/advantage methods are not engine code** — `Part.compute_advantages` and
  `Sample.propagate_rewards` are called by the trainer after scoring. An engine
  fills generation fields such as `segment`, `conditions`, `primitive`, and
  `media_preview`; rewards arrive later from `RewardService`.
