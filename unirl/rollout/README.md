# Rollout

> **Where it fits:** the *rollout* step of the loop —
> **rollout** → reward → advantage → train → sync. In: a request `Sample` from
> the trainer. Out: a filled `Sample`. Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/rollout-engines-new.png" alt="UniRL rollout engines selected by _target_ across direct, separate, and colocated deployment modes" width="100%">
</div>

*Every rollout engine fills one `Sample`. Agentic scheduling and group assembly live
in the driver-side rollout manager.*

## What it is

`unirl.rollout` owns the rollout engines — the box that fills a typed `Sample` by
running a model pipeline and the SDE step kernels. The agentic engine runs the
model/environment turns for one trajectory. It does not compute reward or loss.

## Why it exists

The rollout can come from two unrelated codebases — the in-process training
`Pipeline`, or the SGLang fork sampling in its own subprocess. For on-policy RL they
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
  each returns one `Sample`. Batch engines dispatch `generate` with `DP_SCATTER`.
  Agentic `generate` is undecorated because the driver manager addresses one
  engine slot per trajectory; Ray actor concurrency lets the inner backend batch
  concurrent calls.
- **The typed boundary** (`../types/`). A `Sample` is an ordered chain of `Part`s.
  Each Part carries lineage ids, a raw `primitive`, an encoded `segment`, replay
  conditions, sampling params (including the σ schedule), and optional decoded
  media. Single-stage flows fill one generated Part; composed PE fills its chained
  AR and diffusion Parts.
- **The engines.** `trainside` (in-process — the train actor's pipeline *is* the
  sampler), `sglang_diffusion` (dedicated diffusion), `sglang` (dedicated AR), `vllm_omni`
  (dedicated; BAGEL / HI3 / SD3 / HunyuanVideo), `fastvideo` (dedicated accelerated video
  sampling), and `composed` (chains an AR child + a
  diffusion child for prompt enhancement) are the six single-turn engines.
  `agentic` wraps one of them with an environment to produce multi-turn
  trajectories. Each diffusion engine consumes the Part's pinned sigmas verbatim
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
  plus offload/onload and `sync:`.
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

## Engine anatomy, and adding a model to an existing engine

Engine dirs use two layouts. The compact engines (`trainside`, `fastvideo`,
`composed`, `agentic`) contain `config.py` + `engine.py`. Server-backed engines
(`sglang`, `sglang_diffusion`, `vllm_omni`) also carry `adapters/`
(per-family/modality wire-format translation), `backends/` (server process
management), `utils/`, `weight_sync.py`, and a
runtime-patch dir for the pinned upstream (`sglang_diffusion/_patches/`,
`vllm_omni/patches/`). `vllm_omni` additionally carries worker-subprocess code
(`pipelines/`, `worker/`) and stage boot configs (`stage_configs/`).

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
  YAML under `stage_configs/`. DiT families additionally need a worker-side
  `pipelines/<model>/pipeline.py`; if the AR/DiT worker needs new behavior, add a
  `worker/` extension or `patches/compat_<model>.py`.

## SGLang AR knobs

Qwen3 AR recipes (`examples/ar/qwen3_*_sglang*.yaml`) share one **colocate
full-FT** `engine_kwargs` preset. Async/separate and larger-model recipes keep
the same keys and change the memory numbers. Typed `SGLangEngineConfig` fields
overlay `engine_kwargs`; reserved ports always win. Keys that are not live
SGLang `ServerArgs` fields warn at boot (or raise if
`UNIRL_SGLANG_STRICT_SERVER_ARGS=1`) — see [`engine/README.md`](engine/README.md).

**Colocate full-FT preset** (FSDP train shard time-shares the GPU; TensorWeightSync):

| Knob | Preset | Why |
|---|---|---|
| `mem_fraction_static` | `0.3` | Lower SRT KV reservation so FSDP can all-gather full dense weights. `server_intent()` defaults to `0.88` if omitted — too high for colocate. |
| `enable_lora` | `false` | TensorWeightSync pushes full dense weights; a LoRA pool would be the wrong receive path. |
| `cuda_graph_max_bs` | `16` | CUDA graph stays on (`disable_cuda_graph: false`); SGLang's default `256` captures buffers that fight the weight push. |
| `skip_server_warmup` | `true` | Skip SRT warmup on every colocated boot/wake. |
| `attention_backend` | `triton` | Matches the in-tree 4B full-FT recipes. |

**Async / separate** (`*_sglang_async.yaml`): the engine owns the GPU (no colocated
FSDP shard). Raise `mem_fraction_static` to `0.8` and leave the other preset keys
as-is; keep headroom for NCCL weight-receive buffers. Sync is `NCCLWeightSync`.

**LoRA colocate** (`*_sglang_lora.yaml`): `enable_lora: true` plus the SGLang LoRA
pool knobs. This is the exception to the full-FT `enable_lora: false` receive path
(`LocalLoraWeightSync` instead of TensorWeightSync). Memory numbers are recipe-specific.

**Reserved ports:** `SGLangPorts.reserve()` binds HTTP `port` and `nccl_port` on
the engine's node. Server port is capped at 35535 because SGLang derives
`grpc_port = port + 30000`. Do not set `port` / `nccl_port` in `engine_kwargs` —
the reserved sockets overwrite them.

## Gotchas

- **Never recompute σ inside an engine** — the generated Part's pinned sigmas are
  the single source of truth; `engine/sigma_verify.py` checks the backend echo (it
  guards the GRPO log-prob ratio).
- **Batch `generate` must dispatch `DP_SCATTER`.** Agentic is the intentional
  exception: its undecorated method is reached through one `Handle.slot(...)`.
- **Direct sampling forbids a `sync:` block; dedicated requires one.** The trainside
  engine also can't live on a `layout: separate` slab — `_build_rollout` raises.
- **Quiesce before weight sync / eval / checkpoint on async paths** —
  `RolloutManager.quiesce()` pauses dispatch, drains batch work, and cooperatively
  suspends agentic trajectories at turn boundaries. `sync_weights()` rejects queued
  or in-flight work, pushes weights, and publishes the optimizer-update
  `output_version`; immutable buffered groups keep their original provenance and
  remain subject to the configured filter. Eval/checkpoint admission still requires
  an empty manager. Launch and scoring order remains trainer policy.
- **A resolve/route failure poisons the `RolloutManager`** — samples may already be
  lost, so every later call (including `empty` / `counts`) re-raises the original
  error rather than reporting clean state; only `close()` stays safe.
- **Reward/advantage methods are not engine code** — `Part.compute_advantages` and
  `Sample.propagate_rewards` are called by the trainer after scoring. An engine
  fills generation fields such as `segment`, `conditions`, `primitive`, and
  `media_preview`; rewards arrive later from `RewardService`.
