# Rollout engines

> **Where it fits:** the engine layer of the *rollout* step. In: a request
> `Sample`. Out: a filled `Sample`, or a trajectory `list[Sample]`.
> Full map: [`../README.md`](../README.md).

*`base.py` holds the worker-side contract every per-backend subpackage implements;
driver-side scheduling lives in [`../manager/`](../manager/).*

## What it is

The engine ABCs and the two cross-engine helpers
(`ports.py`, `sigma_verify.py`). The per-backend engines live in the subpackages:
`trainside`, `sglang`, `sglang_diffusion`, `vllm`, `vllm_omni`, `fastvideo`, `composed`,
and the `agentic` coordinator.

## Why it exists

Every backend has to be swappable without the training loop noticing, and the
trainer replays the rollout — so any numerical drift silently pushes the GRPO ratio
off 1.0. These modules are where the two contracts that prevent that are written
down: what a worker-side engine must implement, and what the driver may assume.

## How it works

- **Worker side** (`base.py`). `BaseRolloutEngine` is the ABC for per-backend and
  coordinator engines. Engines complete construction
  in `__init__` — there is no separate initialize step.
- **Driver side** (`../manager/`). `RolloutManager` schedules over per-slot
  launchers; launch ceilings and step loops live in the trainers.
- **σ round-trip** (`sigma_verify.py`). The engine pins the gen Part's sigmas via
  `ensure_sample_sigmas`, forwards them, and asserts the worker echoed back the
  exact schedule it sent.
- **Ports** (`ports.py`). A `ReservedPorts` subclass declares one `int` field per
  port its subprocess needs — declaration order is reservation order. The engine
  reserves at the last responsible moment, in its ctor on its own node, right
  before the spawn. Bind-to-zero de-synchronizes colocated engines with no
  builder-side `base + rank * stride` math.

**Extending it:** subclass `BaseRolloutEngine`, implement synchronous generation
over the whole-`Sample` contract, and dispatch `generate` with `DP_SCATTER`. A
dedicated engine also implements its weight-receive method and a matching `sync:`
handler in `../../distributed/weight_sync`.

## Direct vLLM

`vllm` runs one vLLM instance per TP group in a spawned interpreter, reached over a
pipe, so vLLM's imports stay out of the Worker process.

- **Narrow by construction.** Text-only, PP=1, EP=1, and startup rejects any model
  type other than `qwen3_moe`. `runtime.py` requires exactly `vllm==0.28.0`.
- **Native IPC sync only.** The engine has no NCCL, tensor, or LoRA receive path;
  `IPCWeightSync` selects vLLM's native transfer engine for it
  ([weight sync](../../distributed/weight_sync/README.md)). A sync needs
  `enable_fsdp_offload: true`.
- **Sleep can discard weights.** Native sync sets sleep level 2, so `ARTrainer`
  calls `preserve_weights_for_next_sleep` when the next phase will not sync
  (`trainer/rollout_sleep.py`).
- **Broken means broken.** After a failed publication the engine is poisoned rather
  than retried, because the live weights may be partial.

## Gotchas

- **`__init__.py` must import nothing.** Importing an engine config must not pull
  SGLang, vLLM, or FastVideo; `make_engine` imports the backend lazily.
- **Port reservation is a hint, not a contract** — the sockets are closed
  immediately after binding so the subprocess can bind them itself, which leaves
  the usual bind-to-zero TOCTOU gap. Accepted deliberately.
- **Never hardcode the σ scale.** `sigma_verify` detects it dynamically; dividing
  by 1000 breaks on any model whose `num_train_timesteps` is not 1000.
- **An engine that will serve as an agentic inner must make `generate` safe for
  concurrent callers** — the agentic coordinator drives one trajectory per actor
  call.
- **Unknown SGLang `engine_kwargs` are dropped at boot.** HTTP/native backends
  filter intent against live `ServerArgs`. Typos used to fail silently; they now
  warn, or raise if `UNIRL_SGLANG_STRICT_SERVER_ARGS=1`. UniRL-only keys
  (`concurrency`, `advertise_host`, `health_timeout_s`) are expected drops.
  Recipe-facing colocate/async knobs live in [`../README.md`](../README.md)
  (SGLang AR knobs).
- **SGLang LoRA serialization is topology-sensitive.** TP1 keeps
  `MultiprocessingSerializer` and requires the server to inherit the sender's
  multiprocessing authkey. TP>1 cannot broadcast its one-shot file descriptors, so
  `sglang/backends/http.py` sends inlined base64 pickle bytes instead. Re-check the
  `SafeUnpickler` allowlist and broadcast semantics on a SGLang bump.
- **`rl_on_policy_target` silently switches on deterministic sampling.** SGLang
  0.5.19 derives `enable_deterministic_inference` from it, and every request
  that carries no `sampling_seed` then defaults to seed 42 — so one `n=8` request
  comes back as eight token-identical completions. The adapters therefore send
  deterministic fan-out as `n=1` requests each carrying one derived
  `sampling_seed`. Re-check both halves on a SGLang bump: dropping either one
  restores the clone, silently.
- **SGLang request sampling lives on the Sample, not the engine config.** Direct
  callers must `fork(..., sampling_params=ARSamplingParams(...))` before
  `generate`. Missing or non-`ARSamplingParams` frontiers now error; in-tree
  trainer, PE, and agentic paths already stamp this. `temperature` / `top_p` /
  `top_k` / `max_new_tokens` are gone from `SGLangEngineConfig` — there is no
  engine-config fallback.
- **Explicit vLLM-Omni AR seeds use one call per sample.** HI3 and Qwen3-Omni
  derive each request's seed from `ARSamplingParams.seed` and the AR Part's
  `sample_id`, using the same `derive_sampling_seed` mapping as SGLang. In
  vLLM-Omni 0.28, each call shares one sampling-params object per stage across
  its prompts; splitting preserves per-sample seeds under reordering and
  sharding, but can reduce throughput because the native backend executes calls
  sequentially. With `seed=None`, the adapters keep their existing batching and
  omit the AR seed. Diffusion sampling and x_T recipes are unchanged. This does
  not assign distinct engine seeds to replicas or guarantee bit-identical GPU
  outputs across hardware or engine configurations.
- **vLLM-Omni BAGEL ships the worker's token layout with every row.** `RLBagelPipeline`
  records the KV length, RoPE offset, and canvas in `BagelDiffusionConditions.layouts`,
  and `BagelDiffusionStage._resolve_single` raises when the rebuilt context differs, so
  a prompt or source-image preprocessing change on either side fails the first replay
  instead of silently training against a different sequence.
- **Per-row `metadata["canvas"]` is bounded by `sampling.height` / `sampling.width`
  and only valid for vLLM-Omni BAGEL with `FlowGRPO` (or `BagelFlowUniGRPO` with
  `ratio_norm=False`).** `BagelAdapter.build_inputs` sends one call per contiguous
  same-canvas run (`Sample.reject_row_canvases` rejects the key on every other
  engine), `build_image_segment` right-pads smaller trajectories on the token axis
  to the default canvas's token count so `LatentSegment.latents` stays dense, and
  `BagelDiffusionStage.replay` / `predict_velocity_at` trim back to
  `packed_vae_token_indexes`. Per-token replay anchors (`sde_means` in `FlowDPPO`
  and `BagelFlowUniGRPO` with `ratio_norm=True`) raise in
  `prepare_segment_anchors` when micro shapes differ, and `DiffusionNFT` reads
  `segment.latents[:, -1]` directly and would silently train on the right-padded
  zeros.
- **vLLM-Omni BAGEL packs sibling runs, t2i and it2i alike.** With `guidance_scale`
  and `cfg_img_scale` at 1 or below, each contiguous run of rows from one prompt
  group is one worker request with `num_outputs_per_prompt` set to the run length:
  the prompt (and for it2i the source image) is prefilled once and its KV is
  replicated per sibling. x_T is keyed per row, so a run may be any contiguous
  part of a group and a `RewardStack` micro may cut a group. Above 1 every row is
  its own request.
