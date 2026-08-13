# vLLM-Omni patches

> **Where it fits:** inside the *rollout* step — the `vllm_omni` engine's boot
> (`backends/native.py`) calls `install()` before any worker subprocess is spawned.
> Full map: [`../../../README.md`](../../../README.md).

*Quarantined vllm / vllm-omni monkeypatches behind one idempotent `install()`;
every patch is sentinel-guarded, so repeat installs are safe.*

## What it is

The whole vllm/vllm-omni patch surface, kept out of the engine code proper. The
package top is CPU-importable — runtime imports live in the submodules and load
lazily — so importing this package never pulls vllm.

## Why it exists

vLLM-Omni is a fast-moving pin. Keeping every divergence in one quarantined
package with an explicit retirement condition per patch means an upgrade is a
review of this table, not an archaeology exercise across the engine.

## How it works

`backends/native.py` calls `install()` once at boot; the worker extensions re-run
it defensively in `__new__`. `wrap_mp_process_for_children` **must run first** —
spawn children do not inherit parent patches, and `SpawnProcess` is a *sibling* of
`mp.Process`, not a subclass, so `BaseProcess` is the patch target that catches
every context.

**Extending it:** add the patch to `runtime.py` (or its own `compat_*.py` module if
it needs an import side effect), guard it with a sentinel, call it from
`install()`, and add a row below **with a DELETE-WHEN**. A patch without a
retirement condition is a patch nobody can ever remove.

## The patches

All in `runtime.py` unless noted.

| Patch | Why | DELETE-WHEN |
| --- | --- | --- |
| `wrap_mp_process_for_children` | Re-installs the bundle inside every spawn child. **Must run first** | the rest of the bundle is empty |
| `patch_dit_lora_loader` / `patch_ar_lora_loader` | Stock `DiffusionLoRAManager._load_adapter` loads only from a file path; RL pushes freshly-trained adapter tensors without a disk round-trip (`OmniTensorLoRARequest`). Lifted verbatim from verl-omni | vllm-omni's LoRA managers accept tensor-bag requests natively |
| `patch_fp32_skip` | Punica kernels hard-assert dtype; HI3's MoE router gate is fp32, so non-fp16/bf16 layers must be skipped for LoRA wrapping | vllm's `from_layer` skips unsupported dtypes itself |
| `patch_lora_request_passthrough` | `Omni.generate` never forwards `lora_request`, needed by the HI3 AR-prelude stage. Verified still absent at upstream main (~v0.22.0rc1); `AsyncOmniEngine.add_request` has accepted the kwarg all along, so a small upstream PR forwarding it would retire this | vllm-omni upstreams the kwarg (then the `ar_lora_passthrough` gate drops too) |
| `patch_per_request_ar_seed` | One `SamplingParams` is shared across requests, so a GRPO group's N requests collapse to identical tokens | vllm-omni stops sharing one `SamplingParams` |
| `patch_qwen3_omni_thinker_lora` | Backport of vllm-omni #3915: expose the Thinker LoRA interface, select `thinker_config` during model init, accept the current M-RoPE signature | pin vllm-omni ≥ 0.22 |
| `patch_sigmas_passthrough` | HI3's DiT `scheduler.set_timesteps` never receives `sampling_params.sigmas` | upstream forwards `sigmas` itself |
| `patch_hi3_flow_alignment` | Port of upstream `eed27812` to the pinned v0.20.0 KV-cache API | pin ≥ v0.21 — upstream removed the v0.20.0 `ImageKVCacheManager` API, so the patch self-skips there and is dead code once the pin moves |
| `install_fate_sharing` | `PR_SET_PDEATHSIG` is bound by Linux to the **specific creating thread**, so arming it for children of short-lived init threads kills healthy workers; and a worker inside a CUDA/NCCL call never observes vLLM's `death_pipe` EOF | vllm's own child-reaping is thread-safe |
| `compat_tokenizer` (module) | HI3's `__init__` looks up `<img_ratio_36>` and computes `ratio_36 + 1`; the Base checkpoint ships ratio tokens 0-32 only → `TypeError: … 'NoneType' and 'int'`. Both the slow **and** fast tokenizer classes must be patched, not the shared base. The module import *is* the install trigger (it is the `HI3ARWorkerExtension` qualname target). Upstream ≥ v0.20.0 raises a clean `ValueError` instead — a better error, but the Base ckpt still needs this 0-fallback to work | Base-ckpt support is dropped (Instruct ships the tokens) |
| `compat_hi3_lora` (module) | vllm 0.20 expects a flat `list[tuple[str, str, int, str]]` from `get_expert_mapping`; HI3 returns a 2-tuple, so `process_packed_modules_mapping` trips `ValueError: too many values to unpack` at boot under `enable_lora` | vllm handles the 2-tuple / HI3 returns the flat list |
| `compat_qwen3_omni` (module) | Compatibility helpers for Qwen3-Omni on the pinned runtime | the pin carries them |

## Gotchas

- **`wrap_mp_process_for_children` must be installed first.** Everything else is
  inherited by spawn children only through it.
- **This package top must stay CPU-importable.** Runtime imports belong in the
  submodules, loaded lazily — `import unirl.rollout.engine.vllm_omni.patches` must
  not pull vllm.
- **Every patch needs a DELETE-WHEN row.** Without one it is permanent by default.
- **`patch_hi3_flow_alignment` self-skips on newer pins** — it is dead code, not a
  live patch, once the pin moves past v0.20.0.
