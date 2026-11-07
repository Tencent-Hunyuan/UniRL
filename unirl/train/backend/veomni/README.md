# VeOmni backend (`unirl/train/backend/veomni`)

> **Where it fits:** training backend. In: a foundation model on meta + an FSDP/EP parallel config. Out: a `fully_shard`ed trainable model behind `VeOmniBackend`.
> Full map: [`../readme.md`](../readme.md).

Wraps VeOmni's torch-native distributed layer (FSDP2, EP, Ulysses SP) behind the same `TrainBackend` surface `fsdp` uses, reaching it through a selective-import shim so the wheel's transformers-locked model zoo never loads.

## What it is

`VeOmniBackend` owns the parallel state, the optimizer, and the gradient clip; `wrap.py` builds the `ParallelPlan` and calls `parallelize_model_fsdp2`; `ep/` and `sp/` hold the expert-parallel and sequence-parallel specializations. `_compat.py` is the only module that imports `veomni` — everything else goes through it, and it stubs the package in `sys.modules` so `veomni/__init__.py` and `veomni/models/__init__.py` are never executed.

## Why it exists

VeOmni's wheel ships a model zoo pinned to its own `transformers` version. UniRL pins a different one, and importing the zoo would drag the wrong classes into `ALL_ATTENTION_FUNCTIONS` and the auto-model registry. The shim imports only the distributed modules and the specific patched modelings UniRL asks for, so the two dependency trees coexist.

## How it works

- **Selective import via `sys.modules` stubs.** `ensure_installed()` locates the wheel with `importlib.util.find_spec` (which does not execute it), installs package-shaped stub modules, and re-attaches `load_model_weights` / `rank0_load_and_broadcast_weights` from `veomni.models.module_utils`.
- **Model registration is opt-in per architecture.** `ensure_qwen3_moe_installed()` / `ensure_qwen3_5_moe_installed()` import exactly one `veomni.models.transformers.*` modeling and then register VeOmni's Ulysses attention patch.
- **Parallel state comes from `init_parallel_state`.** `backend.py` calls the public `veomni.distributed.parallel_state.init_parallel_state(dp_size=..., ulysses_size=..., dp_mode="fsdp2", ...)`, folding Ulysses into the FSDP shard mesh so gradients reduce-scatter across SP ranks.

## Gotchas

- **The `veomni` pin is load-bearing; 0.1.12 is the first release that works.** 0.1.11's generated Qwen3-MoE passes `cache_position` to `transformers.masking_utils.create_causal_mask`, which dropped that keyword in 5.x; against UniRL's `transformers==5.12.1` pin the first MoE forward raises `TypeError: create_causal_mask() got an unexpected keyword argument 'cache_position'` (issue #627). Both generated modelings are affected in 0.1.11 — `patched_modeling_qwen3_moe_{gpu,npu}.py` and `patched_modeling_qwen3_5_moe_{gpu,npu}.py:2038`. Three shipped recipes take that path: `qwen3_moe_grpo_30b_a3b_veomni_ep_sglang`, `qwen3_5_moe_grpo_35b_a3b_base_dapo_sglang`, `qwen3_5_moe_grpo_35b_a3b_geo3k_mc_sglang`. Fixed upstream in 0.1.12, whose remaining `cache_position=` arguments go to `linear_attn`/`self_attn`/`language_model`, never to a mask constructor. (`DELETE-WHEN`: UniRL's `transformers` floor moves past the keyword removal, or the Qwen3-MoE path stops routing through VeOmni's patched modeling.)
- **Do not move the pin to 0.1.13.** That tag was never published to PyPI — the index stops at 0.1.12 and the `v0.1.13` tree still stamps `_version.py = "0.1.12"` — so the extra would not resolve at install. The tag also renames the public `init_parallel_state` to `_init_parallel_state`, leaving only `init_parallel_state_from_config`, which `backend.py` does not call. 0.1.12 keeps the public name with the signature `backend.py` passes (`dp_size`, `ulysses_size`, `dp_mode`, `device_type`, `extra_parallel_sizes`, `extra_parallel_names`).
- **Re-check symbols, not just modules, on any bump.** The shim imports fourteen names by path (`MixedPrecisionConfig`, `OpsImplementationConfig`, `clip_grad_norm`, `load_model_to_gpu`, `offload_model_to_cpu`, `ParallelPlan`, `get_parallel_state`, `init_parallel_state`, `gather_outputs`, `slice_input_tensor`, `parallelize_model_fsdp2`, `build_foundation_model`, `apply_veomni_attention_patch`, `group_gemm_fused_moe_forward`). A module can survive a release while a symbol inside it is renamed or made private, and nothing in the repo's lint gate would notice — there is no `tests/` tree to catch it.
