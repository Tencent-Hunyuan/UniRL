"""Quarantined vllm / vllm-omni monkeypatches — one idempotent ``install()``.

The seam's boot (``backends/native.py``) calls :func:`install` before any
worker subprocess is spawned; the worker extensions re-run it defensively in
``__new__``. Every patch is idempotent (sentinel-guarded), so repeat installs
are safe. This package top is CPU-importable — the runtime imports live in
the submodules, loaded lazily.

Patch registry (all in ``runtime.py`` unless noted) with DELETE-WHEN notes:

- ``wrap_mp_process_for_children`` — re-installs the bundle inside every
  spawn child (must run FIRST; spawn children don't inherit parent patches).
  DELETE-WHEN: the rest of the bundle is empty.
- ``patch_dit_lora_loader`` / ``patch_ar_lora_loader`` — in-memory LoRA
  tensor-bag loading (``OmniTensorLoRARequest``).
  DELETE-WHEN: vllm-omni's LoRA managers accept tensor-bag requests natively.
- ``patch_fp32_skip`` — skip LoRA-wrapping non-fp16/bf16 layers (punica
  kernels hard-assert dtype; HI3's MoE router gate is fp32).
  DELETE-WHEN: vllm's ``from_layer`` skips unsupported dtypes itself.
- ``patch_lora_request_passthrough`` — ``lora_request`` kwarg on
  ``Omni.generate`` for the HI3 AR-prelude stage.
  DELETE-WHEN: vllm-omni upstreams the kwarg (then the engine passes it
  unconditionally and the ``ar_lora_passthrough`` gate drops). Verified
  still absent at upstream main (~v0.22.0rc1): ``Omni.generate`` never
  forwards it, though ``AsyncOmniEngine.add_request`` has accepted the
  kwarg all along — a small upstream PR forwarding it would retire this.
- ``patch_per_request_ar_seed`` — fresh per-request AR seed so a GRPO
  group's N requests don't collapse to identical tokens.
  DELETE-WHEN: vllm-omni stops sharing one SamplingParams across requests.
- ``patch_sigmas_passthrough`` — forwards ``sampling_params.sigmas`` into
  HI3's DiT ``scheduler.set_timesteps``.
  DELETE-WHEN: upstream pipeline forwards ``sigmas`` itself.
- ``patch_hi3_flow_alignment`` — port of upstream eed27812 to the pinned
  v0.20.0 KV-cache API.
  DELETE-WHEN: pin ≥ v0.21 — upstream main removed/rewrote the v0.20.0
  ``ImageKVCacheManager`` API entirely, so the patch self-skips there and
  is dead code once the pin moves.
- ``compat_tokenizer`` (module) — ``convert_tokens_to_ids`` returning 0 for
  the Base ckpt's missing ``<img_ratio_36>``; also the
  ``HI3ARWorkerExtension`` qualname target whose module import fires it.
  DELETE-WHEN: Base-ckpt support is dropped (Instruct ships the tokens).
  NB upstream ≥ v0.20.0 raises a clean ValueError on the missing tokens
  instead of the old TypeError — a better error, but the Base ckpt still
  needs this 0-fallback to actually WORK.
- ``compat_hi3_lora`` (module) — unwrap HI3's 2-tuple
  ``get_expert_mapping`` for vllm 0.20's LoRA path.
  DELETE-WHEN: vllm handles the 2-tuple shape / HI3 returns the flat list.
"""

from __future__ import annotations


def install() -> None:
    """Install the full vllm/vllm-omni patch bundle (idempotent).

    Lazy: importing this package stays CPU-safe; the runtime import happens
    here, at the spawn boundary.
    """
    from unirl.rollout.engine.vllm_omni.patches.runtime import VLLMOmniHijack

    VLLMOmniHijack.hijack()


def __getattr__(name: str):
    if name in ("VLLMOmniHijack", "OmniTensorLoRARequest"):
        from unirl.rollout.engine.vllm_omni.patches import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["install", "OmniTensorLoRARequest", "VLLMOmniHijack"]
