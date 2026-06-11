"""Monkey-patch ``DiffusionLoRAManager._load_adapter`` to accept in-memory
LoRA tensors.

vLLM-Omni's stock ``DiffusionLoRAManager._load_adapter`` only loads LoRA
weights from a file path (calls ``LoRAModel.from_local_checkpoint``). For
RL we need to push freshly-trained adapter tensors directly without going
through disk. This module lifts the verl-omni hijack pattern verbatim:

- ``OmniTensorLoRARequest`` extends ``vllm_omni.lora.request.LoRARequest``
  with two extra fields (``peft_config`` dict + ``lora_tensors`` dict).
- ``VLLMOmniHijack.hijack()`` replaces ``DiffusionLoRAManager._load_adapter``
  with a version that branches on the request type: tensor requests go
  through ``LoRAModel.from_lora_tensors``, file-path requests still hit
  the original code path.

Origin: ``verl-omni/verl_omni/utils/vllm_omni/utils.py``. Lifted as-is.
Run ``VLLMOmniHijack.hijack()`` once per worker subprocess (typically
from a worker-extension's ``__new__``).
"""

from __future__ import annotations

from multiprocessing.process import BaseProcess as _MpBaseProcess

from msgspec import field

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel  # type: ignore[no-redef]

from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, logger
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest


class OmniTensorLoRARequest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


# ============================================================
# Subprocess propagation — make spawn children also run hijack
# ============================================================
#
# vllm-omni's ``multiproc_executor`` calls ``mp.set_start_method("spawn", force=True)``.
# Each spawn child is a fresh Python interpreter that does not inherit the
# parent's monkey-patches. Without this hook, ``patch_fp32_skip`` (and other
# patches whose targets are imported by the child) take effect in the driver
# but NOT in the worker subprocesses where vllm.lora.utils.from_layer is
# actually called during model loading — fp32 router gate then crashes punica.
#
# Mirrors the LIN-210 sglang pattern (``samplers/sglang/patches/_spawn_wrap.py``).


class _DiffrlPatchedTarget:
    """Pickleable top-level wrapper that installs patches in the child first.

    Must be a module-level class so spawn's pickler can serialise the wrapped
    target across the process boundary. Nested functions / closures cannot be
    pickled and would break spawn.
    """

    def __init__(self, target):
        self._target = target

    def __call__(self, *args, **kwargs):
        VLLMOmniHijack.hijack()
        return self._target(*args, **kwargs)


_WRAP_SENTINEL = "_diffrl_target_wrapped"


def wrap_mp_process_for_children() -> None:
    """Replace ``BaseProcess.__init__`` so spawned targets install patches first.

    Patching ``mp.Process.__init__`` alone misses spawn-context Process classes
    (vllm-omni's stage launcher uses ``get_mp_context().Process`` ==
    ``SpawnProcess``, a sibling class, not a subclass). All context-specific
    Process classes inherit from ``BaseProcess``, so patching the root catches
    every context in one shot.
    """
    if getattr(_MpBaseProcess, _WRAP_SENTINEL, False):
        return

    orig_init = _MpBaseProcess.__init__

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        args=(),
        kwargs=None,
        *,
        daemon=None,
    ):
        if target is not None and not isinstance(target, _DiffrlPatchedTarget):
            target = _DiffrlPatchedTarget(target)
        orig_init(
            self,
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs=kwargs or {},
            daemon=daemon,
        )

    _MpBaseProcess.__init__ = __init__
    setattr(_MpBaseProcess, _WRAP_SENTINEL, True)


def patch_dit_lora_loader() -> None:
    """Patch ``DiffusionLoRAManager._load_adapter`` (DiT stage) to support in-memory tensors.

    vLLM-Omni's stock loader only accepts on-disk adapters. We branch on the
    request type: ``OmniTensorLoRARequest`` loads from in-memory tensors via
    ``LoRAModel.from_lora_tensors``; everything else falls through to the
    original on-disk loader via ``LoRAModel.from_local_checkpoint``.
    """

    def hijack__load_adapter(self, lora_request: OmniTensorLoRARequest) -> tuple[LoRAModel, PEFTHelper]:
        if not self._expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

        lora_tensors = None

        if isinstance(lora_request, OmniTensorLoRARequest):
            peft_config = lora_request.peft_config
            lora_tensors = lora_request.lora_tensors
            peft_helper = PEFTHelper.from_dict(peft_config)
        else:
            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            logger.debug("Resolved LoRA path: %s", lora_path)

            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                max_position_embeddings=None,  # no need in diffusion
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
            )

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.target_modules,
        )

        if isinstance(lora_request, OmniTensorLoRARequest):
            lora_model = LoRAModel.from_lora_tensors(
                tensors=lora_tensors,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",  # consistent w/ vllm's behavior
                dtype=self.dtype,
                model_vocab_size=None,
                weights_mapper=None,
            )
        else:
            lora_model = LoRAModel.from_local_checkpoint(
                lora_path,
                expected_lora_modules=self._expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",  # consistent w/ vllm's behavior
                dtype=self.dtype,
                model_vocab_size=None,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=None,
            )

        logger.info(
            "Loaded LoRA model: id=%d, num_modules=%d, modules=%s",
            lora_model.id,
            len(lora_model.loras),
            list(lora_model.loras.keys()),
        )

        for lora in lora_model.loras.values():
            lora.optimize()  # ref: _create_merged_loras_inplace, internal scaling

        return lora_model, peft_helper

    setattr(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)


def patch_ar_lora_loader() -> None:
    """Patch ``WorkerLoRAManager._load_adapter`` (AR stage) to support in-memory tensors.

    Best-effort: vllm's worker_manager is only importable in worker subprocesses
    that actually instantiate it. Returns just the ``LoRAModel`` (no peft_helper
    tuple). Mirrors the DiT shim for in-memory tensors and falls through to the
    original on-disk loader for plain ``LoRARequest``.
    """
    try:
        from vllm.lora.worker_manager import WorkerLoRAManager
    except ImportError:
        return

    _orig_ar_load_adapter = WorkerLoRAManager._load_adapter
    if getattr(_orig_ar_load_adapter, "_diffrl_hijacked", False):
        return

    def hijack_ar__load_adapter(self, lora_request, _orig=_orig_ar_load_adapter) -> LoRAModel:
        if not isinstance(lora_request, OmniTensorLoRARequest):
            return _orig(self, lora_request)

        peft_helper = PEFTHelper.from_dict(lora_request.peft_config or {})
        peft_helper.validate_legal(self.lora_config)

        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        lora = self._lora_model_cls.from_lora_tensors(
            tensors=lora_request.lora_tensors or {},
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",
            dtype=self.lora_config.lora_dtype,
            model_vocab_size=self.vocab_size,
            weights_mapper=hf_to_vllm_mapper,
        )
        return lora

    hijack_ar__load_adapter._diffrl_hijacked = True  # type: ignore[attr-defined]
    setattr(WorkerLoRAManager, "_load_adapter", hijack_ar__load_adapter)


def patch_fp32_skip() -> None:
    """Patch ``vllm.lora.utils.from_layer`` to skip non-fp16/bf16 layers.

    punica lora_shrink/expand kernels hard-assert inputs.dtype in [fp16, bf16].
    Skip LoRA wrap for fp32 layers (e.g. HI3 MoE router gate) and for
    non-fp16/bf16 dtypes (e.g. quantized) so the original layer.forward runs
    unmodified. If you intentionally want LoRA on such a layer, choose one:

      (a) cast the layer to bf16 in model code (lose precision)
      (b) wrap with a pure-pytorch LoRA variant (no punica),
          e.g. vllm_omni DiffusionBaseLinearLayerWithLoRA
      (c) filter target_modules so it does not match this layer

    Replaces pod-local file patch on ``vllm/lora/utils.py``.
    """
    try:
        import torch as _torch
        import vllm.lora.utils as _lora_utils
    except (ImportError, AttributeError):
        return  # vllm not available in this process; skip

    _orig_from_layer = _lora_utils.from_layer
    if getattr(_orig_from_layer, "_diffrl_fp32_skip", False):
        return

    def _patched_from_layer(
        layer, max_loras, lora_config, packed_modules_list, model_config=None, _orig=_orig_from_layer
    ):
        _weight = getattr(layer, "weight", None)
        if _weight is not None and _weight.dtype not in (_torch.float16, _torch.bfloat16):
            _lora_utils.logger.warning_once(
                "Skipping LoRA wrap for layer=%s (weight.dtype=%s not in [fp16, bf16]). "
                "punica kernel does not support this dtype. See vllm/lora/utils.py:from_layer "
                "docstring for workarounds if you intended to LoRA this layer.",
                type(layer).__name__,
                _weight.dtype,
            )
            return layer
        return _orig(layer, max_loras, lora_config, packed_modules_list, model_config)

    _patched_from_layer._diffrl_fp32_skip = True  # type: ignore[attr-defined]
    _lora_utils.from_layer = _patched_from_layer

    # Rebind stale references in modules that did `from vllm.lora.utils import
    # from_layer` at top level before our patch ran.
    import importlib as _importlib

    for _modname in (
        "vllm.lora.lora_model",
        "vllm.lora.models",
        "vllm.lora.model_manager",
        "vllm.lora.worker_manager",
    ):
        try:
            _mod = _importlib.import_module(_modname)
        except ImportError:
            continue
        if getattr(_mod, "from_layer", None) is _orig_from_layer:
            _mod.from_layer = _patched_from_layer


def patch_lora_request_passthrough() -> None:
    """Forward ``lora_request`` through ``Omni.generate`` to ``engine.add_request``.

    Required for HI3-Instruct t2i RL (``think_recaption`` mode) so that the AR
    prelude stage in vllm-omni picks up the per-rollout LoRA adapter alongside
    the DiT stage. Without this, ``VLLMOmniRolloutEngine.generate`` cannot pass
    ``lora_request`` into the AR stage's request scheduler — the AR worker runs
    the base model while DiT runs the LoRA-adapted model (half-adapted
    trajectory => silent policy/rollout mismatch).

    Replaces pod-local file patch on ``vllm_omni/entrypoints/omni.py``.
    """
    try:
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
        from vllm_omni.entrypoints.omni import Omni
    except (ImportError, AttributeError):
        return  # vllm-omni not available in this process; skip

    # ── Omni.generate: stash lora_request on the engine instance ──────
    _orig_omni_generate = Omni.generate
    if not getattr(_orig_omni_generate, "_diffrl_lora_request_passthrough", False):

        def _patched_omni_generate(self, *args, lora_request=None, _orig=_orig_omni_generate, **kwargs):
            self.engine._diffrl_pending_lora_request = lora_request
            py_generator = kwargs.get("py_generator", False)
            try:
                result = _orig(self, *args, **kwargs)
            except Exception:
                self.engine._diffrl_pending_lora_request = None
                raise
            if py_generator:
                # ``_orig`` returned a generator — wrap so we clear the stash
                # only when the generator is exhausted / closed.
                def _wrapped(gen, engine):
                    try:
                        yield from gen
                    finally:
                        engine._diffrl_pending_lora_request = None

                return _wrapped(result, self.engine)
            self.engine._diffrl_pending_lora_request = None
            return result

        _patched_omni_generate._diffrl_lora_request_passthrough = True  # type: ignore[attr-defined]
        Omni.generate = _patched_omni_generate

    # ── AsyncOmniEngine.add_request: pickup from stash ────────────────
    _orig_add_request = AsyncOmniEngine.add_request
    if not getattr(_orig_add_request, "_diffrl_lora_request_passthrough", False):

        def _patched_add_request(self, *args, lora_request=None, _orig=_orig_add_request, **kwargs):
            if lora_request is None:
                lora_request = getattr(self, "_diffrl_pending_lora_request", None)
            return _orig(self, *args, lora_request=lora_request, **kwargs)

        _patched_add_request._diffrl_lora_request_passthrough = True  # type: ignore[attr-defined]
        AsyncOmniEngine.add_request = _patched_add_request


def patch_sigmas_passthrough() -> None:
    """Monkey-patch HunyuanImage3Pipeline to forward custom sigmas to DiT scheduler.

    Outer ``HunyuanImage3Pipeline.forward`` extracts sigmas from req and stashes
    on the instance; inner ``HunyuanImage3Text2ImagePipeline.__call__`` picks up
    via ``self.model`` (which references the outer instance) and injects as a
    kwarg so ``scheduler.set_timesteps`` gets the correct schedule.

    Without this, UniRL's FlowMatchSchedulePolicy.sigmas is never
    forwarded to the DiT scheduler (rollout-train sigma mismatch
    max abs diff ~0.158 => GRPO log-prob replay incorrect).

    Replaces pod-local file patch on ``vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py``.
    """
    try:
        from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
            HunyuanImage3Pipeline,
            HunyuanImage3Text2ImagePipeline,
        )

        _orig_outer_forward = HunyuanImage3Pipeline.forward
        if not getattr(_orig_outer_forward, "_diffrl_sigmas_passthrough", False):

            def _patched_outer_forward(self, req, *args, _orig=_orig_outer_forward, **kwargs):
                sigmas = getattr(getattr(req, "sampling_params", None), "sigmas", None)
                self.unirl_sigmas = sigmas
                try:
                    return _orig(self, req, *args, **kwargs)
                finally:
                    self.unirl_sigmas = None

            _patched_outer_forward._diffrl_sigmas_passthrough = True  # type: ignore[attr-defined]
            HunyuanImage3Pipeline.forward = _patched_outer_forward

        _orig_inner_call = HunyuanImage3Text2ImagePipeline.__call__
        if not getattr(_orig_inner_call, "_diffrl_sigmas_passthrough", False):

            def _patched_inner_call(self, *args, _orig=_orig_inner_call, **kwargs):
                outer = getattr(self, "model", None)
                sigmas = getattr(outer, "unirl_sigmas", None) if outer is not None else None
                if sigmas is not None and "sigmas" not in kwargs:
                    kwargs["sigmas"] = sigmas
                return _orig(self, *args, **kwargs)

            _patched_inner_call._diffrl_sigmas_passthrough = True  # type: ignore[attr-defined]
            HunyuanImage3Text2ImagePipeline.__call__ = _patched_inner_call
    except (ImportError, AttributeError):
        pass  # pipeline not available in this process; skip


def patch_per_request_ar_seed() -> None:
    """Stamp a fresh os.urandom seed onto every AR SamplingParams in add_request's
    sampling_params_list. Without this, a GRPO group's N parallel requests all
    re-seed from the same shared SamplingParams ref and collapse to byte-identical
    AR tokens despite temperature > 0.
    """
    try:
        import msgspec as _msgspec
        from vllm import SamplingParams as VLLMSamplingParams
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    except (ImportError, AttributeError):
        return

    _orig = AsyncOmniEngine.add_request
    if getattr(_orig, "_diffrl_per_request_ar_seed", False):
        return

    import os as _os

    def _patched(self, *args, sampling_params_list=None, _orig=_orig, **kwargs):
        if sampling_params_list is not None:
            # SamplingParams is a msgspec.Struct shared across the N add_request
            # calls; ``structs.replace`` produces a brand-new instance per request
            # so the worker queue does not see one object holding the last seed.
            sampling_params_list = [
                _msgspec.structs.replace(sp, seed=int.from_bytes(_os.urandom(4), "big"))
                if isinstance(sp, VLLMSamplingParams) and getattr(sp, "seed", None) is None
                else sp
                for sp in sampling_params_list
            ]
        return _orig(self, *args, sampling_params_list=sampling_params_list, **kwargs)

    _patched._diffrl_per_request_ar_seed = True  # type: ignore[attr-defined]
    AsyncOmniEngine.add_request = _patched


def patch_master_port_unstrip() -> None:
    """Keep ``master_port`` alive through ``AsyncOmniEngine._strip_single_engine_args``.

    At the v0.20.0 pin the ``stage_configs_path`` route strips parent
    ``EngineArgs`` fields (including ``master_port``) from the kwargs that
    become ``base_engine_args`` for the per-stage YAML merge
    (``async_omni_engine.py:1558``), and the post-resolution injection loop
    only re-adds ``enable_sleep_mode`` / ``lora_path`` / ``lora_scale``.
    Net effect: the engine-reserved per-replica master-port base NEVER
    reaches ``OmniDiffusionConfig``, so every stage settles from the shared
    ``(None or 30005) + random(0, 100)`` window with only the 37-stride
    bind-check scan for collision avoidance (``diffusion/data.py:578``).
    Eight colocated replicas race that window; fast-booting models (SD3.5)
    happened to win, slow-booting ones (Qwen-Image, ~35s weight load) lose
    the check-to-bind TOCTOU and die with ``DistNetworkError ... port:
    30005, code: -98`` (LIN-382 qwen probe, 2026-06-07).

    Re-attach the caller's ``master_port`` to the stripped dict so the
    existing ``load_stage_configs_from_yaml`` ``base_engine_args`` merge
    lands it per stage. Stage-YAML keys still win (none of ours define
    ``master_port``); the settle scan stays as the TOCTOU fallback.

    DELETE-WHEN: pin >= v0.21.0rc2 — #3803 honors the injected base
    verbatim (mind the env ``MASTER_PORT`` precedence landmine documented
    in ``docs/vllm-omni-v2-engine.md``).
    """
    try:
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine

        _orig = AsyncOmniEngine._strip_single_engine_args
        if getattr(_orig, "_diffrl_master_port_unstrip", False):
            return

        def _patched_strip(kwargs, _orig=_orig):
            out = _orig(kwargs)
            if isinstance(kwargs, dict):
                master_port = kwargs.get("master_port")
                if master_port is not None:
                    out["master_port"] = master_port
            return out

        _patched_strip._diffrl_master_port_unstrip = True  # type: ignore[attr-defined]
        AsyncOmniEngine._strip_single_engine_args = staticmethod(_patched_strip)
    except (ImportError, AttributeError):
        pass  # vllm-omni not available in this process; skip


def patch_hi3_flow_alignment() -> None:
    """Port of bjf-frz/fix-hi3-flow (vllm-omni eed27812) to v0.20.0's older
    KV-cache API: store full 4-D first-step KV, then scatter live image KV by
    absolute position_ids on subsequent steps. Silent skip on non-v0.20.0.

    Threads position_ids through a thread-local so we only need to patch
    `_save_image_kv_caches`, `_update_image_kv_caches` and a tiny wrapper
    around `HunyuanImage3DecoderLayer.forward` (no need to reimplement
    `ImageKVCacheManager.__call__` for the sake of one line).

    Delete this function once vllm-omni upstream lands the fix in our pinned version.
    """
    try:
        from vllm_omni.diffusion.models.hunyuan_image3 import (
            hunyuan_image3_transformer as _trans,
        )
    except (ImportError, AttributeError):
        return

    _ImageKVCacheManager = _trans.ImageKVCacheManager
    _DecoderLayer = _trans.HunyuanImage3DecoderLayer

    if not hasattr(_ImageKVCacheManager, "_save_image_kv_caches"):
        return

    import threading as _threading

    # Thread-local position_ids stash. Single denoise call chain (DecoderLayer.forward
    # → self_attn → image_attn → _update_image_kv_caches) is synchronous in one
    # thread, so the wrapper sets _tls.position_ids on entry and the patched
    # _update reads it back down the stack.
    _tls = _threading.local()

    _orig_save = _ImageKVCacheManager._save_image_kv_caches
    if not getattr(_orig_save, "_diffrl_hi3_flow_aligned", False):

        def _patched_save_image_kv_caches(self, key, value, seq_len):
            assert key.shape[1] == seq_len, f"first-step q_len({key.shape[1]}) != seq_len({seq_len})"
            self.image_kv_cache_map = (key.contiguous(), value.contiguous())

        _patched_save_image_kv_caches._diffrl_hi3_flow_aligned = True  # type: ignore[attr-defined]
        _ImageKVCacheManager._save_image_kv_caches = _patched_save_image_kv_caches

    _orig_update = _ImageKVCacheManager._update_image_kv_caches
    if not getattr(_orig_update, "_diffrl_hi3_flow_aligned", False):

        def _patched_update_image_kv_caches(self, key, value, seq_len, position_ids=None):
            cached_key, cached_value = self.image_kv_cache_map
            bs, q_len = key.shape[0], key.shape[1]
            if position_ids is None:
                position_ids = getattr(_tls, "position_ids", None)
            assert cached_key.dim() == 4, (
                f"patch_hi3_flow_alignment expects a 4-D cache from the patched "
                f"_save_image_kv_caches; got dim={cached_key.dim()}."
            )
            assert position_ids is not None and position_ids.shape == (bs, q_len), (
                f"position_ids missing or wrong shape: {None if position_ids is None else tuple(position_ids.shape)} "
                f"!= ({bs}, {q_len})"
            )
            result_k = cached_key.clone()
            result_v = cached_value.clone()
            for b in range(bs):
                result_k[b].index_copy_(0, position_ids[b], key[b])
                result_v[b].index_copy_(0, position_ids[b], value[b])
            return result_k.contiguous(), result_v.contiguous()

        _patched_update_image_kv_caches._diffrl_hi3_flow_aligned = True  # type: ignore[attr-defined]
        _ImageKVCacheManager._update_image_kv_caches = _patched_update_image_kv_caches

    _orig_decoder = _DecoderLayer.forward
    if not getattr(_orig_decoder, "_diffrl_hi3_flow_aligned", False):

        def _patched_decoder_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            *args,
            _orig=_orig_decoder,
            **kwargs,
        ):
            _prev = getattr(_tls, "position_ids", None)
            _tls.position_ids = position_ids
            try:
                return _orig(self, hidden_states, attention_mask, position_ids, *args, **kwargs)
            finally:
                _tls.position_ids = _prev

        _patched_decoder_forward._diffrl_hi3_flow_aligned = True  # type: ignore[attr-defined]
        _DecoderLayer.forward = _patched_decoder_forward


class VLLMOmniHijack:
    """Monkey-patches vllm-omni internals to support in-memory LoRA tensors.

    Two managers need patching for HI3 t2i:

    - ``vllm_omni.diffusion.lora.manager.DiffusionLoRAManager._load_adapter``
      drives the DiT stage and returns ``(LoRAModel, PEFTHelper)``.
    - ``vllm.lora.worker_manager.WorkerLoRAManager._load_adapter`` drives the
      AR stage and returns just ``LoRAModel``.

    Both originally only accept on-disk adapters. We branch on the request
    type and load from in-memory tensors when ``OmniTensorLoRARequest`` is
    passed, otherwise fall through to the original loader.
    """

    @staticmethod
    def hijack() -> None:
        # MUST run first: install the mp.Process wrap so any subsequent
        # spawn-spawned subprocesses also run this hijack() at startup.
        # Without this, patches that target functions imported during the
        # child's model-loading phase (notably patch_fp32_skip → from_layer)
        # never take effect in the worker subprocesses.
        wrap_mp_process_for_children()

        patch_dit_lora_loader()
        patch_ar_lora_loader()
        patch_fp32_skip()
        patch_lora_request_passthrough()
        patch_per_request_ar_seed()
        patch_sigmas_passthrough()
        patch_hi3_flow_alignment()
        patch_master_port_unstrip()


__all__ = [
    "OmniTensorLoRARequest",
    "VLLMOmniHijack",
    "patch_hi3_flow_alignment",
    "patch_per_request_ar_seed",
    "patch_sigmas_passthrough",
]
