"""Monkey-patch ``DiffusionLoRAManager._load_adapter`` to accept in-memory LoRA tensors."""

from __future__ import annotations

import importlib.util
import os
import signal
import threading
import time
from multiprocessing.process import BaseProcess as _MpBaseProcess

import torch
from msgspec import field

try:
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    from vllm.lora.models import LoRAModel  # type: ignore[no-redef]

from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.utils import get_adapter_absolute_path
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, logger
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

from unirl.rollout.engine.vllm_omni.patches.compat_moe_workspace import patch_moe_workspace_pool


class OmniTensorLoRARequest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


_FATE_ANCHOR_ENV = "UNIRL_FATE_ANCHOR_PID"
_FATE_POLL_SECONDS = 5.0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True  # unknown failure — never take this as "dead"
    return True


def install_fate_sharing(anchor_pid: int, *, arm_pdeathsig: bool) -> None:
    """Bind this process's lifetime to the root of its spawn chain."""
    if arm_pdeathsig:
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)
        except Exception:  # noqa: BLE001 - best effort; the watchdog below is the real guarantee
            pass

    root = os.environ.get(_FATE_ANCHOR_ENV)
    root_pid = int(root) if root else int(anchor_pid)
    os.environ[_FATE_ANCHOR_ENV] = str(root_pid)

    original_ppid = os.getppid()

    def _watch() -> None:
        while True:
            time.sleep(_FATE_POLL_SECONDS)
            if not _pid_alive(root_pid):
                os._exit(1)
            if original_ppid != 1 and os.getppid() != original_ppid:
                os._exit(1)

    threading.Thread(target=_watch, daemon=True, name="unirl-fate-watchdog").start()


class _DiffrlPatchedTarget:
    """Pickleable top-level wrapper that installs patches in the child first."""

    def __init__(self, target):
        self._target = target
        self._anchor_pid = os.getpid()
        self._arm_pdeathsig = False

    def __call__(self, *args, **kwargs):
        install_fate_sharing(
            getattr(self, "_anchor_pid", os.getppid()),
            arm_pdeathsig=getattr(self, "_arm_pdeathsig", False),
        )
        VLLMOmniHijack.hijack()
        return self._target(*args, **kwargs)


_WRAP_SENTINEL = "_diffrl_target_wrapped"


def wrap_mp_process_for_children() -> None:
    """Replace ``BaseProcess.__init__`` so spawned targets install patches first."""
    if getattr(_MpBaseProcess, _WRAP_SENTINEL, False):
        return

    orig_init = _MpBaseProcess.__init__
    orig_start = _MpBaseProcess.start

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

    def start(self):
        target = getattr(self, "_target", None)
        if isinstance(target, _DiffrlPatchedTarget):
            target._arm_pdeathsig = threading.current_thread() is threading.main_thread()
        return orig_start(self)

    _MpBaseProcess.__init__ = __init__
    _MpBaseProcess.start = start
    setattr(_MpBaseProcess, _WRAP_SENTINEL, True)


def patch_qwen3_omni_thinker_lora() -> None:
    """Backport Qwen3-Omni Thinker LoRA support to vLLM-Omni 0.20."""
    module_name = "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker"
    if importlib.util.find_spec(module_name) is None:
        return

    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
        Qwen3OmniMoeThinkerMultiModalProcessor,
    )

    from unirl.rollout.engine.vllm_omni.patches.compat_qwen3_omni import (
        patch_qwen3_omni_audio_truncation,
        patch_qwen3_omni_audio_video_mrope,
        patch_qwen3_omni_thinker_class,
    )

    patch_qwen3_omni_thinker_class(Qwen3OmniMoeThinkerForConditionalGeneration)
    patch_qwen3_omni_audio_video_mrope(Qwen3OmniMoeThinkerForConditionalGeneration)
    patch_qwen3_omni_audio_truncation(Qwen3OmniMoeThinkerMultiModalProcessor)


def patch_dit_lora_loader() -> None:
    """Patch ``DiffusionLoRAManager._load_adapter`` (DiT stage) to support in-memory tensors."""

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
                max_position_embeddings=None,
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
                device="cpu",
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
                device="cpu",
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
            lora.optimize()

        return lora_model, peft_helper

    setattr(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)


def _deinterleave_fused_qkv_lora_b(lora_b, output_sizes, base_layer):
    """Split HI3's GQA-interleaved fused QKV LoRA-B into ``[q, k, v]`` slices."""
    if len(output_sizes) != 3:
        return None
    head_size = getattr(base_layer, "head_size", None)
    num_kv_heads = getattr(base_layer, "total_num_kv_heads", None)
    if not isinstance(head_size, int) or head_size <= 0:
        return None
    if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        return None
    q_size, k_size, v_size = output_sizes
    if k_size <= 0 or v_size != k_size:
        return None
    groups = q_size // k_size
    if groups * k_size != q_size or k_size != num_kv_heads * head_size:
        return None
    rank = lora_b.shape[1]
    try:
        lora_b_r = lora_b.reshape(num_kv_heads, groups + 2, head_size, rank)
    except RuntimeError:
        return None
    q_b, k_b, v_b = torch.split(lora_b_r, (groups, 1, 1), dim=1)
    return [q_b.reshape(-1, rank), k_b.reshape(-1, rank), v_b.reshape(-1, rank)]


def patch_dit_hi3_lora_weights() -> None:
    """Resolve and safely repack HI3 DiT LoRA weights."""
    try:
        from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
            HunyuanImage3Pipeline,
        )
    except (ImportError, AttributeError):
        return

    original = DiffusionLoRAManager._get_lora_weights
    if getattr(original, "_diffrl_hi3_lora_weights", False):
        return

    def wrapped(self, lora_model, full_module_name, _orig=original):
        weights = _orig(self, lora_model, full_module_name)
        if not isinstance(getattr(self, "pipeline", None), HunyuanImage3Pipeline):
            return weights

        prefix = "transformer.layers."
        if weights is None and full_module_name.startswith(prefix):
            alias = "model.layers." + full_module_name[len(prefix) :]
            weights = lora_model.get_lora(alias)

        if weights is None or not full_module_name.endswith(".qkv_proj"):
            return weights
        if isinstance(weights, PackedLoRALayerWeights):
            return weights

        def fail(reason: str) -> None:
            raise RuntimeError(
                f"Refusing to install HI3 fused-qkv LoRA for {full_module_name}: {reason}. "
                "Applying the interleaved tensor would route attention deltas to the wrong output rows."
            )

        if not isinstance(weights, LoRALayerWeights):
            fail(f"expected LoRALayerWeights, got {type(weights).__name__}")
        lora_b = weights.lora_b
        if not isinstance(lora_b, torch.Tensor) or lora_b.ndim != 2:
            fail("lora_b is not a 2-D tensor")

        lora_modules = getattr(self, "_lora_modules", None) or {}
        base_layer = getattr(lora_modules.get(full_module_name), "base_layer", None)
        output_sizes = [int(size) for size in (getattr(base_layer, "output_sizes", ()) or ())]
        if not output_sizes or int(lora_b.shape[0]) != sum(output_sizes):
            fail("base layer exposes no output_sizes matching lora_b's rows")

        slices = _deinterleave_fused_qkv_lora_b(lora_b, output_sizes, base_layer)
        if slices is None:
            fail("GQA layout is unrecognised (check head_size/total_num_kv_heads)")

        scaling = float(getattr(weights, "scaling", 1.0))
        if scaling != 1.0:
            slices = [part * scaling for part in slices]

        return PackedLoRALayerWeights(
            module_name=weights.module_name,
            rank=weights.rank,
            lora_alphas=[weights.lora_alpha] * 3,
            lora_a=[weights.lora_a] * 3,
            lora_b=slices,
            scaling=[1.0, 1.0, 1.0],
        )

    wrapped._diffrl_hi3_lora_weights = True
    DiffusionLoRAManager._get_lora_weights = wrapped


def patch_ar_lora_loader() -> None:
    """Patch ``WorkerLoRAManager._load_adapter`` (AR stage) to support in-memory tensors."""
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


def patch_ar_merged_lora_fused_tensor() -> None:
    """Accept a single fused lora_b [q+k+v, rank] in MergedQKV set_lora."""
    try:
        from vllm.lora.layers import column_parallel_linear as _cpl
    except (ImportError, AttributeError):
        return

    def _make(orig):
        def _set_lora(self, index, lora_a, lora_b, *args, _orig=orig, **kwargs):
            if isinstance(lora_b, torch.Tensor):
                output_sizes = list(getattr(self.base_layer, "output_sizes", []) or [])
                if output_sizes and int(lora_b.shape[0]) == sum(output_sizes):
                    slices = _deinterleave_fused_qkv_lora_b(lora_b, output_sizes, self.base_layer)
                    lora_b = slices if slices is not None else list(torch.split(lora_b, output_sizes, dim=0))
                    if isinstance(lora_a, torch.Tensor):
                        lora_a = [lora_a] * self.n_slices
            return _orig(self, index, lora_a, lora_b, *args, **kwargs)

        _set_lora._diffrl_fused_merged_tolerant = True  # type: ignore[attr-defined]
        return _set_lora

    for _name in (
        "MergedColumnParallelLinearWithLoRA",
        "MergedQKVParallelLinearWithLoRA",
    ):
        cls = getattr(_cpl, _name, None)
        if cls is None or "set_lora" not in cls.__dict__:
            continue
        orig = cls.__dict__["set_lora"]
        if getattr(orig, "_diffrl_fused_merged_tolerant", False):
            continue
        cls.set_lora = _make(orig)


def patch_fp32_skip() -> None:
    """Patch ``vllm.lora.utils.from_layer`` to skip non-fp16/bf16 layers."""
    try:
        import torch as _torch
        import vllm.lora.utils as _lora_utils
    except (ImportError, AttributeError):
        return

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

    # Rebind modules that imported from_layer before this patch ran.
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
    """Forward ``lora_request`` through ``Omni.generate`` to ``engine.add_request``."""
    try:
        from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
        from vllm_omni.entrypoints.omni import Omni
    except (ImportError, AttributeError):
        return

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

    _orig_add_request = AsyncOmniEngine.add_request
    if not getattr(_orig_add_request, "_diffrl_lora_request_passthrough", False):

        def _patched_add_request(self, *args, lora_request=None, _orig=_orig_add_request, **kwargs):
            if lora_request is None:
                lora_request = getattr(self, "_diffrl_pending_lora_request", None)
            return _orig(self, *args, lora_request=lora_request, **kwargs)

        _patched_add_request._diffrl_lora_request_passthrough = True  # type: ignore[attr-defined]
        AsyncOmniEngine.add_request = _patched_add_request


def patch_sigmas_passthrough() -> None:
    """Monkey-patch HunyuanImage3Pipeline to forward custom sigmas to DiT scheduler."""
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
        pass


def patch_per_request_ar_seed() -> None:
    """Stamp a fresh os.urandom seed onto every AR SamplingParams in add_request's sampling_params_list."""
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
    """Keep ``master_port`` alive through ``AsyncOmniEngine._strip_single_engine_args``."""
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
        pass


def patch_hi3_flow_alignment() -> None:
    """Port of vllm-omni eed27812 to v0.20.0's older KV-cache API; silent skip on any other version."""
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
    """Monkey-patches vllm-omni internals to support in-memory LoRA tensors."""

    @staticmethod
    def hijack() -> None:
        wrap_mp_process_for_children()

        patch_qwen3_omni_thinker_lora()
        patch_dit_lora_loader()
        patch_dit_hi3_lora_weights()
        patch_ar_lora_loader()
        patch_ar_merged_lora_fused_tensor()
        patch_fp32_skip()
        patch_lora_request_passthrough()
        patch_per_request_ar_seed()
        patch_sigmas_passthrough()
        patch_hi3_flow_alignment()
        patch_master_port_unstrip()
        patch_moe_workspace_pool()


__all__ = [
    "OmniTensorLoRARequest",
    "VLLMOmniHijack",
    "patch_hi3_flow_alignment",
    "patch_per_request_ar_seed",
    "patch_sigmas_passthrough",
]
