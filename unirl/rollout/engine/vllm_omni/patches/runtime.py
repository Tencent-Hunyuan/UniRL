"""Monkey-patch ``DiffusionLoRAManager._load_adapter`` to accept in-memory LoRA tensors."""

from __future__ import annotations

import importlib.util
import os
import signal
import threading
import time
from functools import wraps
from multiprocessing.process import BaseProcess as _MpBaseProcess
from types import MethodType

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


def _diffusers_hv15_rmsnorm(norm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run the exact Diffusers RMSNorm ordering on a vLLM RMSNorm module."""
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + norm.variance_epsilon)

    weight = norm.weight
    if weight.dtype in (torch.float16, torch.bfloat16):
        hidden_states = hidden_states.to(weight.dtype)
    return hidden_states * weight


def patch_hv15_refiner_qkv_weight_loader() -> None:
    """Load the HV1.5 token refiner's unfused Q/K/V projections directly."""
    try:
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
            HunyuanVideo15Transformer3DModel,
        )
    except (ImportError, AttributeError):
        return

    original_load_weights = HunyuanVideo15Transformer3DModel.load_weights
    if getattr(original_load_weights, "_diffrl_hv15_refiner_qkv_weight_loader", False):
        return

    stacked_params_mapping = (
        (".to_qkv", ".to_q"),
        (".to_qkv", ".to_k"),
        (".to_qkv", ".to_v"),
        (".add_kv_proj", ".add_q_proj"),
        (".add_kv_proj", ".add_k_proj"),
        (".add_kv_proj", ".add_v_proj"),
    )

    @wraps(original_load_weights)
    def _patched_load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        directly_loaded: set[str] = set()

        def _remaining_weights():
            for name, loaded_weight in weights:
                packed_name = next(
                    (
                        name.replace(weight_name, param_name)
                        for param_name, weight_name in stacked_params_mapping
                        if f"{weight_name}." in name
                    ),
                    None,
                )
                if packed_name is not None and packed_name not in params_dict and name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, loaded_weight)
                    directly_loaded.add(name)
                    continue
                yield name, loaded_weight

        loaded = original_load_weights(self, _remaining_weights())
        return set(loaded or ()) | directly_loaded

    _patched_load_weights._diffrl_hv15_refiner_qkv_weight_loader = True
    HunyuanVideo15Transformer3DModel.load_weights = _patched_load_weights


def patch_hv15_autocast_forward() -> None:
    """Match the trainer's autocast boundary around each HunyuanVideo-1.5 transformer forward."""
    try:
        from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
            HunyuanVideo15Transformer3DModel,
        )
    except (ImportError, AttributeError):
        return

    original_forward = HunyuanVideo15Transformer3DModel.forward
    if getattr(original_forward, "_diffrl_hv15_autocast_forward", False):
        return

    @wraps(original_forward)
    def _patched_forward(self, *args, **kwargs):
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        dtype = self.transformer_blocks[0].norm1.linear.weight.dtype
        if not torch.is_tensor(hidden_states) or hidden_states.device.type != "cuda":
            return original_forward(self, *args, **kwargs)
        if dtype not in (torch.float16, torch.bfloat16):
            return original_forward(self, *args, **kwargs)
        with torch.autocast("cuda", dtype=dtype):
            return original_forward(self, *args, **kwargs)

    _patched_forward._diffrl_hv15_autocast_forward = True
    HunyuanVideo15Transformer3DModel.forward = _patched_forward


def patch_hv15_qk_rmsnorm() -> None:
    """Match Diffusers' HunyuanVideo-1.5 Q/K RMSNorm instead of the pinned fused vLLM-C kernel."""
    try:
        from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
            HunyuanVideo15Attention,
        )
    except (ImportError, AttributeError):
        return

    original_init = HunyuanVideo15Attention.__init__
    if getattr(original_init, "_diffrl_hv15_qk_rmsnorm", False):
        return

    def _patched_init(self, *args, _orig=original_init, **kwargs):
        _orig(self, *args, **kwargs)

        for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
            norm = getattr(self, name, None)
            if norm is None or getattr(norm.forward, "_diffrl_hv15_qk_rmsnorm", False):
                continue
            original_forward = norm.forward

            def _patched_forward(layer, hidden_states, residual=None, _orig_forward=original_forward):
                if residual is not None:
                    return _orig_forward(hidden_states, residual)
                return _diffusers_hv15_rmsnorm(layer, hidden_states)

            _patched_forward._diffrl_hv15_qk_rmsnorm = True
            norm.forward = MethodType(_patched_forward, norm)

    _patched_init._diffrl_hv15_qk_rmsnorm = True
    HunyuanVideo15Attention.__init__ = _patched_init


def patch_hv15_sdpa_attention_mask() -> None:
    """Keep the Diffusers HV1.5 boolean SDPA mask instead of dropping an all-true mask."""
    try:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl
        from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
            HunyuanVideo15Attention,
        )
    except (ImportError, AttributeError):
        return

    original_init = HunyuanVideo15Attention.__init__
    if getattr(original_init, "_diffrl_hv15_sdpa_attention_mask", False):
        return

    def _patched_init(self, *args, _orig=original_init, **kwargs):
        _orig(self, *args, **kwargs)

        for implementation in (self.attn.attention, self.attn.sdpa_fallback):
            if not isinstance(implementation, SDPAImpl):
                continue
            original_forward = implementation._forward_impl
            if getattr(original_forward, "_diffrl_hv15_sdpa_attention_mask", False):
                continue

            def _patched_forward(
                layer,
                query,
                key,
                value,
                attn_metadata=None,
                mask_mode="broadcast_k",
                _orig_forward=original_forward,
            ):
                attention_mask = None if attn_metadata is None else attn_metadata.attn_mask
                if (
                    attention_mask is None
                    or attention_mask.ndim != 2
                    or query.shape[1] != key.shape[1]
                    or attention_mask.shape != (query.shape[0], key.shape[1])
                ):
                    return _orig_forward(query, key, value, attn_metadata, mask_mode)

                batch_size, sequence_length = attention_mask.shape
                attention_mask = attention_mask.bool().view(batch_size, 1, 1, sequence_length)
                attention_mask = attention_mask.repeat(1, 1, sequence_length, 1)
                attention_mask = (attention_mask & attention_mask.transpose(2, 3)).bool()

                query, key, value = (tensor.permute(0, 2, 1, 3) for tensor in (query, key, value))
                output = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=layer.causal,
                    scale=layer.softmax_scale,
                )
                return output.permute(0, 2, 1, 3)

            _patched_forward._diffrl_hv15_sdpa_attention_mask = True
            implementation._forward_impl = MethodType(_patched_forward, implementation)

    _patched_init._diffrl_hv15_sdpa_attention_mask = True
    HunyuanVideo15Attention.__init__ = _patched_init


def _diffusers_hv15_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply HunyuanVideo-1.5 RoPE with Diffusers' fp32 frequency arithmetic."""
    from diffusers.models.embeddings import apply_rotary_emb

    cos, sin = image_rotary_emb
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    freqs = (cos, sin)
    return (
        apply_rotary_emb(query, freqs, sequence_dim=1),
        apply_rotary_emb(key, freqs, sequence_dim=1),
    )


def patch_hv15_rotary_embedding() -> None:
    """Match Diffusers' HunyuanVideo-1.5 RoPE instead of casting frequencies to bf16 for the fused kernel."""
    import inspect

    import torch.nn.functional as F
    from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Attention,
    )

    original_forward = HunyuanVideo15Attention.forward
    if getattr(original_forward, "_diffrl_hv15_rotary_embedding", False):
        return

    expected_parameters = (
        "self",
        "hidden_states",
        "encoder_hidden_states",
        "attention_mask",
        "image_rotary_emb",
    )
    if tuple(inspect.signature(original_forward).parameters) != expected_parameters:
        return

    def _patched_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        if image_rotary_emb is not None:
            query, key = _diffusers_hv15_rotary_embedding(query, key, image_rotary_emb)

        if encoder_hidden_states is not None:
            encoder_qkv, _ = self.add_kv_proj(encoder_hidden_states)
            add_q_size = self.add_kv_proj.num_heads * self.head_dim
            add_kv_size = self.add_kv_proj.num_kv_heads * self.head_dim
            encoder_query, encoder_key, encoder_value = encoder_qkv.split(
                [add_q_size, add_kv_size, add_kv_size],
                dim=-1,
            )

            encoder_query = encoder_query.unflatten(-1, (self.add_kv_proj.num_heads, -1))
            encoder_key = encoder_key.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))
            encoder_value = encoder_value.unflatten(-1, (self.add_kv_proj.num_kv_heads, -1))

            encoder_query = self.norm_added_q(encoder_query)
            encoder_key = self.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=1)
            key = torch.cat([key, encoder_key], dim=1)
            value = torch.cat([value, encoder_value], dim=1)

        attn_metadata = None
        if attention_mask is not None:
            seq_len = query.shape[1]
            attention_mask = F.pad(attention_mask, (seq_len - attention_mask.shape[1], 0), value=True)
            attention_mask = attention_mask.bool()
            attn_metadata = AttentionMetadata(attn_mask=attention_mask)

        hidden_states = self.attn(query, key, value, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
                [hidden_states.shape[1] - encoder_hidden_states.shape[1], encoder_hidden_states.shape[1]],
                dim=1,
            )
            hidden_states = self.to_out[0](hidden_states)
            encoder_hidden_states = self.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states

        hidden_states = self.to_out[0](hidden_states)
        return hidden_states

    _patched_forward._diffrl_hv15_rotary_embedding = True
    HunyuanVideo15Attention.forward = _patched_forward


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
        patch_hv15_refiner_qkv_weight_loader()
        patch_hv15_autocast_forward()
        patch_hv15_qk_rmsnorm()
        patch_hv15_sdpa_attention_mask()
        patch_hv15_rotary_embedding()
        patch_lora_request_passthrough()
        patch_per_request_ar_seed()
        patch_sigmas_passthrough()
        patch_hi3_flow_alignment()
        patch_master_port_unstrip()
        patch_moe_workspace_pool()


__all__ = [
    "OmniTensorLoRARequest",
    "VLLMOmniHijack",
    "patch_hv15_autocast_forward",
    "patch_hv15_qk_rmsnorm",
    "patch_hv15_refiner_qkv_weight_loader",
    "patch_hv15_sdpa_attention_mask",
    "patch_hv15_rotary_embedding",
    "patch_hi3_flow_alignment",
    "patch_per_request_ar_seed",
    "patch_sigmas_passthrough",
]
