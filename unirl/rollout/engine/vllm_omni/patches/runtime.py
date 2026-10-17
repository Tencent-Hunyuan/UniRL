"""Monkey-patch ``DiffusionLoRAManager._load_adapter`` to accept in-memory LoRA tensors."""

from __future__ import annotations

import os
import signal
import threading
import time
from functools import wraps
from multiprocessing.process import BaseProcess as _MpBaseProcess

import torch
from msgspec import field
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.peft_helper import PEFTHelper
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager, logger
from vllm_omni.lora.request import LoRARequest as OmniLoRARequest

from unirl.rollout.engine.vllm_omni.patches.compat_moe_workspace import patch_moe_workspace_pool


class OmniTensorLoRARequest(OmniLoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


_FATE_ANCHOR_ENV = "UNIRL_FATE_ANCHOR_PID"
_FATE_POLL_SECONDS = 5.0


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
            try:
                os.kill(root_pid, 0)
            except ProcessLookupError:
                os._exit(1)
            except OSError:
                pass
            if original_ppid != 1 and os.getppid() != original_ppid:
                os._exit(1)

    threading.Thread(target=_watch, daemon=True, name="unirl-fate-watchdog").start()


class UnirlPatchedTarget:
    """Pickleable top-level wrapper that installs patches in the child first."""

    def __init__(self, target):
        self.target = target
        self.anchor_pid = os.getpid()
        self.arm_pdeathsig = False

    def __call__(self, *args, **kwargs):
        install_fate_sharing(
            self.anchor_pid,
            arm_pdeathsig=self.arm_pdeathsig,
        )
        VLLMOmniHijack.hijack()
        return self.target(*args, **kwargs)


_WRAP_SENTINEL = "_unirl_target_wrapped"


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
        if target is not None and not isinstance(target, UnirlPatchedTarget):
            target = UnirlPatchedTarget(target)
        orig_init(
            self,
            group=group,
            target=target,
            name=name,
            args=args,
            kwargs={} if kwargs is None else kwargs,
            daemon=daemon,
        )

    def start(self):
        target = self._target
        if isinstance(target, UnirlPatchedTarget):
            target.arm_pdeathsig = threading.current_thread() is threading.main_thread()
        return orig_start(self)

    _MpBaseProcess.__init__ = __init__
    _MpBaseProcess.start = start
    setattr(_MpBaseProcess, _WRAP_SENTINEL, True)


def patch_dit_lora_loader() -> None:
    """Patch ``DiffusionLoRAManager._load_adapter`` (DiT stage) to support in-memory tensors."""
    original = DiffusionLoRAManager._load_adapter
    if getattr(original, "_unirl_tensor_lora_loader", False):
        return

    def hijack__load_adapter(
        self,
        lora_request: OmniTensorLoRARequest,
        _orig=original,
    ) -> tuple[LoRAModel, PEFTHelper]:
        if not isinstance(lora_request, OmniTensorLoRARequest):
            return _orig(self, lora_request)
        if not self._expected_lora_modules:
            raise ValueError("No supported LoRA modules found in the diffusion pipeline.")

        logger.debug("Supported LoRA modules: %s", self._expected_lora_modules)

        peft_helper = PEFTHelper.from_dict({} if lora_request.peft_config is None else lora_request.peft_config)

        logger.info(
            "Loaded PEFT config: r=%d, lora_alpha=%d, target_modules=%s",
            peft_helper.r,
            peft_helper.lora_alpha,
            peft_helper.target_modules,
        )

        lora_model = LoRAModel.from_lora_tensors(
            tensors={} if lora_request.lora_tensors is None else lora_request.lora_tensors,
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",
            dtype=self.dtype,
            model_vocab_size=None,
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

    hijack__load_adapter._unirl_tensor_lora_loader = True  # type: ignore[attr-defined]
    setattr(DiffusionLoRAManager, "_load_adapter", hijack__load_adapter)


def deinterleave_fused_qkv_lora_b(lora_b, output_sizes, base_layer):
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
    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        HunyuanImage3Pipeline,
    )

    original = DiffusionLoRAManager._get_lora_weights
    if getattr(original, "_unirl_hi3_lora_weights", False):
        return

    def wrapped(self, lora_model, full_module_name, _orig=original):
        weights = _orig(self, lora_model, full_module_name)
        if not isinstance(self.pipeline, HunyuanImage3Pipeline):
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

        lora_layer = self._lora_modules.get(full_module_name)
        if lora_layer is None:
            fail("module is not registered in _lora_modules")
        base_layer = lora_layer.base_layer
        output_sizes = list(base_layer.output_sizes)
        if not output_sizes or int(lora_b.shape[0]) != sum(output_sizes):
            fail("base layer exposes no output_sizes matching lora_b's rows")

        slices = deinterleave_fused_qkv_lora_b(lora_b, output_sizes, base_layer)
        if slices is None:
            fail("GQA layout is unrecognised (check head_size/total_num_kv_heads)")

        scaling = weights.scaling
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

    wrapped._unirl_hi3_lora_weights = True
    DiffusionLoRAManager._get_lora_weights = wrapped


def patch_ar_lora_loader() -> None:
    """Patch ``WorkerLoRAManager._load_adapter`` (AR stage) to support in-memory tensors."""
    from vllm.lora.worker_manager import WorkerLoRAManager

    _orig_ar_load_adapter = WorkerLoRAManager._load_adapter
    if getattr(_orig_ar_load_adapter, "_unirl_hijacked", False):
        return

    def hijack_ar__load_adapter(self, lora_request, _orig=_orig_ar_load_adapter) -> LoRAModel:
        if not isinstance(lora_request, OmniTensorLoRARequest):
            return _orig(self, lora_request)

        peft_helper = PEFTHelper.from_dict({} if lora_request.peft_config is None else lora_request.peft_config)
        peft_helper.validate_legal(self.lora_config)

        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        if hf_to_vllm_mapper is not None:
            hf_to_vllm_mapper = hf_to_vllm_mapper.get_unstacked_mapper()
        lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)
        lora = self._lora_model_cls.from_lora_tensors(
            tensors={} if lora_request.lora_tensors is None else lora_request.lora_tensors,
            peft_helper=peft_helper,
            lora_model_id=lora_request.lora_int_id,
            device="cpu",
            dtype=self.lora_config.lora_dtype,
            model_vocab_size=self.vocab_size,
            weights_mapper=hf_to_vllm_mapper,
            skip_prefixes=lora_skip_prefixes,
        )
        return lora

    hijack_ar__load_adapter._unirl_hijacked = True  # type: ignore[attr-defined]
    setattr(WorkerLoRAManager, "_load_adapter", hijack_ar__load_adapter)


def patch_ar_merged_lora_fused_tensor() -> None:
    """Accept a single fused lora_b [q+k+v, rank] in MergedQKV set_lora."""
    from vllm.lora.layers import column_parallel_linear as _cpl

    for cls in (
        _cpl.MergedColumnParallelLinearWithLoRA,
        _cpl.MergedQKVParallelLinearWithLoRA,
    ):
        orig = cls.set_lora
        if getattr(orig, "_unirl_fused_merged_tolerant", False):
            continue

        def _set_lora(self, index, lora_a, lora_b, *args, _orig=orig, **kwargs):
            if isinstance(lora_b, torch.Tensor):
                output_sizes = list(self.base_layer.output_sizes)
                if output_sizes and int(lora_b.shape[0]) == sum(output_sizes):
                    slices = deinterleave_fused_qkv_lora_b(lora_b, output_sizes, self.base_layer)
                    lora_b = slices if slices is not None else list(torch.split(lora_b, output_sizes, dim=0))
                    if isinstance(lora_a, torch.Tensor):
                        lora_a = [lora_a] * self.n_slices
            return _orig(self, index, lora_a, lora_b, *args, **kwargs)

        _set_lora._unirl_fused_merged_tolerant = True  # type: ignore[attr-defined]
        cls.set_lora = _set_lora


def patch_fp32_skip() -> None:
    """Patch ``vllm.lora.utils.from_layer`` to skip non-fp16/bf16 layers."""
    import torch as _torch
    import vllm.lora.utils as _lora_utils

    _orig_from_layer = _lora_utils.from_layer
    if getattr(_orig_from_layer, "_unirl_fp32_skip", False):
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

    _patched_from_layer._unirl_fp32_skip = True  # type: ignore[attr-defined]
    _lora_utils.from_layer = _patched_from_layer

    # Rebind modules that imported from_layer before this patch ran.
    import importlib as _importlib

    for _modname in (
        "vllm.lora.lora_model",
        "vllm.lora.model_manager",
        "vllm.lora.worker_manager",
    ):
        _mod = _importlib.import_module(_modname)
        if getattr(_mod, "from_layer", None) is _orig_from_layer:
            _mod.from_layer = _patched_from_layer


def patch_hv15_packed_lora_mapping() -> None:
    """Expose HV1.5's packed QKV mapping to the diffusion LoRA manager."""
    from vllm_omni.diffusion.models.hunyuan_video.hunyuan_video_15_transformer import (
        HunyuanVideo15Transformer3DModel,
    )

    sentinel = "_unirl_hv15_packed_lora_mapping"
    if getattr(HunyuanVideo15Transformer3DModel, sentinel, False):
        return
    if not getattr(HunyuanVideo15Transformer3DModel, "stacked_params_mapping", None):
        HunyuanVideo15Transformer3DModel.stacked_params_mapping = (
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        )
    setattr(HunyuanVideo15Transformer3DModel, sentinel, True)


class HV15TorchLinearWithLoRA(torch.nn.Module):
    """Apply one in-memory LoRA adapter to an ordinary HV1.5 ``nn.Linear``."""

    n_slices = 1

    def __init__(self, base_layer: torch.nn.Linear) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.register_buffer("lora_a", None, persistent=False)
        self.register_buffer("lora_b", None, persistent=False)

    def create_lora_weights(self, *_args, **_kwargs) -> None:
        """Match the manager's layer protocol when its rank buffer grows."""
        self.reset_lora(0)

    def reset_lora(self, index: int) -> None:
        if index != 0:
            raise IndexError(f"HV1.5 torch-linear LoRA only supports adapter slot 0, got {index}")
        self.lora_a = None
        self.lora_b = None

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
    ) -> None:
        if index != 0:
            raise IndexError(f"HV1.5 torch-linear LoRA only supports adapter slot 0, got {index}")
        if not (
            lora_a.ndim == lora_b.ndim == 2
            and lora_a.shape[1] == self.base_layer.in_features
            and lora_b.shape == (self.base_layer.out_features, lora_a.shape[0])
        ):
            raise ValueError(
                "HV1.5 torch-linear LoRA shape mismatch: "
                f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}; expected "
                f"A=[rank, {self.base_layer.in_features}], "
                f"B=[{self.base_layer.out_features}, rank]"
            )
        self.lora_a = lora_a.detach().to(self.base_layer.weight)
        self.lora_b = lora_b.detach().to(self.base_layer.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(hidden_states)
        if self.lora_a is None or self.lora_b is None:
            return result
        lora_input = hidden_states.to(self.lora_a.dtype)
        lora_hidden = torch.nn.functional.linear(lora_input, self.lora_a)
        return result + torch.nn.functional.linear(lora_hidden, self.lora_b)


def patch_hv15_refiner_torch_linear_lora() -> None:
    """Include HV1.5 token-refiner ``nn.Linear`` layers in vLLM's LoRA policy."""
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.lora.utils import replace_submodule
    from vllm_omni.diffusion.lora.utils import _match_target_modules
    from vllm_omni.diffusion.models.hunyuan_video.pipeline_hunyuan_video_1_5 import (
        HunyuanVideo15Pipeline,
    )

    original_replace = DiffusionLoRAManager._replace_layers_with_lora
    if getattr(original_replace, "_unirl_hv15_refiner_torch_linear_lora", False):
        return

    @wraps(original_replace)
    def _patched_replace(self, peft_helper):
        original_replace(self, peft_helper)
        if not isinstance(self.pipeline, HunyuanVideo15Pipeline):
            return
        transformer = self.pipeline.transformer

        blocks = transformer.context_embedder.token_refiner.refiner_blocks
        prefix = "context_embedder.token_refiner.refiner_blocks"
        target_modules = peft_helper.target_modules
        target_pattern = target_modules if isinstance(target_modules, str) and target_modules else None
        target_list = target_modules if isinstance(target_modules, list) and target_modules else None
        if target_pattern is not None:
            import regex as re

        matched = []
        for block_index, block in enumerate(blocks):
            for target_path, module in block.named_modules(remove_duplicate=False):
                module_name = f"{prefix}.{block_index}.{target_path}"
                full_module_name = f"transformer.{module_name}"
                matches_target = (
                    re.search(target_pattern, full_module_name) is not None
                    if target_pattern is not None
                    else (target_list is None or _match_target_modules(full_module_name, target_list))
                )
                if (
                    target_path
                    and matches_target
                    and isinstance(
                        module,
                        (torch.nn.Linear, HV15TorchLinearWithLoRA, BaseLayerWithLoRA),
                    )
                ):
                    matched.append((module_name, full_module_name, module))

        newly_wrapped = 0
        for module_name, full_module_name, module in matched:
            if isinstance(module, torch.nn.Linear):
                module = HV15TorchLinearWithLoRA(module)
                replace_submodule(transformer, module_name, module)
                newly_wrapped += 1
            self._lora_modules[full_module_name] = module

        if newly_wrapped:
            logger.info(
                "Wrapped %d HV1.5 token-refiner nn.Linear layers for online LoRA",
                newly_wrapped,
            )

    _patched_replace._unirl_hv15_refiner_torch_linear_lora = True
    DiffusionLoRAManager._replace_layers_with_lora = _patched_replace


def patch_lora_request_passthrough() -> None:
    """Forward ``lora_request`` through ``Omni.generate`` to ``engine.add_request``."""
    from vllm_omni.engine.async_omni_engine import AsyncOmniEngine
    from vllm_omni.entrypoints.omni import Omni

    _orig_omni_generate = Omni.generate
    if not getattr(_orig_omni_generate, "_unirl_lora_request_passthrough", False):

        def _patched_omni_generate(self, *args, lora_request=None, _orig=_orig_omni_generate, **kwargs):
            self.engine._unirl_pending_lora_request = lora_request
            py_generator = kwargs.get("py_generator", False)
            try:
                result = _orig(self, *args, **kwargs)
            except Exception:
                self.engine._unirl_pending_lora_request = None
                raise
            if py_generator:

                def _wrapped(gen, engine):
                    try:
                        yield from gen
                    finally:
                        engine._unirl_pending_lora_request = None

                return _wrapped(result, self.engine)
            self.engine._unirl_pending_lora_request = None
            return result

        _patched_omni_generate._unirl_lora_request_passthrough = True  # type: ignore[attr-defined]
        Omni.generate = _patched_omni_generate

    _orig_add_request = AsyncOmniEngine.add_request
    if not getattr(_orig_add_request, "_unirl_lora_request_passthrough", False):

        def _patched_add_request(self, *args, lora_request=None, _orig=_orig_add_request, **kwargs):
            if lora_request is None:
                lora_request = getattr(self, "_unirl_pending_lora_request", None)
            return _orig(self, *args, lora_request=lora_request, **kwargs)

        _patched_add_request._unirl_lora_request_passthrough = True  # type: ignore[attr-defined]
        AsyncOmniEngine.add_request = _patched_add_request


def patch_sigmas_passthrough() -> None:
    """Monkey-patch HunyuanImage3Pipeline to forward custom sigmas to DiT scheduler."""
    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        HunyuanImage3Pipeline,
        HunyuanImage3Text2ImagePipeline,
    )

    _orig_outer_forward = HunyuanImage3Pipeline.forward
    if not getattr(_orig_outer_forward, "_unirl_sigmas_passthrough", False):

        def _patched_outer_forward(self, req, *args, _orig=_orig_outer_forward, **kwargs):
            self.unirl_sigmas = req.sampling_params.sigmas
            try:
                return _orig(self, req, *args, **kwargs)
            finally:
                self.unirl_sigmas = None

        _patched_outer_forward._unirl_sigmas_passthrough = True  # type: ignore[attr-defined]
        HunyuanImage3Pipeline.forward = _patched_outer_forward

    _orig_inner_call = HunyuanImage3Text2ImagePipeline.__call__
    if not getattr(_orig_inner_call, "_unirl_sigmas_passthrough", False):

        def _patched_inner_call(self, *args, _orig=_orig_inner_call, **kwargs):
            sigmas = getattr(self.model, "unirl_sigmas", None)
            if sigmas is not None and "sigmas" not in kwargs:
                kwargs["sigmas"] = sigmas
            return _orig(self, *args, **kwargs)

        _patched_inner_call._unirl_sigmas_passthrough = True  # type: ignore[attr-defined]
        HunyuanImage3Text2ImagePipeline.__call__ = _patched_inner_call


class VLLMOmniHijack:
    """Monkey-patches vllm-omni internals to support in-memory LoRA tensors."""

    @staticmethod
    def hijack() -> None:
        wrap_mp_process_for_children()

        patch_dit_lora_loader()
        patch_dit_hi3_lora_weights()
        patch_ar_lora_loader()
        patch_ar_merged_lora_fused_tensor()
        patch_fp32_skip()
        patch_hv15_packed_lora_mapping()
        patch_hv15_refiner_torch_linear_lora()
        patch_lora_request_passthrough()
        patch_sigmas_passthrough()
        patch_moe_workspace_pool()


__all__ = [
    "OmniTensorLoRARequest",
    "VLLMOmniHijack",
    "patch_hv15_packed_lora_mapping",
    "patch_hv15_refiner_torch_linear_lora",
    "patch_sigmas_passthrough",
]
