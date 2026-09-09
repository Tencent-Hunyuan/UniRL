"""Bound CUDA load peaks while materializing CPU-offloaded FastVideo transformers; contract in README.md."""

from __future__ import annotations

from functools import wraps

import torch

_DIT_MODULES = frozenset({"transformer", "transformer_2"})


def patch_offload() -> None:
    """Keep layerwise-offloaded DiT checkpoints off CUDA during materialization."""
    from fastvideo.models.loader import fsdp_load
    from fastvideo.models.loader.component_loader import PipelineComponentLoader

    original_state_load = fsdp_load.load_model_from_full_model_state_dict
    if not getattr(original_state_load, "_unirl_fastvideo_offload", False):

        @wraps(original_state_load)
        def load_state(model, full_sd_iterator, device, param_dtype, *args, **kwargs):
            if bool(kwargs.get("cpu_offload", False)):
                device = torch.device("cpu")
            return original_state_load(model, full_sd_iterator, device, param_dtype, *args, **kwargs)

        load_state._unirl_fastvideo_offload = True
        fsdp_load.load_model_from_full_model_state_dict = load_state

    original = PipelineComponentLoader.load_module
    if getattr(original, "_unirl_fastvideo_offload", False):
        return

    @wraps(original)
    def load_module(module_name, component_model_path, transformers_or_diffusers, fastvideo_args):
        # Upstream stages the whole DiT on GPU before attaching the layerwise hooks, so two
        # 14B experts transiently exceed the device even with layerwise offload requested.
        force_cpu_load = (
            module_name in _DIT_MODULES
            and bool(getattr(fastvideo_args, "dit_layerwise_offload", False))
            and not bool(getattr(fastvideo_args, "dit_cpu_offload", False))
        )
        if force_cpu_load:
            fastvideo_args.dit_cpu_offload = True
        try:
            module = original(module_name, component_model_path, transformers_or_diffusers, fastvideo_args)
            if force_cpu_load and torch.cuda.is_available():
                # The hooks already swapped every ModuleList parameter for a 0-shape CUDA
                # placeholder; this moves only the remaining stem/head parameters.
                module.to(torch.device("cuda", torch.cuda.current_device()))
        finally:
            if force_cpu_load:
                fastvideo_args.dit_cpu_offload = False
        offloaded = force_cpu_load or (
            (module_name in _DIT_MODULES and bool(getattr(fastvideo_args, "dit_cpu_offload", False)))
            or (module_name == "text_encoder" and bool(getattr(fastvideo_args, "text_encoder_cpu_offload", False)))
            or (module_name == "image_encoder" and bool(getattr(fastvideo_args, "image_encoder_cpu_offload", False)))
            or (module_name == "vae" and bool(getattr(fastvideo_args, "vae_cpu_offload", False)))
        )
        if offloaded and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return module

    load_module._unirl_fastvideo_offload = True
    PipelineComponentLoader.load_module = staticmethod(load_module)


__all__ = ["patch_offload"]
