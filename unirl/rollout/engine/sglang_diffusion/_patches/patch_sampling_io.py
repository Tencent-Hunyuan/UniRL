"""Inject the driver-authoritative rollout IO fields the fork added to sglang."""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import field

_local = threading.local()

_SP_INJECT_FIELDS = {
    "sigmas": (None, "list[float] | None"),
    "timesteps": (None, "list[float] | None"),
    "initial_noise": (None, "torch.Tensor | None"),
    "initial_audio_noise": (None, "torch.Tensor | None"),
    "denoise_seeds": (None, "list[str] | None"),
    "max_sequence_length": (None, "int | None"),
    "return_prompt_embeds": (False, "bool"),
    "return_negative_prompt_embeds": (False, "bool"),
    "condition_image": (None, "Any"),
}

_SP_INIT_SENTINEL = "_unirl_sampling_io_init"
_PREP_SENTINEL = "_unirl_sampling_io_prepare"
_EXPAND_SENTINEL = "_unirl_sampling_io_expand"
_VALIDATE_SENTINEL = "_unirl_sampling_io_validate"
_GEN_SENTINEL = "_unirl_diff_gen_index"
_IVL_SENTINEL = "_unirl_ivl_cond_img"
_REQ_FIELD = "denoise_seeds"
_CONDITION_IMAGE_PATH_SENTINEL = "<unirl:condition_image>"


def _make_dataclass_field(name: str, default, type_str: str):
    """Build a dataclasses.Field equivalent to ``name: type = default`` post-hoc."""
    f = field(default=default)
    f.name = name
    f.type = type_str
    f._field_type = dataclasses._FIELD
    return f


def _install_sampling_params_fields(SamplingParams) -> None:
    """Register UniRL fields on SamplingParams and every live subclass."""
    inject_names = tuple(_SP_INJECT_FIELDS)
    seen = {SamplingParams}
    stack = [SamplingParams]
    while stack:
        cls = stack.pop()
        for child in cls.__subclasses__():
            if child not in seen:
                seen.add(child)
                stack.append(child)

        own_fields = cls.__dict__.get("__dataclass_fields__")
        if own_fields is None:
            own_fields = dict(cls.__dataclass_fields__)
            cls.__dataclass_fields__ = own_fields

        for name, (default, type_str) in _SP_INJECT_FIELDS.items():
            if name not in own_fields:
                own_fields[name] = _make_dataclass_field(name, default, type_str)
            if name not in cls.__dict__:
                setattr(cls, name, default)

        orig_init = cls.__dict__.get("__init__")
        if orig_init is None or getattr(orig_init, _SP_INIT_SENTINEL, False):
            continue

        def __init__(self, *args, __orig_init=orig_init, **kwargs):
            extra = {k: kwargs.pop(k) for k in inject_names if k in kwargs}
            __orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                object.__setattr__(self, k, v)

        setattr(__init__, _SP_INIT_SENTINEL, True)
        cls.__init__ = __init__


def _partition_prompt_tensor(value, *, prompt_index: int, num_prompts: int, num_outputs: int, field: str):
    """Select one prompt's driver tensor before upstream expands its outputs."""
    import torch

    if value is None:
        return None
    if not torch.is_tensor(value) or value.dim() < 1:
        raise TypeError(f"{field} must be a tensor with a batch dimension")
    count = value.shape[0]
    total = num_prompts * num_outputs
    if count == 1:
        return value
    if count == total:
        start = prompt_index * num_outputs
        return value[start : start + num_outputs]
    if count == num_prompts:
        return value[prompt_index : prompt_index + 1]
    raise ValueError(f"{field} batch dim {count} must be 1, num_prompts={num_prompts}, or total_outputs={total}")


def _slice_output_tensor(value, *, output_index: int, num_outputs: int, field: str):
    """Select one expanded output's driver tensor while preserving batch dim."""
    import torch

    if value is None:
        return None
    if not torch.is_tensor(value) or value.dim() < 1:
        raise TypeError(f"{field} must be a tensor with a batch dimension")
    if value.shape[0] == 1:
        return value
    if value.shape[0] != num_outputs:
        raise ValueError(f"{field} prompt batch dim {value.shape[0]} must be 1 or num_outputs_per_prompt={num_outputs}")
    return value[output_index : output_index + 1]


def patch_sampling_io() -> None:
    """Inject rollout IO fields onto SamplingParams/Req and copy them in prepare_request."""
    import sglang.multimodal_gen.configs.sample.sampling_params as sp_mod
    import sglang.multimodal_gen.runtime.entrypoints.diffusion_generator as dg_mod
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
    import sglang.multimodal_gen.runtime.pipelines_core.request_utils as req_utils_mod
    import sglang.multimodal_gen.runtime.pipelines_core.schedule_batch as sb_mod
    import sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation as ivl_mod
    import torch

    SamplingParams = sp_mod.SamplingParams
    DiffGenerator = dg_mod.DiffGenerator
    InputValidationStage = ivl_mod.InputValidationStage

    _install_sampling_params_fields(SamplingParams)

    raw_from_user = SamplingParams.from_user_sampling_params_args
    if not getattr(raw_from_user, _SP_INIT_SENTINEL, False):

        def from_user_sampling_params_args(model_path, server_args, *args, **kwargs):
            _install_sampling_params_fields(SamplingParams)
            return raw_from_user(model_path, server_args, *args, **kwargs)

        setattr(from_user_sampling_params_args, _SP_INIT_SENTINEL, True)
        SamplingParams.from_user_sampling_params_args = staticmethod(from_user_sampling_params_args)

    req_fields = sb_mod.Req.__dataclass_fields__
    if _REQ_FIELD not in req_fields:
        req_fields[_REQ_FIELD] = _make_dataclass_field(_REQ_FIELD, None, "list[str] | None")
    sb_mod.SAMPLING_PARAMS_FIELDS.update(_SP_INJECT_FIELDS)

    orig_prepare = utils_mod.prepare_request
    if not getattr(orig_prepare, _PREP_SENTINEL, False):

        def prepare_request(server_args, sampling_params, *args, **kwargs):
            req = orig_prepare(server_args, sampling_params, *args, **kwargs)

            if sampling_params.sigmas is not None:
                req.sigmas = sampling_params.sigmas
            if sampling_params.timesteps is not None:
                req.timesteps = torch.as_tensor(sampling_params.timesteps, dtype=torch.float32)
            if sampling_params.initial_noise is not None:
                req.latents = sampling_params.initial_noise
            if sampling_params.initial_audio_noise is not None:
                req.audio_latents = sampling_params.initial_audio_noise
            if sampling_params.denoise_seeds is not None:
                req.denoise_seeds = sampling_params.denoise_seeds

            condition_image = sampling_params.condition_image
            if condition_image is not None:
                stash = getattr(_local, "condition_image_per_prompt", None)
                if isinstance(stash, list) and len(stash) > 1:
                    idx = getattr(_local, "condition_image_idx", 0)
                    if idx >= len(stash):
                        raise RuntimeError(
                            f"prepare_request: condition_image index {idx} >= "
                            f"stash length {len(stash)} — generate() prompt count "
                            f"mismatch. This is a UniRL patch bug."
                        )
                    condition_image = stash[idx]
                    _local.condition_image_idx = idx + 1
                req.condition_image = condition_image

            return req

        setattr(prepare_request, _PREP_SENTINEL, True)
        utils_mod.prepare_request = prepare_request
        dg_mod.prepare_request = prepare_request

    orig_expand = utils_mod.expand_request_outputs
    if not getattr(orig_expand, _EXPAND_SENTINEL, False):

        def expand_request_outputs(req, *, num_prompts=1, prompt_index=0, **kwargs):
            num_outputs = req.num_outputs_per_prompt
            if not 0 <= prompt_index < num_prompts:
                raise IndexError(f"prompt_index={prompt_index} is outside num_prompts={num_prompts}")

            req.latents = _partition_prompt_tensor(
                req.latents,
                prompt_index=prompt_index,
                num_prompts=num_prompts,
                num_outputs=num_outputs,
                field="initial_noise",
            )
            req.audio_latents = _partition_prompt_tensor(
                req.audio_latents,
                prompt_index=prompt_index,
                num_prompts=num_prompts,
                num_outputs=num_outputs,
                field="initial_audio_noise",
            )
            denoise_seeds = req.denoise_seeds
            if denoise_seeds is not None:
                if not isinstance(denoise_seeds, (list, tuple)):
                    raise TypeError("denoise_seeds must be a list or tuple")
                total = num_prompts * num_outputs
                if len(denoise_seeds) != total:
                    raise ValueError(f"denoise_seeds length {len(denoise_seeds)} must equal total_outputs={total}")
                start = prompt_index * num_outputs
                req.denoise_seeds = list(denoise_seeds[start : start + num_outputs])

            expanded = orig_expand(
                req,
                num_prompts=num_prompts,
                prompt_index=prompt_index,
                **kwargs,
            )
            for output_index, output_req in enumerate(expanded):
                output_req.latents = _slice_output_tensor(
                    output_req.latents,
                    output_index=output_index,
                    num_outputs=num_outputs,
                    field="initial_noise",
                )
                output_req.audio_latents = _slice_output_tensor(
                    output_req.audio_latents,
                    output_index=output_index,
                    num_outputs=num_outputs,
                    field="initial_audio_noise",
                )
                seeds = output_req.denoise_seeds
                if seeds is not None:
                    if len(seeds) != num_outputs:
                        raise ValueError(
                            f"denoise_seeds prompt length {len(seeds)} must equal num_outputs_per_prompt={num_outputs}"
                        )
                    output_req.denoise_seeds = [seeds[output_index]]
            return expanded

        setattr(expand_request_outputs, _EXPAND_SENTINEL, True)
        req_utils_mod.expand_request_outputs = expand_request_outputs
        utils_mod.expand_request_outputs = expand_request_outputs
        dg_mod.expand_request_outputs = expand_request_outputs
        ivl_mod.expand_request_outputs = expand_request_outputs

    orig_json_safe = sp_mod._json_safe
    if not getattr(orig_json_safe, "_unirl_tensor_safe", False):

        def _json_safe(obj):
            if torch.is_tensor(obj):
                return f"<tensor:{tuple(obj.shape)}:{obj.dtype}>"
            if obj.__class__.__name__ == "Image" or obj.__class__.__module__.startswith("PIL."):
                return f"<pil:{getattr(obj, 'size', None)}:{getattr(obj, 'mode', None)}>"
            if isinstance(obj, dict):
                return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [_json_safe(v) for v in obj]
            return orig_json_safe(obj)

        _json_safe._unirl_tensor_safe = True  # type: ignore[attr-defined]
        sp_mod._json_safe = _json_safe

    orig_validate = SamplingParams._validate_with_pipeline_config
    if not getattr(orig_validate, _VALIDATE_SENTINEL, False):

        def _validate_with_pipeline_config(self, pipeline_config):
            if self.condition_image is None or self.image_path is not None:
                return orig_validate(self, pipeline_config)
            self.image_path = _CONDITION_IMAGE_PATH_SENTINEL
            try:
                return orig_validate(self, pipeline_config)
            finally:
                self.image_path = None

        setattr(_validate_with_pipeline_config, _VALIDATE_SENTINEL, True)
        SamplingParams._validate_with_pipeline_config = _validate_with_pipeline_config

    orig_generate = DiffGenerator.generate
    if not getattr(orig_generate, _GEN_SENTINEL, False):

        def generate(self, sampling_params_kwargs=None, *args, **kwargs):
            ci = (sampling_params_kwargs or {}).get("condition_image")
            if isinstance(ci, list) and len(ci) > 1:
                _local.condition_image_per_prompt = ci
                _local.condition_image_idx = 0
            else:
                _local.condition_image_per_prompt = None
                _local.condition_image_idx = 0
            try:
                return orig_generate(self, sampling_params_kwargs, *args, **kwargs)
            finally:
                _local.condition_image_per_prompt = None
                _local.condition_image_idx = 0

        setattr(generate, _GEN_SENTINEL, True)
        DiffGenerator.generate = generate

    orig_ivl_forward = InputValidationStage.forward
    if not getattr(orig_ivl_forward, _IVL_SENTINEL, False):

        def forward(self, batch, server_args):
            batch = orig_ivl_forward(self, batch, server_args)

            condition_image = batch.condition_image
            if condition_image is None or batch.image_path is not None or batch.vae_image_sizes is not None:
                return batch

            img = condition_image[-1] if isinstance(condition_image, list) else condition_image
            condition_image_width = img.width
            condition_image_height = img.height
            batch.original_condition_image_size = (condition_image_width, condition_image_height)

            saved_height = batch.height
            saved_width = batch.width
            self.preprocess_condition_image(batch, server_args, condition_image_width, condition_image_height)
            batch.height = saved_height
            batch.width = saved_width
            return batch

        setattr(forward, _IVL_SENTINEL, True)
        InputValidationStage.forward = forward
