"""Inject the driver-authoritative rollout IO fields the fork added to sglang."""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import field

logger = logging.getLogger(__name__)

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
_VALIDATE_SENTINEL = "_unirl_sampling_io_validate"
_REQ_FIELD = "denoise_seeds"


def patch_sampling_io() -> None:
    """Inject rollout IO fields onto SamplingParams/Req and copy them in prepare_request."""
    import sglang.multimodal_gen.configs.sample.sampling_params as sp_mod
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
    import sglang.multimodal_gen.runtime.pipelines_core.schedule_batch as sb_mod

    SamplingParams = sp_mod.SamplingParams

    _install_sampling_params_fields(SamplingParams)

    _wrap_from_user_sampling_params_args(SamplingParams)

    _install_req_denoise_seeds(sb_mod)

    _wrap_prepare_request(utils_mod, SamplingParams)

    _install_json_safe_tensor_guard(sp_mod)

    # Allow condition_image to satisfy I2I validation when image_path is absent.
    _wrap_validate_with_pipeline_config(SamplingParams)

    # Index condition_image per prompt because upstream does not.
    _wrap_diff_generator_generate()

    # Preprocess PIL-only requests so Edit-Plus receives VAE image sizes.
    _wrap_input_validation_condition_image()


def _install_json_safe_tensor_guard(sp_mod) -> None:
    """Make ``sampling_params._json_safe`` tolerate ``torch.Tensor`` values."""
    import torch

    orig = getattr(sp_mod, "_json_safe", None)
    if orig is None or getattr(orig, "_unirl_tensor_safe", False):
        return

    def _json_safe(obj):
        if torch.is_tensor(obj):
            return f"<tensor:{tuple(obj.shape)}:{obj.dtype}>"
        if obj.__class__.__name__ == "Image" or obj.__class__.__module__.startswith("PIL."):
            return f"<pil:{getattr(obj, 'size', None)}:{getattr(obj, 'mode', None)}>"
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_safe(v) for v in obj]
        return orig(obj)

    _json_safe._unirl_tensor_safe = True  # type: ignore[attr-defined]
    sp_mod._json_safe = _json_safe


_CONDITION_IMAGE_PATH_SENTINEL = "<unirl:condition_image>"


def _wrap_validate_with_pipeline_config(SamplingParams) -> None:
    """AROUND-wrap ``_validate_with_pipeline_config`` so ``condition_image`` satisfies the I2I ``image_path`` need."""
    orig = SamplingParams.__dict__.get("_validate_with_pipeline_config")
    if orig is None:
        return  # pragma: no cover - upstream method missing
    if getattr(orig, _VALIDATE_SENTINEL, False):
        return

    def _validate_with_pipeline_config(self, pipeline_config, __orig=orig):
        condition_image = getattr(self, "condition_image", None)
        if condition_image is None or getattr(self, "image_path", None) is not None:
            return __orig(self, pipeline_config)
        self.image_path = _CONDITION_IMAGE_PATH_SENTINEL
        try:
            return __orig(self, pipeline_config)
        finally:
            self.image_path = None

    setattr(_validate_with_pipeline_config, _VALIDATE_SENTINEL, True)
    SamplingParams._validate_with_pipeline_config = _validate_with_pipeline_config


_GEN_SENTINEL = "_unirl_diff_gen_index"


def _wrap_diff_generator_generate() -> None:
    """AROUND-wrap ``DiffGenerator.generate`` to index ``condition_image`` per prompt."""
    try:
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )
    except Exception:  # pragma: no cover - environment dependent
        return

    orig = DiffGenerator.__dict__.get("generate")
    if orig is None:
        return
    if getattr(orig, _GEN_SENTINEL, False):
        return

    def generate(self, sampling_params_kwargs=None, *args, **kwargs):
        ci = (sampling_params_kwargs or {}).get("condition_image")
        if isinstance(ci, list) and len(ci) > 1:
            _local.condition_image_per_prompt = ci
            _local.condition_image_idx = 0
        else:
            _local.condition_image_per_prompt = None
            _local.condition_image_idx = 0
        try:
            return orig(self, sampling_params_kwargs, *args, **kwargs)
        finally:
            # Always clear so a later T2I call in the same thread can't pick up a stale Edit-Plus stash.
            _local.condition_image_per_prompt = None
            _local.condition_image_idx = 0

    setattr(generate, _GEN_SENTINEL, True)
    DiffGenerator.generate = generate


_IVL_SENTINEL = "_unirl_ivl_cond_img"


def _wrap_input_validation_condition_image() -> None:
    """AROUND-wrap ``InputValidationStage.forward`` to preprocess ``condition_image`` when ``image_path`` is None."""
    try:
        import sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation as ivl_mod
    except ImportError:
        return  # pragma: no cover - upstream module missing

    IVL = getattr(ivl_mod, "InputValidationStage", None)
    if IVL is None:
        return  # pragma: no cover

    orig_forward = IVL.__dict__.get("forward")
    if orig_forward is None or getattr(orig_forward, _IVL_SENTINEL, False):
        return

    def forward(self, batch, server_args, __orig=orig_forward):
        batch = __orig(self, batch, server_args)

        condition_image = getattr(batch, "condition_image", None)
        image_path = getattr(batch, "image_path", None)
        vae_image_sizes = getattr(batch, "vae_image_sizes", None)
        if condition_image is None or image_path is not None or vae_image_sizes is not None:
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
    IVL.forward = forward


def _make_dataclass_field(name: str, default, type_str: str):
    """Build a dataclasses.Field equivalent to ``name: type = default`` post-hoc."""
    f = field(default=default)
    f.name = name
    f.type = type_str
    f._field_type = dataclasses._FIELD
    return f


def _iter_subclasses(cls):
    """Yield ``cls`` and all of its subclasses (recursively, de-duplicated)."""
    seen = {cls}
    stack = [cls]
    yield cls
    while stack:
        parent = stack.pop()
        for child in parent.__subclasses__():
            if child not in seen:
                seen.add(child)
                stack.append(child)
                yield child


def _install_sampling_params_fields(SamplingParams) -> None:
    """Register the four fields on SamplingParams and every live subclass."""
    for cls in _iter_subclasses(SamplingParams):
        _register_and_wrap_init(cls)


def _register_and_wrap_init(cls) -> None:
    """Add the four fields to ``cls`` and wrap its ``__init__`` to accept them."""
    own_fields = cls.__dict__.get("__dataclass_fields__")
    if own_fields is None:
        own_fields = dict(getattr(cls, "__dataclass_fields__", {}))
        cls.__dataclass_fields__ = own_fields

    for name, (default, type_str) in _SP_INJECT_FIELDS.items():
        if name not in own_fields:
            own_fields[name] = _make_dataclass_field(name, default, type_str)
        if name not in cls.__dict__:
            setattr(cls, name, default)

    orig_init = cls.__dict__.get("__init__")
    if orig_init is None or getattr(orig_init, _SP_INIT_SENTINEL, False):
        return

    inject_names = tuple(_SP_INJECT_FIELDS)

    def __init__(self, *args, __orig_init=orig_init, **kwargs):
        extra = {k: kwargs.pop(k) for k in inject_names if k in kwargs}
        __orig_init(self, *args, **kwargs)
        for k, v in extra.items():
            object.__setattr__(self, k, v)

    setattr(__init__, _SP_INIT_SENTINEL, True)
    cls.__init__ = __init__


def _wrap_from_user_sampling_params_args(SamplingParams) -> None:
    """AROUND-wrap the staticmethod that constructs the user SamplingParams."""
    orig = SamplingParams.__dict__.get("from_user_sampling_params_args")
    if orig is None:
        raise AttributeError("SamplingParams.from_user_sampling_params_args missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _SP_INIT_SENTINEL, False):
        return

    def from_user_sampling_params_args(model_path, server_args, *args, **kwargs):
        _install_sampling_params_fields(SamplingParams)
        return raw(model_path, server_args, *args, **kwargs)

    setattr(from_user_sampling_params_args, _SP_INIT_SENTINEL, True)
    SamplingParams.from_user_sampling_params_args = staticmethod(from_user_sampling_params_args)


def _install_req_denoise_seeds(sb_mod) -> None:
    """Add ``denoise_seeds`` as a first-class field on upstream ``Req``."""
    Req = sb_mod.Req
    own_fields = Req.__dict__.get("__dataclass_fields__")
    if own_fields is None:  # pragma: no cover - Req is a dataclass, always present
        own_fields = dict(getattr(Req, "__dataclass_fields__", {}))
        Req.__dataclass_fields__ = own_fields
    if _REQ_FIELD not in own_fields:
        own_fields[_REQ_FIELD] = _make_dataclass_field(_REQ_FIELD, None, "list[str] | None")

    spf = getattr(sb_mod, "SAMPLING_PARAMS_FIELDS", None)
    if isinstance(spf, set):
        spf.update(_SP_INJECT_FIELDS)


def _wrap_prepare_request(utils_mod, SamplingParams) -> None:
    """AROUND-wrap ``prepare_request`` to copy the four IO fields onto the Req."""
    orig = utils_mod.prepare_request
    if getattr(orig, _PREP_SENTINEL, False):
        return

    def prepare_request(server_args, sampling_params, *args, **kwargs):
        import torch

        req = orig(server_args, sampling_params, *args, **kwargs)

        sigmas = getattr(sampling_params, "sigmas", None)
        if sigmas is not None:
            req.sigmas = sigmas

        timesteps = getattr(sampling_params, "timesteps", None)
        if timesteps is not None:
            req.timesteps = torch.as_tensor(timesteps, dtype=torch.float32)

        initial_noise = getattr(sampling_params, "initial_noise", None)
        if initial_noise is not None:
            req.latents = initial_noise

        initial_audio_noise = getattr(sampling_params, "initial_audio_noise", None)
        if initial_audio_noise is not None:
            req.audio_latents = initial_audio_noise

        denoise_seeds = getattr(sampling_params, "denoise_seeds", None)
        if denoise_seeds is not None:
            req.denoise_seeds = denoise_seeds

        max_sequence_length = getattr(sampling_params, "max_sequence_length", None)
        if max_sequence_length is not None:
            req.max_sequence_length = int(max_sequence_length)

        condition_image = getattr(sampling_params, "condition_image", None)
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

    # Rebind imported prepare_request aliases; patching the source module alone is ineffective.
    import sys

    for _mod in list(sys.modules.values()):
        try:
            if getattr(_mod, "prepare_request", None) is orig:
                _mod.prepare_request = prepare_request
        except Exception:  # pragma: no cover - defensive
            pass
