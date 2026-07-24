"""Inject the driver-authoritative rollout IO fields the fork added to sglang.

UniRL is the single source of truth for the denoising schedule, the
initial latent ``x_T``, and per-sample SDE noise grouping. To express that over
``DiffGenerator.generate(sampling_params_kwargs=...)`` the rollout engine
(``rollout/engine/sglang/request.py``) ships four extra keys inside
``sampling_params_kwargs``:

* ``sigmas`` -- driver-pinned σ schedule (GRPO σ-consistency; train-side replay
  recomputes log-probs against the *same* σ, so any drift breaks iter-0 ratios).
* ``timesteps`` -- optional explicit timestep override forwarded to
  ``scheduler.set_timesteps`` (caller owns the unit mapping).
* ``initial_noise`` -- driver-authoritative ``x_T`` (UniRL ``NoiseRecipe``),
  landed on ``Req.latents`` so ``LatentPreparationStage`` consumes it verbatim.
* ``denoise_seeds`` -- per-sample SDE determinism keys (same-group samples
  share noise; distinct groups are per-sample unique).

Stock upstream lacks all four on ``SamplingParams`` and never copies them onto
``Req``; ``Req.denoise_seeds`` does not exist upstream at all. The fork
(``sglang-drl`` @ ``e9b570654..HEAD``) carries them in source:

* ``sigmas`` / ``timesteps`` are real ``SamplingParams`` dataclass fields and are
  copied onto ``Req`` in ``prepare_request`` (``entrypoints/utils.py``).
* ``initial_noise`` / ``denoise_seeds`` are popped from ``sampling_params_kwargs``
  in the fork's ``diffusion_generator.generate`` and assigned **after**
  ``prepare_request`` as ``req.latents`` / ``req.denoise_seeds``.

We cannot edit sglang source, and upstream ``DiffGenerator.generate`` does NOT
pop these (it forwards the whole kwargs dict into
``SamplingParams.from_user_sampling_params_args`` and then runs
``dataclasses.replace`` per prompt). So this patch RE-HOMES the fork's split
behaviour into two upstream chokepoints that the UniRL path always hits:

1. **Field injection on ``SamplingParams`` (+ every subclass).** Make all four
   acceptable construction kwargs with the fork's defaults/types, registered as
   genuine dataclass fields so they survive ``dataclasses.replace`` (upstream
   ``generate`` rebuilds the params per prompt -- a non-field attribute would be
   silently dropped) and ``dataclasses.fields`` / ``asdict``. Because every
   ``SamplingParams`` subclass is itself ``@dataclass`` (its own ``__init__`` /
   ``__dataclass_fields__``; a subclass ``__init__`` does NOT call
   ``super().__init__``), the injection is applied to the whole live subclass
   tree, and re-applied lazily at the construction chokepoint so model-specific
   subclasses imported after install are still covered.

2. **AROUND-wrap ``prepare_request``** to copy all four off the (fully merged /
   adjusted) ``sampling_params`` onto the returned ``Req`` -- consolidating the
   fork's ``prepare_request`` (sigmas/timesteps) and ``diffusion_generator``
   (initial_noise->latents, denoise_seeds) assignments into the one site that
   is common to upstream's ``generate`` / OpenAI / HTTP entrypoints.

Plus a one-field injection on ``Req`` for ``denoise_seeds`` (the only one of
the four missing from upstream ``Req``; ``latents`` / ``sigmas`` / ``timesteps``
already exist), so the assignment lands as a first-class ``Req`` attribute rather
than being delegated onto ``sampling_params`` by ``Req.__setattr__``.

Idempotent; setattr / field-injection / AROUND-wrap only -- no sglang source edits.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import field

logger = logging.getLogger(__name__)

# Thread-local stash for the per-prompt ``condition_image`` index.
#
# Upstream ``DiffGenerator.generate`` (diffusion_generator.py:209-221) loops
# over prompts and builds one ``Req`` per prompt via ``dataclasses.replace``
# + ``prepare_request``. It indexes ``image_path`` per prompt via
# ``_resolve_image_paths_per_prompt`` but does NOT index ``condition_image`` --
# so every per-prompt ``sampling_params`` carries the WHOLE list, and
# ``InputValidationStage.preprocess_condition_image`` (input_validation.py:117)
# then uses ``batch.condition_image[-1]`` as the source image. Every prompt in
# a multi-prompt batch ends up conditioned on the LAST source image.
#
# We fix this by stashing the list in thread-local state at the start of
# ``generate`` and indexing it per Req in ``_wrap_prepare_request`` (which
# upstream calls once per prompt, in prompt order, in the same thread).
_local = threading.local()

# Fork field name -> (default, human-readable type) for SamplingParams injection.
# Defaults/types mirror the fork's sampling_params.py diff (sigmas/timesteps real
# fields) and the fork's construction-site pops (initial_noise/denoise_seeds).
#
# return_prompt_embeds / return_negative_prompt_embeds are the fork's
# conditions-path opt-in flags (sampling_params.py: bool, default False). They
# must be genuine SamplingParams fields so (a) DiffGenerator.generate's per-prompt
# ``dataclasses.replace`` preserves them and (b) they reach the worker as
# ``req.return_prompt_embeds`` -- Req has no such field, so ``Req.__getattr__``
# delegates the read to ``sampling_params`` (where this injection lands them).
# ``patch_conditions`` reads them off the result Req in ``_req_to_output_batch``.
_SP_INJECT_FIELDS = {
    "sigmas": (None, "list[float] | None"),
    "timesteps": (None, "list[float] | None"),
    "initial_noise": (None, "torch.Tensor | None"),
    "denoise_seeds": (None, "list[str] | None"),
    "return_prompt_embeds": (False, "bool"),
    "return_negative_prompt_embeds": (False, "bool"),
    # Edit-Plus source-image PIL. ``Req.condition_image`` is a real dataclass
    # field (schedule_batch.py:59), but ``SamplingParams`` lacks it, so without
    # injection a PIL passed in sampling kwargs would be dropped at
    # ``SamplingParams`` construction. SGLang's ``InputValidationStage`` checks
    # ``batch.condition_image is not None`` BEFORE ``image_path``
    # (input_validation.py:108), so pre-populating it via this field bypasses
    # the file-path load. ``_wrap_prepare_request`` copies it onto the Req.
    "condition_image": (None, "Any"),
}

# Sentinels.
_SP_INIT_SENTINEL = "_unirl_sampling_io_init"
_PREP_SENTINEL = "_unirl_sampling_io_prepare"
_VALIDATE_SENTINEL = "_unirl_sampling_io_validate"
_REQ_FIELD = "denoise_seeds"


def patch_sampling_io() -> None:
    """Inject rollout IO fields onto SamplingParams/Req and copy them in prepare_request.

    Import-safe (all sglang imports are local) and idempotent.
    """
    import sglang.multimodal_gen.configs.sample.sampling_params as sp_mod
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
    import sglang.multimodal_gen.runtime.pipelines_core.schedule_batch as sb_mod

    SamplingParams = sp_mod.SamplingParams

    # (1) Make the four fields constructible on SamplingParams + every subclass.
    _install_sampling_params_fields(SamplingParams)

    # (1b) Re-apply the injection lazily at the construction chokepoint so that
    # model-specific subclasses (e.g. Flux2KleinSamplingParams) imported *after*
    # install -- the registry resolves and imports them inside
    # from_user_sampling_params_args, before it constructs them -- are covered.
    _wrap_from_user_sampling_params_args(SamplingParams)

    # (2) Add the one missing Req field so denoise_seeds lands on Req directly
    # (latents / sigmas / timesteps already exist upstream).
    _install_req_denoise_seeds(sb_mod)

    # (3) Copy the four onto Req after prepare_request builds it.
    _wrap_prepare_request(utils_mod, SamplingParams)

    # (4) Make sampling_params._json_safe tolerate the injected Tensor fields
    # (initial_noise / timesteps) so _set_output_file_name's params->filename JSON
    # hash (json.dumps(_json_safe(asdict(self)))) doesn't crash on a Tensor.
    _install_json_safe_tensor_guard(sp_mod)

    # (5) Bypass the I2I ``image_path`` requirement when ``condition_image`` is
    # set. Edit-Plus ships the source-image PIL via ``condition_image`` (a Req
    # field populated in ``_wrap_prepare_request``), NOT via ``image_path`` (a
    # file path). Upstream's ``_validate_with_pipeline_config`` raises
    # ``ValueError`` for I2I task types when ``image_path is None`` — without
    # this bypass the first ``generate()`` crashes before the PIL ever reaches
    # ``InputValidationStage`` (which checks ``batch.condition_image is not
    # None`` BEFORE ``image_path`` at input_validation.py:108, so the PIL path
    # is correct once validation passes).
    _wrap_validate_with_pipeline_config(SamplingParams)

    # (6) Index ``condition_image`` per prompt. Upstream's per-prompt loop in
    # ``DiffGenerator.generate`` indexes ``image_path`` but NOT
    # ``condition_image``, so a multi-prompt batch would condition every prompt
    # on the last source image (input_validation.py:117). Stash the list at
    # the start of ``generate`` and index it per Req in
    # ``_wrap_prepare_request``.
    _wrap_diff_generator_generate()

    # (7) Upstream's ``InputValidationStage.forward`` only calls
    # ``preprocess_condition_image`` (which sets ``batch.vae_image_sizes`` via
    # ``config.preprocess_vae_image``) inside the ``if batch.image_path is not
    # None`` branch. Edit-Plus ships the source image as a PIL via
    # ``condition_image`` with ``image_path=None``, so that branch is skipped
    # and ``vae_image_sizes`` stays None → ``_prepare_edit_cond_kwargs`` raises
    # ``TypeError: 'NoneType' object is not iterable``. Wrap forward to invoke
    # ``preprocess_condition_image`` when ``condition_image`` is set but
    # ``image_path`` is not.
    _wrap_input_validation_condition_image()


def _install_json_safe_tensor_guard(sp_mod) -> None:
    """Make ``sampling_params._json_safe`` tolerate ``torch.Tensor`` values.

    Injecting ``initial_noise`` (and ``timesteps``) as ``SamplingParams`` fields
    means ``_set_output_file_name`` -> ``json.dumps(_json_safe(asdict(self)))``
    now meets a Tensor, which upstream ``_json_safe`` passes through unchanged ->
    ``TypeError: Object of type Tensor is not JSON serializable``. That dump only
    feeds a deterministic output-filename hash (and rollout uses
    ``save_output=False``), so replacing Tensors with a shape/dtype placeholder is
    harmless. Recurses with the patched function so Tensors nested in dict/list
    are covered; scalars / Enum / callables delegate to the original. Idempotent.
    """
    import torch

    orig = getattr(sp_mod, "_json_safe", None)
    if orig is None or getattr(orig, "_unirl_tensor_safe", False):
        return

    def _json_safe(obj):
        if torch.is_tensor(obj):
            return f"<tensor:{tuple(obj.shape)}:{obj.dtype}>"
        # PIL.Image.Image (Edit-Plus ``condition_image``) — not JSON
        # serializable; only feeds the output-filename hash, so a placeholder
        # is harmless. ``save_output=False`` in rollout anyway.
        if obj.__class__.__name__ == "Image" or obj.__class__.__module__.startswith("PIL."):
            return f"<pil:{getattr(obj, 'size', None)}:{getattr(obj, 'mode', None)}>"
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_safe(v) for v in obj]
        return orig(obj)

    _json_safe._unirl_tensor_safe = True  # type: ignore[attr-defined]
    sp_mod._json_safe = _json_safe


# ------------------------------------------------------------------ #
# (5) Bypass I2I image_path validation when condition_image is set
# ------------------------------------------------------------------


#: Stand-in image_path used only for the duration of the wrapped validation
#: call — never persisted, never dereferenced (upstream 0.5.12.post1 only
#: None-checks image_path inside _validate_with_pipeline_config).
_CONDITION_IMAGE_PATH_SENTINEL = "<unirl:condition_image>"


def _wrap_validate_with_pipeline_config(SamplingParams) -> None:
    """AROUND-wrap ``_validate_with_pipeline_config`` to let ``condition_image``
    satisfy the I2I ``image_path`` requirement.

    Edit-Plus ships the source-image PIL via the ``condition_image`` field (a
    Req dataclass field, populated by ``_wrap_prepare_request``), NOT via
    ``image_path`` (a file-path string). Upstream's validation raises
    ``ValueError`` for I2I task types when ``image_path is None`` — this would
    crash the first ``generate()`` before the PIL reaches ``InputValidationStage``
    (which checks ``batch.condition_image is not None`` BEFORE ``image_path``,
    so the PIL path is correct once validation passes).

    Instead of re-implementing upstream's checks (which would silently skip
    any validation a future sglang adds to this method), the wrap runs the
    FULL original validation with ``image_path`` temporarily stubbed to a
    sentinel string. Upstream 0.5.12.post1 only None-checks ``image_path``
    here, so the sentinel (a) satisfies ``requires_image_input()`` — the
    intended bypass — and (b) keeps the ``accepts_image_input()`` reject path
    live: a T2I-only task type given a ``condition_image`` now fails
    validation instead of silently ignoring the image. The sentinel is
    restored in ``finally`` and never escapes the call.

    When ``condition_image`` is None (T2I) or ``image_path`` is genuinely set,
    the original validation runs untouched. Idempotent.
    """
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


# ------------------------------------------------------------------ #
# (6) Per-prompt condition_image indexing in DiffGenerator.generate
# ------------------------------------------------------------------


_GEN_SENTINEL = "_unirl_diff_gen_index"


def _wrap_diff_generator_generate() -> None:
    """AROUND-wrap ``DiffGenerator.generate`` to index ``condition_image``
    per prompt.

    Upstream's per-prompt loop (diffusion_generator.py:209-221) indexes
    ``image_path`` via ``_resolve_image_paths_per_prompt`` but NOT
    ``condition_image`` -- every per-prompt ``dataclasses.replace`` carries
    the whole list, and ``InputValidationStage.preprocess_condition_image``
    (input_validation.py:117) then uses ``batch.condition_image[-1]`` as the
    source image. Every prompt in a multi-prompt batch ends up conditioned
    on the LAST source image.

    This wrap stashes the list in thread-local state at the start of
    ``generate``; ``_wrap_prepare_request`` (called once per prompt, in
    prompt order, in the same thread) consumes one element per call.
    Single-prompt path (list len 1 or scalar PIL) is a passthrough.

    Safety: ``generate`` is synchronous in ``local_mode=True`` (the only
    mode UniRL uses) and calls ``prepare_request`` sequentially in the same
    thread, so the thread-local counter is correct. Concurrent ``generate``
    calls in different threads each get their own thread-local slot.

    Idempotent. No-op when ``DiffGenerator`` is unavailable in this
    interpreter (e.g. a CPU-only unit-test process importing only the
    rollout math).
    """
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
        # Stash the per-prompt condition_image list BEFORE the per-prompt loop
        # so _wrap_prepare_request can index it. Reset the counter regardless
        # so a leftover stash from a prior call can't corrupt this one.
        ci = (sampling_params_kwargs or {}).get("condition_image")
        if isinstance(ci, list) and len(ci) > 1:
            _local.condition_image_per_prompt = ci
            _local.condition_image_idx = 0
        else:
            # Clear any stale stash so _wrap_prepare_request falls back to the
            # scalar-PIL passthrough (single prompt or T2I).
            _local.condition_image_per_prompt = None
            _local.condition_image_idx = 0
        try:
            return orig(self, sampling_params_kwargs, *args, **kwargs)
        finally:
            # Always clear so a later T2I call in the same thread can't pick
            # up a stale Edit-Plus stash.
            _local.condition_image_per_prompt = None
            _local.condition_image_idx = 0

    setattr(generate, _GEN_SENTINEL, True)
    DiffGenerator.generate = generate


# ------------------------------------------------------------------ #
# (7) InputValidationStage: call preprocess_condition_image when only
#     condition_image (PIL) is set, no image_path
# ------------------------------------------------------------------


_IVL_SENTINEL = "_unirl_ivl_cond_img"


def _wrap_input_validation_condition_image() -> None:
    """AROUND-wrap ``InputValidationStage.forward`` to invoke
    ``preprocess_condition_image`` when ``condition_image`` is set but
    ``image_path`` is None.

    Upstream's forward (input_validation.py:380-412) only calls
    ``preprocess_condition_image`` (which sets ``batch.vae_image_sizes`` via
    ``config.preprocess_vae_image``) inside the ``if batch.image_path is not
    None`` branch. Edit-Plus ships the source image as a PIL via
    ``condition_image`` with ``image_path=None``, so that branch is skipped
    and ``vae_image_sizes`` stays None → ``_prepare_edit_cond_kwargs`` raises
    ``TypeError: 'NoneType' object is not iterable``.

    This wrap detects the PIL-only path and calls
    ``preprocess_condition_image`` after upstream forward returns, using the
    PIL's own width/height as the condition-image size (mirroring what
    upstream would have done inside the image_path branch). No-op when
    ``condition_image`` is None (T2I) or when ``image_path`` is set (upstream
    already handled it). Idempotent.
    """
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

        # Only act on the PIL-only path: condition_image set, image_path None,
        # and vae_image_sizes not yet populated (upstream didn't call
        # preprocess_condition_image).
        condition_image = getattr(batch, "condition_image", None)
        image_path = getattr(batch, "image_path", None)
        vae_image_sizes = getattr(batch, "vae_image_sizes", None)
        if condition_image is None or image_path is not None or vae_image_sizes is not None:
            return batch

        # Mirror upstream's preprocess_condition_image entry (input_validation.py:131-165):
        # it needs a single PIL to read width/height, and calls
        # config.preprocess_vae_image(batch, self.vae_image_processor) which
        # populates batch.vae_image_sizes.
        img = condition_image[-1] if isinstance(condition_image, list) else condition_image
        condition_image_width = img.width
        condition_image_height = img.height
        batch.original_condition_image_size = (condition_image_width, condition_image_height)

        # Preserve the driver-pinned output dims. ``preprocess_condition_image``
        # overwrites batch.height/width with the source-image-derived VAE size
        # (~1024²) UNLESS ``extra["explicit_fields"]`` lists them. Upstream's
        # ``_explicit_fields`` (set in ``from_user_sampling_params_args``) is a
        # plain attribute that ``dataclasses.replace`` drops in
        # ``DiffGenerator.generate``'s per-prompt loop, so for the PIL path
        # (which skips the image_path branch that re-sets it) ``explicit_fields``
        # is empty here and the user's 384² gets clobbered to 1024². The
        # driver-authoritative ``initial_noise`` was created at 384², so a
        # clobbered 1024² makes ``maybe_pack_latents`` reshape against the
        # wrong grid → ``RuntimeError: shape '[1,16,64,2,64,2]' is invalid for
        # input of size 36864``. UniRL's adapter always pins height/width
        # (build_sampling), so save/restore them across the call.
        saved_height = batch.height
        saved_width = batch.width
        self.preprocess_condition_image(batch, server_args, condition_image_width, condition_image_height)
        batch.height = saved_height
        batch.width = saved_width
        return batch

    setattr(forward, _IVL_SENTINEL, True)
    IVL.forward = forward


# ------------------------------------------------------------------ #
# (1) SamplingParams field injection
# ------------------------------------------------------------------ #


def _make_dataclass_field(name: str, default, type_str: str):
    """Build a dataclasses.Field equivalent to ``name: type = default`` post-hoc."""
    f = field(default=default)
    f.name = name
    f.type = type_str
    # Mark as a real (init=True) dataclass field so fields()/replace()/asdict()
    # treat it like any source-declared field.
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
    """Register the four fields on SamplingParams and every live subclass.

    Each ``@dataclass`` subclass owns its ``__init__`` and ``__dataclass_fields__``
    (a subclass ``__init__`` does NOT chain to ``super().__init__``), so to make
    the new kwargs constructible on the concrete class actually used we register +
    wrap ``__init__`` on every class in the tree. Idempotent.
    """
    for cls in _iter_subclasses(SamplingParams):
        _register_and_wrap_init(cls)


def _register_and_wrap_init(cls) -> None:
    """Add the four fields to ``cls`` and wrap its ``__init__`` to accept them.

    Registration (``__dataclass_fields__`` + class-level default) makes the field
    visible to ``dataclasses.fields`` / ``replace`` / ``asdict``. The ``__init__``
    wrapper strips the four keys before the (kwarg-strict) generated ``__init__``
    runs, then re-applies them via ``object.__setattr__`` so construction --
    including the ``dataclasses.replace`` call inside upstream ``generate`` --
    accepts and round-trips them.
    """
    own_fields = cls.__dict__.get("__dataclass_fields__")
    if own_fields is None:
        # Not the class's own dict (inherited view) -- give it its own so the
        # injected entry is visible to fields(cls)/replace(cls-instance).
        own_fields = dict(getattr(cls, "__dataclass_fields__", {}))
        cls.__dataclass_fields__ = own_fields

    for name, (default, type_str) in _SP_INJECT_FIELDS.items():
        if name not in own_fields:
            own_fields[name] = _make_dataclass_field(name, default, type_str)
        # Class-level default so plain ``getattr(sp, name)`` works pre-construction
        # and ``_merge_with_user_params`` (reads ``getattr(type(self), name)``)
        # sees a sane default rather than raising.
        if name not in cls.__dict__:
            setattr(cls, name, default)

    orig_init = cls.__dict__.get("__init__")
    if orig_init is None or getattr(orig_init, _SP_INIT_SENTINEL, False):
        # Either inherits __init__ (will be reached via a patched ancestor's
        # wrapper -- but every dataclass defines its own, so this is rare) or is
        # already wrapped. Nothing to do.
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
    """AROUND-wrap the staticmethod that constructs the user SamplingParams.

    The model-specific subclass is imported by the registry *inside* this method
    (before it constructs the params), so re-running the field injection here --
    cheap and idempotent -- guarantees the concrete subclass is wrapped even when
    it was not imported at hijack time.
    """
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


# ------------------------------------------------------------------ #
# (2) Req.denoise_seeds injection
# ------------------------------------------------------------------ #


def _install_req_denoise_seeds(sb_mod) -> None:
    """Add ``denoise_seeds`` as a first-class field on upstream ``Req``.

    Upstream ``Req`` lacks it (``latents`` / ``sigmas`` / ``timesteps`` already
    exist). Without this, ``req.denoise_seeds = ...`` would be delegated by
    ``Req.__setattr__`` onto ``sampling_params`` (which gets ``dataclasses.replace``
    -d), rather than living on ``Req`` like the fork's ``Req.denoise_seeds``.
    ``Req.__init__`` iterates ``__dataclass_fields__`` for defaults, so a registered
    field gets a ``None`` default automatically.
    """
    Req = sb_mod.Req
    own_fields = Req.__dict__.get("__dataclass_fields__")
    if own_fields is None:  # pragma: no cover - Req is a dataclass, always present
        own_fields = dict(getattr(Req, "__dataclass_fields__", {}))
        Req.__dataclass_fields__ = own_fields
    if _REQ_FIELD not in own_fields:
        own_fields[_REQ_FIELD] = _make_dataclass_field(_REQ_FIELD, None, "list[str] | None")

    # Keep SAMPLING_PARAMS_FIELDS coherent: it is frozen at schedule_batch import.
    # Adding our SamplingParams fields to it is harmless and avoids the (unused in
    # our path) auto-create branch in Req.__setattr__ ever delegating these names.
    spf = getattr(sb_mod, "SAMPLING_PARAMS_FIELDS", None)
    if isinstance(spf, set):
        spf.update(_SP_INJECT_FIELDS)


# ------------------------------------------------------------------ #
# (3) prepare_request copy
# ------------------------------------------------------------------ #


def _wrap_prepare_request(utils_mod, SamplingParams) -> None:
    """AROUND-wrap ``prepare_request`` to copy the four IO fields onto the Req.

    Mirrors the fork: ``sigmas`` / ``timesteps`` (coerced to a float32 tensor) per
    ``entrypoints/utils.py``; ``initial_noise`` -> ``req.latents`` and
    ``denoise_seeds`` per ``diffusion_generator.generate``. Copying happens AFTER
    the original builds + validates the Req -- nothing in upstream
    ``prepare_request`` reads these fields, so post-hoc assignment is behaviour
    equivalent and keeps the wrapper a thin pass-through.
    """
    orig = utils_mod.prepare_request
    if getattr(orig, _PREP_SENTINEL, False):
        return

    def prepare_request(server_args, sampling_params, *args, **kwargs):
        import torch

        req = orig(server_args, sampling_params, *args, **kwargs)

        # Req.sigmas shadows SamplingParams.sigmas (both fields); assigning to the
        # Req field sets it on the Req directly.
        sigmas = getattr(sampling_params, "sigmas", None)
        if sigmas is not None:
            req.sigmas = sigmas

        # Req.timesteps is a Tensor downstream (handed to scheduler.set_timesteps);
        # coerce exactly as the fork does.
        timesteps = getattr(sampling_params, "timesteps", None)
        if timesteps is not None:
            req.timesteps = torch.as_tensor(timesteps, dtype=torch.float32)

        # Driver-authoritative x_T: LatentPreparationStage consumes Req.latents.
        initial_noise = getattr(sampling_params, "initial_noise", None)
        if initial_noise is not None:
            req.latents = initial_noise

        # Per-sample SDE noise grouping (Req field injected by this patch).
        denoise_seeds = getattr(sampling_params, "denoise_seeds", None)
        if denoise_seeds is not None:
            req.denoise_seeds = denoise_seeds

        # Edit-Plus source-image PIL. Req.condition_image is a real dataclass
        # field (schedule_batch.py:59), so this assignment lands on the Req
        # directly (not delegated to sampling_params). Upstream's
        # InputValidationStage then preprocesses the PIL (resize + VAE encode
        # → batch.image_latent). No-op when the adapter didn't set it (T2I).
        condition_image = getattr(sampling_params, "condition_image", None)
        if condition_image is not None:
            # Multi-prompt indexing: when the adapter emitted a list of PILs
            # (one per unique prompt), upstream's per-prompt loop in
            # ``DiffGenerator.generate`` carries the WHOLE list on every
            # per-prompt ``sampling_params`` (it indexes ``image_path`` but
            # not ``condition_image``). ``_wrap_diff_generator_generate``
            # stashed the list in thread-local state at the start of
            # ``generate``; index it per Req here, mirroring
            # ``image_paths_per_prompt[i]``. Single-prompt path (list len 1
            # or scalar PIL) is a passthrough.
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

    # CRITICAL: ``from ...entrypoints.utils import prepare_request`` binds the name
    # BY VALUE in importing modules at import time (e.g. ``diffusion_generator``,
    # whose ``generate()`` calls its own module-level ``prepare_request``). Patching
    # ``utils_mod.prepare_request`` alone therefore never reaches the real caller --
    # the Req would ship without sigmas / initial_noise / denoise_seeds. Rebind
    # every already-imported module that holds the original by value.
    import sys

    for _mod in list(sys.modules.values()):
        try:
            if getattr(_mod, "prepare_request", None) is orig:
                _mod.prepare_request = prepare_request
        except Exception:  # pragma: no cover - defensive
            pass
