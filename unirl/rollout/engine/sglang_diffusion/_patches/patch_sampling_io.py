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
from dataclasses import field

logger = logging.getLogger(__name__)

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
}

# Sentinels.
_SP_INIT_SENTINEL = "_unirl_sampling_io_init"
_PREP_SENTINEL = "_unirl_sampling_io_prepare"
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
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_json_safe(v) for v in obj]
        return orig(obj)

    _json_safe._unirl_tensor_safe = True  # type: ignore[attr-defined]
    sp_mod._json_safe = _json_safe


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
