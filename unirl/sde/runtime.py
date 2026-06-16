"""Shared SDE runtime entrypoints.

Three layers, all owned by this module:

1. **Pure math** — :func:`get_sigma_schedule` for the FlowMatch σ schedule.
   Static branch implemented here (diffusers' static path has issue #13243);
   dynamic branch delegates to diffusers (its dynamic path is bug-free).
   The dynamic μ is chosen by :meth:`FlowMatchSchedulePolicy.compute_mu` —
   the **single per-model override point**. Its default delegates to
   :func:`calculate_dynamic_mu` (linear in image_seq_len); FLUX.2-klein
   overrides it with an empirical μ that also depends on num_inference_steps.

2. **Schedule policy** — :class:`FlowMatchSchedulePolicy` is the model-owned
   schedule data (loaded once, constant per actor) the σ computation needs:
   shift, the 5 dynamic-shift knobs, vae_scale_factor and patch_size.
   (NB: "static" elsewhere in this module names the no-μ *shift branch*, a
   different axis from this once-loaded config.) :meth:`from_pretrained` reads
   the three diffusers-standard JSONs (``scheduler/scheduler_config.json``,
   ``transformer/config.json``, ``vae/config.json``) under a model
   checkpoint directory and assembles a policy. The loader is **pure I/O
   on small JSONs** — no model weights, no Bundle, no Pipeline. Available
   main-side regardless of whether the actor loaded a full Bundle, which
   is what lets sglang / vllm-omni engines compute σ without holding the
   model in memory.

3. **Glue** — :func:`ensure_req_sigmas` validates a ``RolloutReq`` and pins
   ``policy.compute_sigma(...)`` onto ``RolloutReq.sigmas`` (every rollout
   engine calls it at the top of its ``generate``).

Naming convention (a symbol's name tells you its layer):

- ``FlowMatchSchedulePolicy.compute_*`` are **methods** — model-aware
  behavior that reads the policy's own fields (``compute_mu`` → the
  per-model μ; ``compute_sigma`` → the full per-request σ).
- free ``get_sigma_schedule`` / ``calculate_dynamic_mu`` are **stateless
  math primitives** — fully-resolved scalars in, no policy state.
- ``ensure_req_sigmas`` is **request glue** — it operates on a RolloutReq.

Ownership map (kept explicit so reading the code doesn't require
following six getattr chains)::

    Policy        owned by  MODEL CHECKPOINT (scheduler/transformer/vae JSONs)
    Params (T,H,W) owned by REQUEST (RolloutReq.sampling_params.diffusion)
    σ computation owned by THIS MODULE (pure function)
    σ flow        carried by RolloutReq.sigmas (set by engine, read by
                  pipeline / worker / replay; verified end-to-end by
                  unirl.rollout.engine.sigma_verify)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from unirl.types.sampling import get_diffusion_params

logger = logging.getLogger(__name__)


# ===========================================================================
# Layer 1 — pure math
# ===========================================================================


def get_sigma_schedule(
    num_steps: int,
    shift: float = 3.0,
    device: Optional[torch.device] = None,
    *,
    mu: Optional[float] = None,
    time_shift_type: str = "exponential",
    shift_terminal: Optional[float] = None,
) -> torch.Tensor:
    """Compute the FlowMatch σ schedule of length ``num_steps + 1``.

    ``mu`` is the static↔dynamic **mode switch**:

    - ``mu is None`` → **static**: SD3-paper shift applied once,
      ``t' = shift·t / (1 + (shift-1)·t)``. Computed here instead of
      delegated because diffusers' ``use_dynamic_shifting=False`` path
      double-applies the shift (#13243). ``time_shift_type`` is unused.
    - ``mu is not None`` → **dynamic**: delegate to diffusers, passing the
      ``linspace(1, 1/T)`` base grid every real FlowMatch pipeline uses
      (omitting ``sigmas=`` degenerates diffusers' small-σ tail to
      ``≈ 1/num_train_timesteps``). ``shift`` is unused.

    ``shift_terminal`` (dynamic branch only; Qwen-Image ships ``0.02``,
    SD3/Flux ship ``null``): forwarded into the diffusers scheduler, whose
    ``set_timesteps`` applies the canonical ``stretch_shift_to_terminal``
    after the mu shift — exactly the official inference order. No current
    static-shift model uses it, so the static branch fails fast rather than
    growing an untested hand-rolled stretch.
    """
    if mu is None:
        # DELETE-WHEN: diffusers #13243 fixed → drop this branch and route
        # static through diffusers too (symmetric with the dynamic branch).
        if shift_terminal is not None:
            raise ValueError(
                f"get_sigma_schedule: shift_terminal={shift_terminal!r} is only "
                f"supported on the dynamic branch (mu is not None); no static-"
                f"shift model declares it. Pass mu= or drop shift_terminal."
            )
        t = torch.linspace(1.0, 0.0, num_steps + 1)
        sigmas = (shift * t) / (1 + (shift - 1) * t)
    else:
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            time_shift_type=time_shift_type,
            shift_terminal=shift_terminal,
        )
        base_sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(num_inference_steps=num_steps, sigmas=base_sigmas, mu=mu)
        sigmas = scheduler.sigmas
    if device is not None:
        sigmas = sigmas.to(device)
    return sigmas


def calculate_dynamic_mu(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Linear interpolation of dynamic-shift μ from image sequence length.

    Mirrors diffusers' ``calculate_shift`` used by SD3 / Flux pipelines.
    This is the **default** μ formula: :meth:`FlowMatchSchedulePolicy.compute_mu`
    calls it, and a model subclass overrides ``compute_mu`` when its μ differs
    (e.g. FLUX.2-klein's empirical μ). Feed the result into
    :func:`get_sigma_schedule` via ``mu=...``.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


# ===========================================================================
# Layer 2 — schedule policy (model-owned config + behavior, from the checkpoint)
# ===========================================================================


def _read_json(path: Path) -> Optional[dict]:
    """Read a JSON file; return ``None`` on any failure (missing / unreadable)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, OSError):
        return None


def _vae_scale_factor_from_block_out_channels(block_out_channels: Any) -> Optional[int]:
    """Derive ``vae_scale_factor`` from ``block_out_channels`` length.

    Mirrors diffusers' convention
    (``2 ** (len(vae.config.block_out_channels) - 1)``, see
    ``pipeline_stable_diffusion_3.py:219`` and ``pipeline_flux.py:209``).
    Returns ``None`` for malformed inputs.
    """
    try:
        n = len(block_out_channels)
        if n < 1:
            return None
        return 2 ** (n - 1)
    except TypeError:
        return None


def _normalize_patch_size(value: Any, default: int) -> int:
    """Coerce a raw ``patch_size`` config value to a single spatial int.

    Some video transformers declare ``patch_size`` as a 3D
    ``[t_patch, h_patch, w_patch]`` list (e.g. diffusers'
    ``WanTransformer3DModel`` ships ``[1, 2, 2]``); the
    dynamic-shifting math here only consumes the spatial patch
    (``image_seq_len = (H // ... // patch_size) * (W // ... // patch_size)``,
    which assumes ``h_patch == w_patch``). Scalar inputs pass through
    unchanged; list/tuple inputs take the last element (the W patch),
    matching the canonical ``h == w`` convention. ``None`` falls back to
    ``default``.

    Without this normalization a checkpoint with list-valued
    ``patch_size`` would raise ``TypeError: int() argument must be a
    string, a bytes-like object or a real number, not 'list'`` at
    :meth:`FlowMatchSchedulePolicy.from_pretrained` time — even for
    static-only policies that never read the field at sample time.
    """
    if value is None:
        return int(default)
    if isinstance(value, (list, tuple)):
        if not value:
            return int(default)
        return int(value[-1])
    return int(value)


def _normalize_shift_terminal(value: Any) -> Optional[float]:
    """Coerce a raw ``shift_terminal`` config value to ``Optional[float]``.

    ``scheduler_config.json`` ships JSON ``null`` for models without the
    terminal stretch (SD3/Flux) and a float for those with it (Qwen-Image:
    ``0.02``). Diffusers gates the stretch on truthiness, so ``0``/``0.0``
    means disabled — normalize falsy to ``None`` to keep one disabled
    spelling throughout the policy.
    """
    if not value:
        return None
    return float(value)


@dataclass
class FlowMatchSchedulePolicy:
    """The model-owned σ schedule policy. Loaded once per actor.

    Built either from a pretrained checkpoint directory
    (:meth:`from_pretrained`) or from explicit fields
    (:meth:`static_only`). It is **lightweight and pickleable** —
    pass-by-value across Ray IPC, no Bundle / model weights required to
    construct it; its only behavior is the σ math
    (:meth:`compute_mu` / :meth:`compute_sigma`).

    Field semantics
    ---------------
    ``shift``: static FlowMatch time-shift. Per-model defaults: SD3=3.0,
    Flux=1.0, Wan=5.0, HunyuanVideo=1.0, HunyuanImage3=3.0. Always wins
    over ``scheduler_config.shift`` (user-configured override).

    ``use_dynamic_shifting``, ``base_shift``, ``max_shift``,
    ``base_image_seq_len``, ``max_image_seq_len``, ``time_shift_type``:
    dynamic-shift block. Sourced from
    ``<pretrained>/scheduler/scheduler_config.json``. When
    ``use_dynamic_shifting=True``, :meth:`compute_sigma` derives μ from
    image_seq_len (via :meth:`compute_mu`) and delegates to diffusers'
    dynamic branch; otherwise these fields are ignored.

    ``shift_terminal``: terminal-stretch target (Qwen-Image: ``0.02``;
    SD3/Flux: ``null``). Also sourced from ``scheduler_config.json``;
    applied by diffusers after the dynamic shift. ``None`` disables it —
    byte-identical schedules for every model that doesn't declare it.

    ``vae_scale_factor``, ``patch_size``: latent-grid divisors used in
    image_seq_len = ``(H // vae_scale_factor // patch_size) * (W // ...)``.
    Sourced from ``<pretrained>/vae/config.json`` and
    ``<pretrained>/transformer/config.json``. Only used in dynamic
    branch.
    """

    shift: float = 3.0
    use_dynamic_shifting: bool = False
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_image_seq_len: int = 256
    max_image_seq_len: int = 4096
    time_shift_type: str = "exponential"
    shift_terminal: Optional[float] = None
    vae_scale_factor: int = 8
    patch_size: int = 2

    def compute_mu(self, image_seq_len: int, num_inference_steps: int) -> float:
        """Dynamic-shift μ for this policy — the single per-model override point.

        Default delegates to :func:`calculate_dynamic_mu` (linear in
        ``image_seq_len``; ``num_inference_steps`` is unused in the base
        formula). Override in a model-specific subclass whose μ differs —
        e.g. FLUX.2-klein's empirical μ depends on **both** ``image_seq_len``
        and ``num_inference_steps`` (see ``Flux2KleinSchedulePolicy``). Only
        the μ value is model-specific; the schedule application (base grid +
        diffusers time-shift) stays shared in :meth:`compute_sigma`.
        """
        return calculate_dynamic_mu(
            image_seq_len,
            base_seq_len=self.base_image_seq_len,
            max_seq_len=self.max_image_seq_len,
            base_shift=self.base_shift,
            max_shift=self.max_shift,
        )

    def compute_sigma(
        self,
        *,
        num_inference_steps: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Apply this policy to a request's ``(T, H, W)`` → σ tensor ``[T+1]``.

        - static (``use_dynamic_shifting=False``): uses ``shift`` only.
        - dynamic: derive ``image_seq_len`` from ``(H, W)`` via
          ``vae_scale_factor`` / ``patch_size``, take μ from
          :meth:`compute_mu` (the per-model override point), then apply the
          diffusers dynamic shift.

        The stateless math beyond this point lives in the free functions
        :func:`get_sigma_schedule` / :func:`calculate_dynamic_mu`.
        """
        if not self.use_dynamic_shifting:
            return get_sigma_schedule(num_inference_steps, self.shift, device, shift_terminal=self.shift_terminal)
        latent_h = int(height) // int(self.vae_scale_factor)
        latent_w = int(width) // int(self.vae_scale_factor)
        image_seq_len = (latent_h // int(self.patch_size)) * (latent_w // int(self.patch_size))
        mu = self.compute_mu(image_seq_len, num_inference_steps)
        return get_sigma_schedule(
            num_inference_steps,
            self.shift,
            device,
            mu=mu,
            time_shift_type=self.time_shift_type,
            shift_terminal=self.shift_terminal,
        )

    @classmethod
    def static_only(cls, shift: float) -> "FlowMatchSchedulePolicy":
        """Build a static-shift-only policy. Use when no pretrained dir is
        available (tests, ad-hoc smoke runs)."""
        return cls(shift=float(shift), use_dynamic_shifting=False)

    @classmethod
    def _dynamic_from_overrides(
        cls,
        shift: float,
        overrides: Optional[Dict[str, Any]],
        path: Any,
    ) -> "FlowMatchSchedulePolicy":
        """Construct a dynamic-shift policy from an explicit overrides dict.

        Helper for :meth:`from_pretrained` when ``require_dynamic=True``
        and the pretrained checkpoint isn't locally readable (e.g. HF
        repo ID like ``Qwen/Qwen-Image``). The model's Pipeline is
        responsible for passing its canonical dynamic-shift fields in
        ``overrides``; if it didn't, raise loudly so the σ contract
        bug surfaces at engine init instead of at first rollout.
        """
        if not overrides:
            raise RuntimeError(
                f"FlowMatchSchedulePolicy.from_pretrained: caller declared "
                f"require_dynamic=True for path={path!r} but provided no "
                f"dynamic_overrides. The checkpoint isn't locally readable "
                f"so we can't load scheduler_config.json, and without "
                f"explicit dynamic fields we'd silently produce a static "
                f"policy (which mis-shifts dynamic-shift models like "
                f"Qwen-Image). Pre-download the scheduler/scheduler_config.json "
                f"from HF Hub OR have the model's Pipeline.build_schedule_policy "
                f"pass dynamic_overrides with use_dynamic_shifting=True + "
                f"base_shift / max_shift / base_image_seq_len / max_image_seq_len / "
                f"time_shift_type (+ shift_terminal where the model declares it) fields."
            )
        defaults = cls()
        return cls(
            shift=float(shift),
            use_dynamic_shifting=True,
            base_shift=float(overrides.get("base_shift", defaults.base_shift)),
            max_shift=float(overrides.get("max_shift", defaults.max_shift)),
            base_image_seq_len=int(overrides.get("base_image_seq_len", defaults.base_image_seq_len)),
            max_image_seq_len=int(overrides.get("max_image_seq_len", defaults.max_image_seq_len)),
            time_shift_type=str(overrides.get("time_shift_type", defaults.time_shift_type)),
            shift_terminal=_normalize_shift_terminal(overrides.get("shift_terminal", defaults.shift_terminal)),
            vae_scale_factor=int(overrides.get("vae_scale_factor", defaults.vae_scale_factor)),
            patch_size=_normalize_patch_size(overrides.get("patch_size"), defaults.patch_size),
        )

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path, None],
        *,
        shift: float,
        require_dynamic: bool = False,
        dynamic_overrides: Optional[Dict[str, Any]] = None,
    ) -> "FlowMatchSchedulePolicy":
        """Build a policy by reading the diffusers-standard JSON layout.

        Tries three files under ``path``::

            <path>/scheduler/scheduler_config.json   → dynamic-shift fields
            <path>/transformer/config.json           → patch_size
            <path>/vae/config.json                   → vae_scale_factor

        Missing files / missing keys fall back to dataclass defaults;
        the scheduler JSON specifically gets a ``logger.warning`` (it
        carries the dynamic-shift block, so silent fallback there
        would be a real bug for dynamic-shift models). The ``shift``
        arg always wins over any ``scheduler_config.shift`` (some
        checkpoints ship with stale shift values).

        Path resolution
        ---------------
        - ``path is None`` → :meth:`static_only` (explicit opt-in).
        - ``path`` doesn't exist locally:
            - If ``require_dynamic=False`` (default): fall back to
              :meth:`static_only` with a debug log. **Correct for
              static-shift HF repo IDs** like
              ``stabilityai/stable-diffusion-3.5-medium``.
            - If ``require_dynamic=True``: caller has declared this
              model NEEDS dynamic shifting (e.g. Qwen-Image). Use
              ``dynamic_overrides`` if provided; otherwise RAISE so the
              error surfaces at engine init instead of silently shipping
              wrong σ schedules.
        - ``path`` is an existing local directory → read JSONs.

        ``require_dynamic`` + ``dynamic_overrides`` were added to fix the
        silent fallback-to-static for HF-repo-ID checkpoints whose model
        config declared dynamic shifting (the 2026-05-18 review's Phase
        I.4). Each Pipeline's ``build_schedule_policy()`` knows its own
        dynamic-shift posture and passes the right hints.
        """
        # Local JSON dir unreadable — either no path given, or an HF repo ID
        # not yet on disk. Both fall back the same way: require_dynamic →
        # build from overrides (raises if absent); otherwise static-only.
        root = Path(path) if path is not None else None
        if root is None or not root.exists():
            if require_dynamic:
                return cls._dynamic_from_overrides(shift, dynamic_overrides, path)
            if root is not None:
                logger.debug(
                    "FlowMatchSchedulePolicy.from_pretrained: %s does not exist "
                    "locally (likely an HF repo ID — bundle.from_pretrained will "
                    "resolve it). Falling back to static_only(shift=%s).",
                    root,
                    shift,
                )
            return cls.static_only(shift)

        defaults = cls()  # canonical default values
        sched_path = root / "scheduler" / "scheduler_config.json"
        sched = _read_json(sched_path)
        if sched is None:
            # Dynamic-shift information lives in this JSON; silent
            # fallback to static would mis-shift a dynamic-shift model
            # (caught by ``verify_engine_used_sigmas`` at rollout time
            # but worth surfacing here so the cause is obvious in
            # logs).
            logger.warning(
                "FlowMatchSchedulePolicy.from_pretrained: %s not found; "
                "dynamic-shift fields default to static-only behavior. "
                "If the model wants dynamic shift, σ will drift and the "
                "drift assert will raise at the first rollout.",
                sched_path,
            )
            sched = {}
        trans = _read_json(root / "transformer" / "config.json") or {}
        vae = _read_json(root / "vae" / "config.json") or {}

        vae_scale_factor = _vae_scale_factor_from_block_out_channels(vae.get("block_out_channels"))
        return cls(
            shift=float(shift),
            use_dynamic_shifting=bool(sched.get("use_dynamic_shifting", defaults.use_dynamic_shifting)),
            base_shift=float(sched.get("base_shift", defaults.base_shift)),
            max_shift=float(sched.get("max_shift", defaults.max_shift)),
            base_image_seq_len=int(sched.get("base_image_seq_len", defaults.base_image_seq_len)),
            max_image_seq_len=int(sched.get("max_image_seq_len", defaults.max_image_seq_len)),
            time_shift_type=str(sched.get("time_shift_type", defaults.time_shift_type)),
            shift_terminal=_normalize_shift_terminal(sched.get("shift_terminal", defaults.shift_terminal)),
            vae_scale_factor=int(vae_scale_factor or defaults.vae_scale_factor),
            patch_size=_normalize_patch_size(trans.get("patch_size"), defaults.patch_size),
        )


# ===========================================================================
# Layer 3 — request glue (pin σ onto a RolloutReq)
# ===========================================================================


def ensure_req_sigmas(req: Any, policy: FlowMatchSchedulePolicy) -> None:
    """Compute and pin the σ schedule onto ``req.sigmas``.

    Every rollout engine calls this once at the top of ``generate(req)``.

    ``req`` must expose ``req.sigmas`` (read/write) and
    ``req.sampling_params`` with diffusion params containing
    ``num_inference_steps`` / ``height`` / ``width`` keys.

    All three keys are **required** —  silent ``height=1024`` /
    ``width=1024`` defaults would mis-derive μ for dynamic-shift models
    when the request actually rendered at a different resolution
    (e.g. WAN T2V at 480×832). The driver always sets all three at request
    construction (the trainer's ``_build_req``); absence means a wiring bug.
    """
    if req.sigmas is not None:
        return
    diffusion = get_diffusion_params(req.sampling_params)
    if diffusion is None:
        raise ValueError(
            "ensure_req_sigmas: req.sampling_params must contain diffusion params for σ schedule computation."
        )
    req.sigmas = policy.compute_sigma(
        num_inference_steps=int(diffusion.num_inference_steps),
        height=int(diffusion.height),
        width=int(diffusion.width),
    )


__all__ = [
    # Layer 2 — the per-model schedule object (engines build it; models subclass)
    "FlowMatchSchedulePolicy",
    # Layer 3 — request glue (rollout engines call this)
    "ensure_req_sigmas",
    # Layer 1 — stateless math primitives (used directly by tests / advanced callers)
    "get_sigma_schedule",
    "calculate_dynamic_mu",
]
