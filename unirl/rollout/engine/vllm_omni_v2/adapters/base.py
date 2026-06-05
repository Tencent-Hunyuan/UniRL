"""Driver-side ``RolloutReq``↔``RolloutResp`` conversion: the adapter ABC + registry.

A thin top ABC (registry + boilerplate with sensible defaults) over per-output-
shape base adapters (``hi3_ar_dit`` / ``hi3_ar_only`` / ``dit_image`` /
``dit_video``) that hold the conversion logic as overridable methods. Concrete adapters
override only what differs and self-register by ``modality`` key — the same
axis v1 branched on inline. Selected once at engine construction via
:func:`get_adapter`.

Pure: never imports vllm-omni — adapters consume the seam's ``OmniRawResult``
protocol and emit :class:`GenerateCall` intent; tokenization reaches the
runtime through the injected ``tokenize_fn`` (the seam's ``tokenize_prompt``
verb). The adapter is bound to the engine config + model config at
construction so its conversion methods don't thread them.

The per-modality topology knobs (which stage YAML, env/boot quirks, LoRA
transport) are class attributes here — lifted from v1 ``engine.py``'s
``_HI3_MODALITIES`` / ``_DIT_BEARING_MODALITIES`` / ``_HI3_MULTI_GPU_MODALITIES``
frozensets so the engine never branches on a modality string.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.vllm_omni_v2.backends import GenerateCall, OmniRawResult
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, type["ModelAdapter"]] = {}


def register_adapter(key: str):
    """Class decorator: register an adapter under its ``modality`` key."""

    def deco(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        require(
            key not in _REGISTRY,
            f"adapter key {key!r} already registered by {_REGISTRY.get(key)!r}",
        )
        _REGISTRY[key] = cls
        cls.modality = key
        return cls

    return deco


def get_adapter(key: str) -> type["ModelAdapter"]:
    """Look up the adapter class for a ``modality`` key."""
    require(
        key in _REGISTRY,
        f"unknown modality {key!r}; registered: {sorted(_REGISTRY)}",
    )
    return _REGISTRY[key]


def registered_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------- #
# ABC
# --------------------------------------------------------------------------- #


class ModelAdapter(ABC):
    """Thin ABC: registry key + topology knobs + the two conversion seams.

    The conversion *logic* lives on the per-shape base adapters; this ABC
    declares the boilerplate every adapter shares (boot intent, schedule
    policy, validation) and the two abstract methods the engine drives.
    """

    modality: str = ""

    # ---- topology knobs (one line per v1 frozenset membership) ----
    #: The stage-config YAML this modality boots (+ where it ships).
    stage_yaml: str = ""
    stage_yaml_source: str = "local"
    #: ``Omni(mode=...)`` kwarg; ``None`` omits it (v1 engine.py:377-378).
    omni_mode: Optional[str] = None
    #: Request carries diffusion params → pin σ via ``ensure_req_sigmas``
    #: (v1 ``_DIT_BEARING_MODALITIES``; AR-only requests would raise on it).
    needs_sigmas: bool = True
    #: Driver-side tokenizer for ``build_prompt_tokens`` (v1 engine.py:322 —
    #: everything except sd35_t2i / t2v, including dit_recaption, which loads
    #: one without using it; kept for parity).
    needs_driver_tokenizer: bool = True
    #: HI3 AR-prelude family: pass ``lora_request`` as a top-level
    #: ``Omni.generate`` kwarg (requires the passthrough patch; v1
    #: ``_HI3_MODALITIES`` — see patches/__init__ for the DELETE-WHEN).
    ar_lora_passthrough: bool = False
    #: HI3 multi-GPU stages: clear ``CUDA_VISIBLE_DEVICES`` before boot so
    #: vllm-omni pins stages to their yaml ``runtime.devices`` (v1
    #: ``_HI3_MULTI_GPU_MODALITIES``). ⚠️ Safe only when the engine is wired
    #: as a single multi-GPU actor — see the v1 colocate-landmine note.
    clear_cuda_visible: bool = False
    #: Re-push LoRA after wake via the byte-copy transport (TP>1 stages where
    #: the zero-copy handle crashes ranks 2..N; v1 wake branch).
    lora_copy_transport: bool = False

    def __init__(
        self,
        config: Any,
        model_config: Any,
        *,
        strategy: Any = None,
        tokenize_fn: Optional[Callable[..., List[int]]] = None,
    ) -> None:
        self.cfg = config
        self.model_config = model_config
        self.tokenize_fn = tokenize_fn
        self._sde_label = self.resolve_sde_label(strategy)
        self.validate()

    # ---- SDE label (parity no-op; injection point only) ----
    @staticmethod
    def resolve_sde_label(strategy: Any) -> Optional[str]:
        """Deliberately ``None``: vllm-omni rides raw ``eta`` + ``sde_indices``
        through ``extra_args`` (the worker pipeline applies SDE on those
        steps), unlike sglang_diffusion's kernel-label selection. This hook
        exists so a future kernel-label path has its seam; do not "complete"
        it — that would break v1 parity.
        """
        del strategy
        return None

    # ---- boot intent (consumed by ``config.server_intent``) ----
    def boot_kwargs(self) -> Dict[str, Any]:
        """Model-specific boot intent beyond the generic config spelling.

        The generic pieces (model_path, enable_sleep_mode, timeouts, the
        ``omni_extra`` escape hatch, ports) are the config's job; this conveys
        only what the modality knows: which stage YAML, the driver-tokenizer
        need, the CVD quirk, and the ``Omni(mode=...)`` kwarg.
        """
        require(bool(self.stage_yaml), f"{type(self).__name__} must set stage_yaml")
        kwargs: Dict[str, Any] = {
            "stage_yaml": self.stage_yaml,
            "stage_yaml_source": self.stage_yaml_source,
            "needs_driver_tokenizer": bool(self.needs_driver_tokenizer),
            "clear_cuda_visible": bool(self.clear_cuda_visible),
        }
        if self.omni_mode is not None:
            kwargs["mode"] = self.omni_mode
        return kwargs

    # ---- σ schedule policy (generic FlowMatch; v1 engine.py:420-427) ----
    def schedule_policy(self) -> Any:
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        mc = self.model_config
        return FlowMatchSchedulePolicy.from_pretrained(
            self.cfg.model_path,
            shift=float(mc.shift),
            require_dynamic=bool(getattr(mc, "use_dynamic_shifting", False)),
            dynamic_overrides=getattr(mc, "dynamic_shift_overrides", None),
        )

    # ---- construction-time validation (v1 engine.py:404-409) ----
    def validate(self) -> None:
        mc = self.model_config
        require(
            mc is not None and hasattr(mc, "shift"),
            f"{type(self).__name__} requires model_config.shift; got "
            f"{type(mc).__name__}. Use a registered model preset "
            f"(e.g. ``sd3``, ``wan21``, ``wan22``, ``hunyuan_image3``).",
        )

    # ---- per-request validation (ports v1 ``_validate_request``) ----
    def validate_request(self, req: RolloutReq) -> None:
        """Modality-specific request gate; default accepts everything."""

    # ---- the two conversion seams the engine drives ----
    @abstractmethod
    def build_inputs(self, req: RolloutReq) -> List[GenerateCall]:
        """Translate a ``RolloutReq`` into the seam's generate calls.

        Normally one call carrying the whole batch; ``dit_recaption`` returns
        N seeded single-prompt calls.
        """

    @abstractmethod
    def build_response(self, req: RolloutReq, per_request: List[List[OmniRawResult]]) -> RolloutResp:
        """Translate the seam's per-request-grouped results into a ``RolloutResp``."""

    # ---- shared DiT-kwargs boilerplate (every DiT-bearing shape calls it) ----
    def core_diff_kwargs(self, req: RolloutReq, diff_params: Any) -> Dict[str, Any]:
        """The diffusion sampling kwargs common to every DiT stage.

        Every value reads off the request's typed ``DiffusionSamplingParams``
        — the engine keeps no sampling defaults. ``eta`` rides as a typed
        first-class field; ``guidance_scale_provided`` marks the explicit CFG
        choice; trajectory latents are always requested (dense — replay needs
        ``x_t`` at every slot).
        """
        from unirl.rollout.engine.vllm_omni_v2.utils.sigmas import sigmas_list_from_req

        num_inference_steps = int(diff_params.num_inference_steps)
        diff_kwargs: Dict[str, Any] = dict(
            height=int(diff_params.height),
            width=int(diff_params.width),
            num_inference_steps=num_inference_steps,
            guidance_scale=float(diff_params.guidance_scale),
            guidance_scale_provided=True,
            eta=float(diff_params.eta),
            return_trajectory_latents=True,
            return_trajectory_decoded=False,
            num_outputs_per_prompt=1,
        )
        sigmas = sigmas_list_from_req(req, num_inference_steps)
        if sigmas is not None:
            diff_kwargs["sigmas"] = sigmas
        return diff_kwargs

    @staticmethod
    def sde_extra_args(diff_params: Any) -> Dict[str, Any]:
        """Sparse SDE step indices, normalized for the ``extra_args`` channel."""
        extra_args: Dict[str, Any] = {}
        sde_indices = getattr(diff_params, "sde_indices", None)
        if sde_indices is not None:
            extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})
        return extra_args


__all__ = [
    "ModelAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
